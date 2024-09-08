from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import List
import haggis.npy_util
import matlab.engine
from matlab.engine import MatlabEngine
import matplotlib.collections
import matplotlib.pyplot as plt
import sofar
import math
import numpy as np
import src
import pyfar as pf
import scipy
from astropy.coordinates import cartesian_to_spherical, spherical_to_cartesian
from src.orientations import view_up_to_matrix
from src.utils import (
    amp2db,
    cartesian_std,
    closest_even_integer,
    energy2db,
    find_onset_sample,
    sample_amp_repr,
    mygrid,
    oriented_arrow,
    smoothed_ir_level,
)
import haggis
import matplotlib
import os
import json
import polars as pl


def matlab_with_SDMtools() -> MatlabEngine:
    eng = matlab.engine.start_matlab(background=False)

    project_path = Path(src.__path__[0]).parent
    eng.addpath(eng.genpath(str(project_path)))

    # SDMtools_path = str(project_path.joinpath("matlab/SDMtools").resolve())
    # eng.addpath(eng.genpath(SDMtools_path))

    eng.SOFAstart(nargout=0)
    return eng


# p: distance to origin, in direction of normal vector (i.e. if normal vector points towards origin, should be negative)
# n: 3-element ndarray, normal vector of length 1 pointing outside the room
Plane = namedtuple("Plane", "p n")


def reflection_matrix_from_plane(plane: Plane):
    np.testing.assert_array_max_ulp(np.linalg.norm(plane.n), 1)
    n = np.expand_dims(plane.n, 1)
    return np.identity(3) - 2 * n @ n.transpose()


def estimate_mixing_time_95(volume):
    """Mixing Time according to Lindau2012, tmp95, in seconds"""
    return (0.0117 * volume + 50.1) / 1e3


class SdmIsmAnalysis:
    def __init__(
        self,
        # reusable properties of the analysis between different loudspeakers/microphones are set here
        eng: MatlabEngine | None,
        c=343,
        planes: List[Plane] = None,
        # statistically speaking, reflections are diffuse after 2-3 reflections (see Lindau), but assignment at the end of mixing time will otherwise be biased towards the outer image sources
        max_reflection_order=5,
        min_reflection_distance=0.3,
        mixing_time=0.1,  # seconds
        brir_duration=1.2,  # seconds, set to duration where noise would dominate
        post_quantization_window_size=47,  # samples
        filter_size=512,
        no_doa_directions=50,
        HRTF_path="data/external/RWTH-2020-11307/Kemar_HRTF_sofa.sofa",
    ) -> None:
        if eng is None:
            self.eng = matlab_with_SDMtools()
        else:
            self.eng = eng
        assert c > 0
        self.c = c

        assert len(planes) > 0
        self.planes = planes

        assert max_reflection_order > 0
        assert max_reflection_order <= 5
        self.max_reflection_order = max_reflection_order
        assert min_reflection_distance >= 0
        self.min_reflection_distance = min_reflection_distance
        self.mixing_time = mixing_time

        self.brir_duration = brir_duration

        # assert odd number
        assert (post_quantization_window_size / 2) % 1 == 0.5
        self.post_quantization_window_size = post_quantization_window_size

        assert filter_size > 0
        assert filter_size % 2 == 0  # even
        self.filter_size = filter_size

        assert no_doa_directions > 0
        self.no_doa_directions = no_doa_directions

        self.HRTF_path = HRTF_path

    def read_sofa(self, input_sofa_path: Path):
        self.input_sofa_path = input_sofa_path
        self.input_sofa = sofar.read_sofa(self.input_sofa_path, verify=False)

        # fix for zenodo.10450779 dataset
        if self.input_sofa.GLOBAL_RoomType == "":
            self.input_sofa.GLOBAL_RoomType = "reverberant"
        self.input_sofa.verify()

        try:  # never change sample rate
            assert self.fs == self.input_sofa.Data_SamplingRate
        except AttributeError:
            self.fs = self.input_sofa.Data_SamplingRate

        return self

    def calculate_doa(
        self, source_index: int, direct_sound_onset_sample=1000, shift_window_size=48
    ):
        assert source_index >= 0
        self.source_index = source_index

        # Source Orientation
        try:
            source_view = self.input_sofa.SourceView[self.source_index]
        except IndexError:
            source_view = self.input_sofa.SourceView[0]

        try:
            source_up = self.input_sofa.SourceUp[self.source_index]
        except IndexError:
            source_up = self.input_sofa.SourceUp[0]

        self.source_orientation = view_up_to_matrix(source_view, source_up)

        # Listener Orientation
        assert self.input_sofa.ListenerView.shape == (1, 3)
        assert self.input_sofa.ListenerUp.shape == (1, 3)
        self.listener_orientation = view_up_to_matrix(
            self.input_sofa.ListenerView[0], self.input_sofa.ListenerUp[0]
        )
        assert self.listener_orientation.shape == (3, 3)

        # align direct sound to certain, in dataset this is done according to ISO but not effectively
        self.direct_sound_onset_sample = direct_sound_onset_sample
        irs = pf.Signal(self.input_sofa.Data_IR[self.source_index, :, :], self.fs)
        onset_sample = find_onset_sample(irs[6])  # use main mic at index 6
        shift = self.direct_sound_onset_sample - onset_sample
        N = irs.n_samples
        if shift >= 0:  # right shift
            # window (non-cropping) to create zeros on the right and create smooth fade in on the left from zeros
            windowed = pf.dsp.time_window(
                irs,
                (
                    0,
                    shift_window_size,
                    N - shift_window_size - 1 - shift,
                    N - 1 - shift,
                ),
            )
        else:  # left shift
            # window (non-cropping) to create zeros on the left and create fade out on the right
            windowed = pf.dsp.time_window(
                irs,
                (-shift, -shift + shift_window_size, N - shift_window_size - 1, N - 1),
            )
        shifted = pf.dsp.time_shift(windowed, shift)

        # 1st dimension: IR length, 2nd dimension: no. of mics
        Raw_RIR = np.transpose(shifted.time)
        # main mic at index 6
        self.ir = shifted[6]

        self.SRIR_data = self.eng.create_SRIR_data(
            "MicArray",
            "SDM-TU-ILMENAU",
            "Room",
            "NA",
            "SourcePos",
            "NA",
            "ReceiverPos",
            "NA",
            "fs",
            self.fs,
            "MixingTime",
            self.mixing_time,
            "DOASmooth",
            16,
            "Denoise",
            False,  # I don't know how to denoise the diffuse part, so don't do it at all
            "FilterRaw",  # unused, we drive this process ourselves in `additional_processing_after_SDMPar.m`
            1,
            "AlignDOA",
            0,
            "Raw_RIR",
            Raw_RIR,
        )

        # Initialize SDM analysis struct (from SDM Toolbox)
        self.SDM_Struct = self.eng.createSDMStruct(
            "c",
            self.c,
            "fs",
            self.SRIR_data["fs"],
            "micLocs",
            self.SRIR_data["ArrayGeometry"],
            "winLen",
            62,
        )

        self.SRIR_data["DOA"] = self.eng.SDMPar(
            self.SRIR_data["Raw_RIR"], self.SDM_Struct
        )

        self.listener_position = self.input_sofa.ListenerPosition[self.source_index]
        self.source_position = self.input_sofa.SourcePosition[self.source_index]

        raw_doa = np.array(self.SRIR_data["DOA"])
        # align doa and direct sound in space (DS always starts at 1000 samples, no matter the distance)
        x = raw_doa[:, 0]
        y = raw_doa[:, 1]
        z = raw_doa[:, 2]
        r, lat, lon = cartesian_to_spherical(x, y, z)
        self.direct_sound_distance = np.linalg.norm(
            self.listener_position - self.source_position
        )
        self.direct_sound_distance_samples = int(
            round(self.fs * self.direct_sound_distance / self.c)
        )
        radius_change_samples = (
            self.direct_sound_distance_samples - self.direct_sound_onset_sample
        )

        # Let's say an imsrc should arrive at 3200 samples absolute time (0 when listener and source coincide)
        # Then you have to add this value to get the time when they actually arrive, let's say 3732
        self.sample_offset = -radius_change_samples

        self.radius_change = radius_change_samples / self.fs * self.c
        r_red = r + self.radius_change
        x2, y2, z2 = spherical_to_cartesian(r_red, lat, lon)
        doa_red = np.stack((np.array(x2), np.array(y2), np.array(z2)), axis=1)

        # correct listener orientation
        doa_list_or = (self.listener_orientation @ doa_red.T).T

        # correct listener orientation in MATLAB as well but don't perform other alignments
        self.SRIR_data["DOA"] = (self.listener_orientation @ raw_doa.T).T

        # add listener pos to DOA
        self.doa = doa_list_or + self.listener_position

        # now, perform preparation steps for BRIR synthesis later, such that we are at the same stage
        # as after `Analyze_SRIR` and `PreProcess_Synthesize_SDM_Binaural` in BinauralSDM
        self.original_DSonset = int(
            self.SRIR_data["DSonset"]
        )  # this minus one is the numer of samples removed in the beginning
        self.original_DS_idx = self.SRIR_data["DS_idx"]
        self.SRIR_data = self.eng.additional_processing_after_SDMPar(self.SRIR_data)
        self.reference_P_RIR = self.SRIR_data["P_RIR"]

        return self

    def plot_doa(
        self,
        height=2.8,
        max_abs_order=1,
        lim=15,
        arrow_len=0.5,
        execute_for_sources=(lambda: None),
        return_legend_callback=False,
        draw_source=True,
    ):
        """assumes floor at z=0 and ceiling pointing in z-direction at `height`"""
        height_orders = list(range(-max_abs_order, max_abs_order + 1))
        lower_heights = [float("-inf")]
        lower_heights.extend([ord * height for ord in height_orders])
        upper_heights = [lower_heights[1]]
        upper_heights.extend([(ord + 1) * height for ord in height_orders])
        # last category, everything above
        lower_heights.append(upper_heights[-1])
        upper_heights.append(float("inf"))

        self.ir_amplitude_repr = sample_amp_repr(np.squeeze(self.ir.time))
        plt.figure(figsize=(6.4, 4.8), layout="constrained")
        plt.gca().set_aspect("equal")
        plt.xlim(-lim, lim)
        plt.ylim(-lim, lim)
        for lower_height, upper_height in zip(lower_heights, upper_heights):
            mask = (self.doa[:, 2] > lower_height) & (self.doa[:, 2] <= upper_height)
            plt.scatter(
                self.doa[mask][:, 0],
                self.doa[mask][:, 1],
                label=f"z ∈ ({lower_height}, {upper_height}] m",
                alpha=0.5,
                s=10 * self.ir_amplitude_repr[mask],
                edgecolor="none",
            )
        execute_for_sources()
        oriented_arrow(
            self.listener_position,
            self.listener_orientation,
            label="Listener",
            color="blue",
        )
        draw_source and oriented_arrow(
            self.source_position, self.source_orientation, label="Source", color="red"
        )
        for i, plane in enumerate(self.planes):
            if plane.n[2] != 0:
                continue
            xy1 = plane.p * plane.n
            slope = -plane.n[0] / plane.n[1] if plane.n[1] != 0 else float("inf")
            plt.axline(
                (xy1[0], xy1[1]), slope=slope, label=f"Plane {i}", color=f"C{i % 10}"
            )
        mygrid()
        plt.ylabel("y in m")
        plt.xlabel("x in m")

        def legend_callbback():
            plt.gca().legend(loc="upper left", bbox_to_anchor=(1, 0, 0.5, 1))

        if return_legend_callback:
            return legend_callbback
        legend_callbback()

    def plot_doa_ism(self, **kwargs):
        legend_callback = self.plot_doa(
            **kwargs,
            execute_for_sources=self._plot_ism,
            return_legend_callback=True,
            draw_source=False,
        )
        # add selected imsrc as special symbols
        position_offsets = np.array(
            [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, -1], [1, -1], [-1, 1]]
        )
        textpos_used = np.zeros((self.runs.shape[0], 2))
        label_set = False
        for row in self.runs.with_row_index().iter_rows(named=True):
            pos = self.ism_positions[row["src"]]
            oriented_arrow(
                pos,
                self.ism_orientation(row["src"]),
                color="b",
                alpha=1,
            )
            if not label_set:
                oriented_arrow(
                    pos,
                    self.ism_orientation(row["src"]),
                    color="b",
                    alpha=1,
                    label="Selected Src",
                )
                label_set = True
            xy_pos = pos[0:2]
            for position_offset_index in range(position_offsets.shape[0]):
                textpos = xy_pos + position_offsets[position_offset_index]
                if np.min(np.linalg.norm(textpos - textpos_used, axis=1)) > 1:
                    break
            textpos_used[row["index"]] = textpos

            plt.text(
                textpos[0],
                textpos[1],
                row["src"],
                alpha=0.7,
                horizontalalignment="center",
                verticalalignment="center",
            )
        legend_callback()

    def _plot_ism(self):
        # IDEA indicate handedness through left/right half arrows
        # what about up/down?
        for i in range(0, len(self.ism_positions)):
            oriented_arrow(
                self.ism_positions[i], self.ism_orientation(i), color="black", alpha=0.4
            )

    def calculate_ism(self):
        self.plane_reflections = [
            reflection_matrix_from_plane(plane) for plane in self.planes
        ]

        self.mixing_distance = self.c * self.mixing_time

        # 1st index indicates ism order, 2nd index just index
        ism_pos = [[self.source_position]]
        # (
        #    index in previous order array that created the image source,
        #    index of the plane on which it was reflected
        # )
        ism_history = [[(-1, -1)]]

        for order in range(1, self.max_reflection_order + 1):
            ism_pos.append([])
            ism_history.append([])
            for to_reflect_idx, to_reflect in enumerate(ism_pos[order - 1]):
                for plane_idx, plane in enumerate(self.planes):
                    d = plane.p - np.dot(to_reflect, plane.n)
                    if d <= 0:  # validity check
                        continue
                    reflected = to_reflect + 2 * d * plane.n
                    listener_distance = np.linalg.norm(
                        reflected - self.listener_position
                    )
                    if (
                        listener_distance - self.direct_sound_distance
                    ) > self.mixing_distance:  # proximity check
                        continue
                    # ensure distance to already existing sources is smaller than threshold
                    # this avoid duplicates (commutation of reflections) and hard to distinguish image sources
                    # but might result in matching error
                    distances_to_other_image_sources = np.linalg.norm(
                        reflected - np.array([x for l in ism_pos for x in l]), axis=1
                    )
                    if np.min(distances_to_other_image_sources) <= 0.1:
                        continue
                    ism_pos[order].append(reflected)
                    ism_history[order].append((to_reflect_idx, plane_idx))

        self.ism_positions_by_order = ism_pos
        self.ism_histories_by_order = ism_history

        # flatten
        self.ism_positions = np.array([x for l in ism_pos for x in l])

        self.ism_histories = []
        for order, order_list in enumerate(ism_history):
            for parent, plane in order_list:
                history = []
                order_iterator = order
                while order_iterator > 0:
                    history.append(plane)
                    order_iterator -= 1
                    (parent, plane) = ism_history[order_iterator][parent]
                history.reverse()
                self.ism_histories.append(history)

        # convenience variables
        self.expected_imsrc_sample = np.array(
            [
                self.imsrc_center_sample(self.listener_position - ism_position)
                for ism_position in self.ism_positions
            ]
        )

        return self

    def print_ism_summary(self):
        ism_count = [len(x) for x in self.ism_positions_by_order]
        ism_count.append(sum(ism_count))
        ism_order = list(range(0, self.max_reflection_order + 1))
        ism_order.append("all")
        print("Order|Count")
        for order, count in zip(ism_order, ism_count):
            print(f"{order:>5}|{count:5}")

    def ism_orientation(self, ism_index):
        history = self.ism_histories[ism_index]
        orientation = self.source_orientation
        for plane_index in history:
            orientation = self.plane_reflections[plane_index] @ orientation
        return orientation

    def quantize_doa(self):
        # only look at samples in mixing time, afterwards is assumed to be diffuse
        self.mixing_time_samples = int(round(self.fs * self.mixing_time))
        self.early_samples = (
            self.direct_sound_distance_samples + self.mixing_time_samples
            # TODO programming error left to fix: This should be self.direct_sound_onset_sample instead...
        )
        # shape: (time, isms, 3)
        ism2doa_vectors = np.expand_dims(self.ism_positions, 0) - np.expand_dims(
            self.doa[: self.early_samples], 1
        )
        distances = np.linalg.norm(ism2doa_vectors, axis=2)
        self.closest_imsource_index = np.argmin(distances, axis=1)
        # argmin chooses first nan if nan is present, so NaNs in DOA are mapped to 0 (direct sound)
        return self

    def smooth_quantized_doa(self):
        almost_half_window_size = int((self.post_quantization_window_size - 1) / 2)
        padded = np.pad(self.closest_imsource_index, almost_half_window_size, "edge")
        windows = np.lib.stride_tricks.sliding_window_view(
            padded, self.post_quantization_window_size
        )
        self.smoothed_imsource_index = np.array(
            list(map(lambda win: scipy.stats.mode(win).mode, windows))
        )
        # not exactly smoothing, but assing everything before direct sound to direct sound
        self.smoothed_imsource_index[: self.direct_sound_onset_sample] = 0
        return self

    def plot_imsource_index_ir(
        self,
        imsource_index,
        only=None,
        plot_imsrc_time=True,
        plot_smoothed_level=True,
        **kwargs,
    ):
        """Colors the IR according to imsource_index (-1 is diffuse, rest is colored by tab20 color cycle)"""
        plt.figure(figsize=(6.7, 4.6), layout="constrained")
        pf.plot.time(self.ir, color=(0, 0, 0, 0), **kwargs)  # just set up axis
        unique = np.unique(imsource_index)
        times = self.ir.times[: len(imsource_index)]
        ir = self.ir.time[0, : len(imsource_index)]
        linestyles = ("-", "--", "-.", ":")
        colors = plt.cm.tab20(range(20))
        for unique_index, src_idx in enumerate(unique):
            linestyle = linestyles[(unique_index // len(colors)) % len(linestyles)]
            color = colors[unique_index % len(colors)]
            row_inds = self.runs.with_row_index().filter(pl.col("src").eq(src_idx))[
                "index"
            ]
            if len(row_inds) == 0:
                label = f"\u2012\u2012 src{src_idx}"
            else:
                label = f"{row_inds[0]:02} src{src_idx}"
            if src_idx < 0:
                color = "yellow"
                label = "diffuse"
                linestyle = "-"
            if only is not None and src_idx not in only:
                color = "lightgrey"
                linestyle = "-"
                label = None
            mask = imsource_index == src_idx
            runs = haggis.npy_util.mask2runs(mask)
            lines_data = [
                np.vstack((times[start:stop], ir[start:stop])).transpose()
                for (start, stop) in runs
            ]
            lines = matplotlib.collections.LineCollection(
                lines_data, label=label, linestyle=linestyle, color=color, **kwargs
            )
            plt.gca().add_collection(lines)
            for i in range(len(runs)):
                start, stop = runs[i]
                if start == stop - 1:
                    plt.scatter(times[start], ir[start], marker=".", color=color)
            # if wanted, plot time where reflection is expected
            if plot_imsrc_time and src_idx >= 0:
                plt.axvline(
                    self.expected_imsrc_sample[src_idx] / self.fs,
                    0,
                    0.1,
                    linestyle=linestyle,
                    color=color,
                    alpha=1,
                )
        if plot_smoothed_level:
            pf.plot.time(smoothed_ir_level(self.ir), label="SPL")
            plt.axhline(0, color="black", linestyle="--", alpha=0.25)
        plt.gca().autoscale()
        self.set_early_xlim()
        plt.gcf().legend(loc="outside right upper")
        src.utils.mygrid()

    def set_early_xlim(self, offset_seconds=0, add_seconds=0):
        plt.xlim(
            self.direct_sound_onset_sample / self.fs
            - 1e-3
            + offset_seconds
            - add_seconds,
            self.early_samples / self.fs + 1e-3 + offset_seconds + add_seconds,
        )

    def source_order(self, imsrc_idx):
        return len(self.ism_histories[imsrc_idx])

    def run_moments(self, start, stop, smoothed_level=None):
        """
        Calculates moments of a run in smoothed DOA based on the smoothed ir level:
        0. zeroth moment, the total energy
        1. first moment around the imsrc expected arrival sample, normalized by zeroth moment, in samples
           (positive value indicates that the run center of mass is after the expected imsrc sample)
        2. second central moment (i.e. around first moment), normalized by maximum energy in window and
           window_width**2, then converted to energy decibels

        The window of imsrc 0 is shortened to start 1 ms before the direct_sound_onset_sample. This avoids low values for moment 2.
        """
        if smoothed_level is None:
            smoothed_level = self.smoothed_ir_level

        imsrc = self.smoothed_imsource_index[start]
        assert np.all(
            self.smoothed_imsource_index[start:stop] == imsrc
        ), "the whole run should be of a single imsrc"

        if imsrc == 0:
            # direct sound has much zero-level to the left which makes m2 be too high
            # reduce start index in that case
            start = self.direct_sound_onset_sample - int(round(1e-3 * self.fs))

        moment0 = np.sum(smoothed_level.time[0, start:stop])

        absolute_moment1 = (
            np.sum(smoothed_level.time[0, start:stop] * np.arange(start, stop))
            / moment0
        )
        # now make relative to expected imsrc sample
        moment1 = absolute_moment1 - self.expected_imsrc_sample[imsrc]

        moment2 = 10 * np.log10(
            np.sum(
                smoothed_level.time[0, start:stop]
                * (np.arange(start, stop) - absolute_moment1) ** 2
            )
            / np.max(smoothed_level.time[0, start:stop])
            / (stop - start) ** 2
        )

        assert not np.isnan(moment0)
        assert not np.isnan(moment1)
        assert not np.isnan(moment2)

        return moment0, moment1, moment2

    def calculate_extraction_metrics(self):
        """
        Post-processing for smoothed, quantized DOA. Remove completely un-extractable reflections.
        """
        # We start with setting up some parameters and selecting direct sound (which is easy)
        # - we set the fade time for later, it is meant excluding the parts of the window that are 1 or 0
        # - we have to ensure it is even, because we want the half point between sample borders of zones
        # - it's a bit arbitrary, but the minimum width of zones we expect is about post_quantization_window_size/2
        # so if half the reflection window is outside, we get about the expected minimum zone width afterwards
        # self.reflection_win_fade_samples = closest_even_integer(self.post_quantization_window_size * .5)
        self.reflection_win_fade_samples = closest_even_integer(
            self.post_quantization_window_size * 0.9
        )  # empirically increased fade time
        self.min_reflection_samples = (
            self.reflection_win_fade_samples + 1
        )  # allow for at least one [1.] in the middle of the window

        # set up things that can later be used to see result of calculations
        # initialize to diffuse (i.e. -1)
        self.pruned_imsource_index = np.full(self.smoothed_imsource_index.shape, -1)

        self.smoothed_ir_level = smoothed_ir_level(self.ir)

        # select the run of the direct sound at the expected time and also treat everything prior as direct sound
        direct_mask = self.smoothed_imsource_index == 0
        direct_runs = haggis.npy_util.mask2runs(direct_mask)
        direct_starts = direct_runs[:, 0]
        direct_starts_before = direct_starts[
            direct_starts <= self.direct_sound_onset_sample
        ]
        assert direct_starts_before.shape[0] > 0
        # last run before direct sound onset is main run
        direct_sound_main_run = direct_runs[direct_starts_before.shape[0] - 1]
        assert direct_sound_main_run.shape == (2,)
        # select direct sound main run and everything before
        self.pruned_imsource_index[: direct_sound_main_run[1]] = 0
        m0, m1, m2 = self.run_moments(0, direct_sound_main_run[1])
        run_data = [
            {
                "src": 0,
                "start": 0,
                "stop": direct_sound_main_run[1],
                "m0": m0,
                "m1": m1,
                "m2": m2,
            }
        ]

        # For the other reflections we need to go from possibly many different runs
        # per imsrc to one ordained run per imsrc.
        # Previously, I did this by selecting the one with the highest HF energy.
        # However, I now know that this process often selected runs far away from the expected
        # imsrc arrival time, resulting in non-physically correct extrapolation that we would
        # want to avoid.
        # To reduce this, I'd suggest the following instead:
        # 1. Filter all runs that are too short to be extracted with the given window
        #    size (min_reflection_samples)
        # 2. From remaining runs per imsrc, choose the run whose first moment in smoothed level is
        #    closest to expected imsrc_time.
        for src_idx in range(1, len(self.ism_positions)):
            mask = self.smoothed_imsource_index == src_idx
            runs = haggis.npy_util.mask2runs(mask)
            # 1. remove runs too short for extraction
            runs = runs[runs[:, 1] - runs[:, 0] >= self.min_reflection_samples]
            if runs.size == 0:
                continue
            # 2. choose run closest to expected imsrc sample (i.e., lowest absolute first moment m1)
            moments = np.array([self.run_moments(*startstop) for startstop in runs])
            main_run_index = np.argmin(np.abs(moments[:, 1]))
            main_run = runs[main_run_index]
            self.pruned_imsource_index[main_run[0] : main_run[1]] = src_idx
            run_data.append(
                {
                    "src": src_idx,
                    "start": main_run[0],
                    "stop": main_run[1],
                    "m0": moments[main_run_index, 0],
                    "m1": moments[main_run_index, 1],
                    "m2": moments[main_run_index, 2],
                }
            )

        # Conver to dataframe and calculate more metrics
        self.runs = pl.DataFrame(run_data)

        self.smoothed_ir_level_db = pf.dsp.decibel(
            self.smoothed_ir_level, domain="time"
        )[0]
        self.peaks, self.peak_props = scipy.signal.find_peaks(
            self.smoothed_ir_level_db, height=-np.inf, prominence=0
        )

        def highest_peak_idx_in_peaks(row):
            # mask for peaks inside window
            mask = (row["start"] <= self.peaks) & (self.peaks < row["stop"])
            # find highest peak in window
            highest_peak = np.argmax(
                np.ma.array(self.peak_props["peak_heights"], mask=~mask)
            )
            # will return 0 if no peak is in the window, which represents an invalid value
            # since the first peak should be in the noise before direct sound, this never makes sense
            return highest_peak

        def peak_prominence(row):
            highest_peak = row["highest_peak_idx_in_peaks"]
            if highest_peak == 0:
                # since the first peak should be in the noise before direct sound, this never makes sense
                # return 0, since there is no peak in this window
                return np.nan
            return self.peak_props["prominences"][highest_peak]

        def doa_std(row):
            doas = self.doa[row["start"] : row["stop"]]
            return cartesian_std(doas)

        def doa_stability(row, threshold_degree):
            highest_peak = row["highest_peak_idx_in_peaks"]
            if highest_peak == 0:
                return 0
            peak_sample = self.peaks[highest_peak]
            peak_doa = self.doa[peak_sample]
            doas = self.doa[row["start"] : row["stop"]]
            peak_sample_in_doas = peak_sample - row["start"]
            angles_to_peak = np.arccos(
                np.clip(
                    np.dot(doas, peak_doa)
                    / np.linalg.norm(doas, axis=1)
                    / np.linalg.norm(peak_doa),
                    0,
                    1,
                )
            )
            threshold = np.deg2rad(threshold_degree)
            mask = angles_to_peak < threshold
            runs = haggis.npy_util.mask2runs(mask)
            run_around_peak = np.argmax(
                (peak_sample_in_doas >= runs[:, 0]) & (peak_sample_in_doas < runs[:, 1])
            )
            run_length = runs[run_around_peak, 1] - runs[run_around_peak, 0]
            # Puomio has DOA stability of 0 in some paper, so he subtracts 1 from length
            return run_length - 1

        def doa_nn(row):
            return np.count_nonzero(
                self.closest_imsource_index[row["start"] : row["stop"]] == row["src"]
            ) / (row["stop"] - row["start"])

        # HF moments
        self.hp_f3 = 1e3  # approximate frequency where speakers start getting directive
        self.hp_iir = scipy.signal.butter(1, self.hp_f3, fs=self.fs, btype="highpass")
        self.hf_ir = self.ir.copy()
        self.hf_ir.time = scipy.signal.filtfilt(
            *self.hp_iir, self.ir.time[0, : self.early_samples]
        )
        assert self.hf_ir.cshape == (1,)
        self.smoothed_hf_level = smoothed_ir_level(self.hf_ir)
        hf_moments = np.array(
            [
                self.run_moments(
                    self.runs["start"][i], self.runs["stop"][i], self.smoothed_hf_level
                )
                for i in range(self.runs.shape[0])
            ]
        )

        self.runs = self.runs.with_columns(
            imsrc_sample=self.expected_imsrc_sample[self.runs["src"]],
            highest_peak_idx_in_peaks=pl.struct(["start", "stop"]).map_elements(
                highest_peak_idx_in_peaks, return_dtype=pl.datatypes.Int64
            ),
            doa_std=pl.struct(["start", "stop"]).map_elements(
                doa_std, return_dtype=pl.datatypes.Float64
            ),
            doa_nn=pl.struct(["start", "stop", "src"]).map_elements(
                doa_nn, return_dtype=pl.datatypes.Float64
            ),
            distance_boost=amp2db(
                np.linalg.norm(
                    self.ism_positions[self.runs["src"]] - self.listener_position,
                    axis=1,
                )
            ),
            order=np.array([self.source_order(src) for src in self.runs["src"]]),
            m0_hf=hf_moments[:, 0],
            m1_hf=hf_moments[:, 1],
            m2_hf=hf_moments[:, 2],
        )
        # add dependent columns in second step
        self.runs = self.runs.with_columns(
            peak_prominence=pl.struct(["highest_peak_idx_in_peaks"]).map_elements(
                peak_prominence, return_dtype=pl.datatypes.Float64
            ),
            doa_stability_1deg=pl.struct(
                ["start", "stop", "highest_peak_idx_in_peaks"]
            ).map_elements(
                partial(doa_stability, threshold_degree=1),
                return_dtype=pl.datatypes.Int64,
            ),
            doa_stability_5deg=pl.struct(
                ["start", "stop", "highest_peak_idx_in_peaks"]
            ).map_elements(
                partial(doa_stability, threshold_degree=5),
                return_dtype=pl.datatypes.Int64,
            ),
            relative_imsrc_time=(pl.col("imsrc_sample") - pl.col("start"))
            / (pl.col("stop") - pl.col("start")),
            directivity_boost=np.nan,  # filled in inverse filtered area
        )

        return self

    def plot_extraction_metrics(self):
        fig, axs = plt.subplots(
            ncols=1,
            nrows=4,
            figsize=(6.4, 8.2),
            squeeze=False,
            layout="constrained",
            sharex=True,
            height_ratios=(1, 1, 1, 0.4),
        )
        n = self.runs.shape[0]
        for ax in axs.reshape(-1):
            ax.grid(True, which="major", color="grey", linestyle="-")
            ax.grid(True, which="minor", color="lightgrey", linestyle=":")
            ax.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
            ax.set_axisbelow(True)
        axs[0, 0].set_xticks(
            range(n),
            labels=self.runs.with_row_index().select(
                tick=pl.concat_str([pl.col("index"), pl.col("src")], separator="\n")
            )["tick"],
        )

        axs[0, 0].set_title("Energy Metrics")
        axs[0, 0].axhline(
            6, 0, 1, color="tab:red", label="filter energy limit", zorder=1
        )
        bottom = (
            energy2db(self.runs["m0"] / self.runs["m0"][0])
            - self.runs["distance_boost"][0]
            - self.runs["directivity_boost"][0]
        )
        axs[0, 0].bar(
            range(n), self.runs["distance_boost"], bottom=bottom, label="dist. boost"
        )
        axs[0, 0].scatter(
            range(n), bottom, marker="_", color="black", label="m0 (rel.)"
        )
        bottom += self.runs["distance_boost"]
        axs[0, 0].bar(
            range(n),
            self.runs["directivity_boost"],
            bottom=bottom,
            color="tab:green",
            label="dir. boost",
        )
        axs[0, 0].scatter(
            range(n),
            self.runs["filter_energy"],
            color="tab:orange",
            label="filter energy",
        )
        axs[0, 0].set_ylabel("dB")
        axs[0, 0].legend(bbox_to_anchor=(1, 1))

        axs[1, 0].set_title("Time Metrics")
        samples = (
            self.runs.select(samples=pl.col("stop") - pl.col("start"))["samples"]
            / self.fs
            * 1e3
        )
        samples[0] /= 10
        axs[1, 0].bar(range(n), samples, label="length")
        axs[1, 0].text(
            0.1,
            0.2,
            "10% scale",
            rotation=90,
            horizontalalignment="center",
            verticalalignment="bottom",
            fontsize=8,
        )
        axs[1, 0].bar(
            range(n), self.runs["doa_nn"] * samples, color="tab:green", label="doa_nn"
        )
        doa5 = self.runs["doa_stability_5deg"] / self.fs * 1e3
        doa5[0] /= 10
        axs[1, 0].bar(range(n), doa5, color="tab:brown", label="doa_stability_5deg")
        doa1 = self.runs["doa_stability_1deg"] / self.fs * 1e3
        doa1[0] /= 10
        axs[1, 0].bar(range(n), doa1, color="tab:olive", label="doa_stability_1deg")
        axs[1, 0].set_ylabel("ms")
        m1_ms = self.runs["m1"] / self.fs * 1e3
        m1_ms[0] /= 10
        axs[1, 0].scatter(
            range(n),
            m1_ms + self.runs["relative_imsrc_time"] * samples,
            label="m1",
            color="tab:orange",
        )
        axs[1, 0].scatter(
            range(n),
            self.runs["relative_imsrc_time"] * samples,
            color="tab:red",
            label="imsrc time",
            marker="+",
        )
        axs[1, 0].legend(bbox_to_anchor=(1, 1))

        axs[2, 0].set_title("Risk Metrics")
        axs[2, 0].axhline(50, 0, 1, color="tab:red", label="7*m2+dir limit", zorder=1)
        bars = axs[2, 0].bar(
            range(n),
            self.runs["directivity_boost"] + 7 * self.runs["m2"],
            label="7*m2",
            color="tab:blue",
        )
        axs[2, 0].bar_label(bars, self.runs["order"])
        axs[2, 0].bar(
            range(n),
            self.runs["directivity_boost"],
            label="dir. boost",
            color="tab:green",
        )
        axs[2, 0].scatter(
            range(n),
            self.runs["peak_prominence"],
            label="peak prominence",
            color="tab:orange",
        )
        axs[2, 0].set_ylabel("dB")
        h, l = axs[2, 0].get_legend_handles_labels()
        order_proxy = matplotlib.lines.Line2D([], [], color="black")
        h.append(order_proxy)
        l.append("ISM order")
        axs[2, 0].legend(h, l, bbox_to_anchor=(1, 1))

        axs[3, 0].set_title("Spatial Metrics")
        axs[3, 0].bar(range(n), self.runs["doa_std"], label="doa_std")
        axs[3, 0].legend(bbox_to_anchor=(1, 1))
        axs[3, 0].set_ylabel("m")
        axs[3, 0].annotate(
            "row\nsrc",
            xy=(0, 0),
            xytext=(-26, -28),
            textcoords="offset points",
            ha="left",
            va="bottom",
        )

    def read_source_fr(self, filepath):
        self.source_fr_path = filepath
        self.source_fr_sofa = sofar.read_sofa(self.source_fr_path)
        if hasattr(self, "fs"):
            assert self.source_fr_sofa.N[-1] == self.fs / 2
        else:
            self.fs = self.source_fr_sofa.N[-1] * 2
        self.source_ir_samples = 2 * (
            self.source_fr_sofa.N.shape[0] - 1
        )  # size after irfft
        assert self.source_ir_samples % 2 == 0  # even
        # create unit vectors for directions
        x, y, z = spherical_to_cartesian(
            1,
            self.source_fr_sofa.ListenerPosition[:, 1] / 180 * np.pi,
            self.source_fr_sofa.ListenerPosition[:, 0] / 180 * np.pi,
        )
        self.source_fr_unit_vectors = np.vstack((x, y, z)).T
        return self

    def inverse_filtered_area(self, ir: pf.Signal, imsrc_idx, debug=False):
        """For an imsrc in pruned, cut out a large area around it and apply the inverse spatial FR"""
        # Idea:
        # - We take the filter length from source_fr_sofa (2**14 in our case)
        # - We cut out like this: --- 2**13 samples ---- center sample from image source listener distance ----- 2**13 - 1 samples ----- (with Hann window)
        #   So the center sample is slightly behind the middle of the window (due to even window size)
        # - We do the rfft (automatic in pyfar) and divide by the closest spatial FR from source_fr_sofa

        # imsrc center sample
        imsrc_to_listener = self.listener_position - self.ism_positions[imsrc_idx]
        imsrc_center_sample = self.imsrc_center_sample(imsrc_to_listener)

        # window area
        win_start = imsrc_center_sample - self.source_ir_samples // 2
        if win_start < 0:
            left_pad = -win_start
            win_start = 0
        else:
            left_pad = 0
        padded = pf.dsp.pad_zeros(ir, left_pad, "beginning")
        area = pf.dsp.time_window(
            padded,
            (
                win_start,
                win_start + self.source_ir_samples - 1,  # inclusive end index in pyfar
            ),
            crop="window",
        )
        assert area.n_samples == self.source_ir_samples

        # find closest source tf: Find exit angle of source towards listener
        # project into view-left-up system and normalize
        vlu_imsrc_to_listener = self.ism_orientation(imsrc_idx).T @ imsrc_to_listener.T
        vlu_imsrc_to_listener /= np.linalg.norm(vlu_imsrc_to_listener)
        # find closest source_fr_unit_vector by maximum scalar product
        cos_distances = self.source_fr_unit_vectors @ vlu_imsrc_to_listener.T
        source_fr_idx = np.argmax(cos_distances)
        source_fr = (
            self.source_fr_sofa.Data_Real[source_fr_idx, 0, :]
            + 1j * self.source_fr_sofa.Data_Imag[source_fr_idx, 0, :]
        )
        source_fr_signal = pf.Signal(
            source_fr, self.fs, domain="freq", n_samples=self.source_ir_samples
        )
        max_index = np.argmax(np.abs(source_fr_signal.time))
        source_fr_signal = pf.dsp.time_shift(source_fr_signal, -max_index)
        source_fr_inv = pf.dsp.regularized_spectrum_inversion(
            source_fr_signal, (0, 0), regu_final=0.1 * 10 ** (-30 / 20 * 2)
        )  # about 30 dB of max boost

        self.runs = self.runs.with_columns(
            directivity_boost=pl.when(pl.col("src").eq(imsrc_idx))
            .then(energy2db(pf.dsp.energy(source_fr_inv))[0])
            .otherwise(pl.col("directivity_boost"))
        )

        if debug:
            return (
                area * source_fr_inv,
                source_fr_idx,
                imsrc_center_sample,
                area,
                source_fr_signal,
                source_fr_inv,
            )
        else:
            return (
                area * source_fr_inv,
                imsrc_center_sample,
                source_fr_signal,
                win_start,
                left_pad,
                imsrc_to_listener,
            )

    def imsrc_center_sample(self, imsrc_to_listener):
        imsrc_listener_distance = np.linalg.norm(imsrc_to_listener)
        imsrc_listener_distance_samples = imsrc_listener_distance / self.c * self.fs
        imsrc_center_sample = int(
            round(imsrc_listener_distance_samples + self.sample_offset)
        )
        return imsrc_center_sample

    def extract_filter(self, ir, imsrc_idx, debug=False):
        (
            inv,
            imsrc_center_sample,
            source_fr_signal,
            area_start,
            area_left_pad,
            imsrc_to_listener,
        ) = self.inverse_filtered_area(ir, imsrc_idx)
        # first, get the run
        sel_idx = np.argmax(self.runs["src"] == imsrc_idx)
        row = self.runs.row(sel_idx, named=True)
        run = np.array([row["start"], row["stop"]])
        # make run relative to imsrc_center
        relative_run = run - imsrc_center_sample
        # translate to the inversed windowed area, which has center at index len//2
        inv_center_idx = inv.n_samples // 2
        run_in_inv = relative_run + inv_center_idx
        # caculate fades
        half_fade = (
            self.reflection_win_fade_samples // 2
        )  # is even, so no rounding error
        fade_in_start = (
            run_in_inv[0] - half_fade - 1
        )  # remove one extra to extend non-zero parts to full window length, 0 would otherwise be at fade_in_start
        fade_in_stop = (
            run_in_inv[0] + half_fade - 1 + 1
        )  # inclusive indexing and extendig by one cancels
        assert (
            fade_in_stop - fade_in_start == self.reflection_win_fade_samples - 1 + 2
        )  # pyfar uses inclusive indexing here & consider 0 and 1 at indices
        fade_out_start = run_in_inv[1] - half_fade - 1
        fade_out_stop = run_in_inv[1] + half_fade - 1 + 1
        assert (
            fade_out_stop - fade_out_start == self.reflection_win_fade_samples - 1 + 2
        )
        # window
        windowed, win = pf.dsp.time_window(
            inv,
            (fade_in_start, fade_in_stop, fade_out_start, fade_out_stop),
            return_window=True,
        )
        # fftshift to remove time delay
        shifted = windowed.copy()
        shifted.time = scipy.fft.fftshift(shifted.time)
        # cut by moving right and cutting off things to the right and tapering to left with symmetric Tukey
        prepare_cut = pf.dsp.time_shift(shifted, self.filter_size // 2)
        cut = pf.dsp.time_window(
            prepare_cut, (0, self.filter_size - 1), window=("tukey", 0.5), crop="window"
        )
        # normalize level to 1 m distance
        cut.time = cut.time * np.linalg.norm(imsrc_to_listener)

        # extract from IR, use "windowed" as we assume that the windowing for "cut" is larger than the windowed signal
        reinverted = source_fr_signal * windowed
        # remove the padding added for processing
        reinverted_padding_removed = pf.dsp.time_window(
            reinverted,
            (area_left_pad, area_left_pad + self.reflection_win_fade_samples),
            shape="left",
            crop="window",
        )
        ir_extracted = ir.copy()
        ir_extracted.time[
            :, area_start : area_start + reinverted_padding_removed.n_samples
        ] -= reinverted_padding_removed.time

        if debug:
            return (
                windowed,
                win,
                inv,
                cut,
                ir_extracted,
                reinverted_padding_removed,
                source_fr_signal,
            )
        else:
            return cut, ir_extracted

    def extract_selected(self, debug=False):
        # Note: I expect the sources in the measurement to have a different on-axis FR than in the source SOFA.
        # One way to deal with this is to calibrate against the on-axis measurement and then assume a different
        # spatial FR. However, this adds complexity and is not guaranteed to benefit. E.g., this is most interesting for
        # low frequencies but the LF behavior of the speaker can't be accurately known from the in-room measurement.
        # Therefore, I'll just leave these errors included in the reflection filters.
        # My guess is that they can mostly be removed in postprocessing by normalizing to the direct sound filter, should
        # reflection coefficients be wanted.

        # Later Note: Effect Was left in, as directivity differences are probably an even bigger factor in important mid range
        assert (
            self.runs.shape[0] > 0
        ), "there must be at least one reflection to extract"

        diffuse = self.ir.copy()
        self.diffuse_history = pf.Signal(
            np.zeros((len(self.runs), self.ir.n_samples)),
            sampling_rate=self.fs,
            n_samples=self.ir.n_samples,
            domain="time",
        )
        history_idx = 0
        if debug:
            source_fr_signals = [None] * len(self.runs)
        filters = pf.Signal(
            np.zeros((len(self.runs), self.filter_size)),
            sampling_rate=self.fs,
            n_samples=self.filter_size,
            domain="time",
        )

        # sort asceding by start of run
        sel_indices = self.runs.with_row_index().sort("start")["index"]
        for sel_idx in sel_indices:
            imsrc_idx = self.runs["src"][sel_idx]
            if debug:
                (
                    _windowed,
                    _win,
                    _inv,
                    filters[sel_idx],
                    new_diffuse,
                    _reinverted_padding_removed,
                    source_fr_signal,
                ) = self.extract_filter(diffuse, imsrc_idx, debug=True)
                source_fr_signals[sel_idx] = source_fr_signal
            else:
                filters[sel_idx], new_diffuse = self.extract_filter(diffuse, imsrc_idx)
            self.diffuse_history[history_idx] = new_diffuse
            history_idx += 1
            diffuse = new_diffuse

        self.filters = filters
        self.diffuse = diffuse
        self.runs = self.runs.with_columns(
            filter_energy=energy2db(pf.dsp.energy(filters))
            - energy2db(pf.dsp.energy(filters[0]))
        )

        if debug:
            return filters, diffuse, self.diffuse_history, source_fr_signals
        else:
            return filters, diffuse

    def plot_extraction_history(self):
        pf.plot.time(self.ir)
        pf.plot.time(self.diffuse_history)
        self.set_early_xlim()

    def filter_reflections(self):
        self.runs = self.runs.filter(
            pl.col("doa_nn") > 0.5,
            pl.col("relative_imsrc_time") >= 0,
            pl.col("relative_imsrc_time") <= 1,
            pl.col("filter_energy") < 6,
            7 * pl.col("m2") + pl.col("directivity_boost")
            < 50,  # arbitrary, empirical (sample size 1) values and formula for this criterium
        )
        # rebuild pruned imsource index
        self.pruned_imsource_index = np.full(self.smoothed_imsource_index.shape, -1)
        for row in self.runs.iter_rows(named=True):
            self.pruned_imsource_index[row["start"] : row["stop"]] = row["src"]

    def ensure_HRTF_and_BRIR_data(self):
        if hasattr(self, "BRIR_data") and hasattr(self, "HRIR"):
            return
        self.create_HRTF_and_BRIR_data()

    def create_HRTF_and_BRIR_data(self):
        self.BRIR_data = self.eng.create_BRIR_data(
            "MixingTime",
            self.mixing_time,
            "HRTF_Subject",
            "NA",
            "HRTF_Type",
            "SOFA",
            "HRTF_Path",
            self.HRTF_path,
            "FFTshiftHRIRs",
            True,  # RWTH KEMAR HRTFs are time aligned to first sample
            "BandsPerOctave",
            1.0,
            "EqTxx",
            20,
            "RTModRegFreq",
            False,
            "Length",
            self.brir_duration,
            "RenderingCondition",
            f"Quantized{self.no_doa_directions}DOA",
            "Attenuation",
            0,
            "AzOrient",
            np.array([float(az) for az in np.arange(-180, 180, 30)]),
            "ElOrient",
            np.array([0.0]),
            "DOAAzOffset",
            0.0,
            "DOAElOffset",
            0.0,
            "QuantizeDOAFlag",
            True,
            "DOADirections",
            self.no_doa_directions,
            "ExportSofaFlag",
            False,
            "ExportWavFlag",
            False,
            "ExportDSERcFlag",
            False,
            "ExportDSERsFlag",
            False,
            "DestinationPath",
            "NULL",
            "fs",
            float(self.fs),
        )
        # from Demo_BinauralSDM_QuantizedDOA_andRTModAP.m
        # AllPass filtering for the late reverb (increasing diffuseness and
        # smoothing out the EDC)
        self.BRIR_data["allpass_delays"] = [37, 113, 215]  # in samples
        self.BRIR_data["allpass_RT"] = [0.1, 0.1, 0.1]  # in seconds

        # from PreProcess_Synthesize_SDM_Binaural.m
        # Prepare filter bank for reverb compensation
        self.BRIR_data["FilterBank_minFreq"] = 62.5
        self.BRIR_data["FilterBank_maxFreq"] = 20000.0
        self.BRIR_data["FilterBank_snfft"] = float(
            round(
                (self.BRIR_data["MixingTime"] + self.BRIR_data["TimeGuard"])
                * self.BRIR_data["fs"]
            )
        )

        (
            self.BRIR_data["G"],
            self.BRIR_data["FilterBank_g"],
            self.BRIR_data["FilterBank_f1"],
            self.BRIR_data["FilterBank_f2"],
        ) = self.eng.oneOver_n_OctBandFilter(
            2.0 * self.BRIR_data["FilterBank_snfft"],
            self.BRIR_data["BandsPerOctave"],
            self.BRIR_data["fs"],
            self.BRIR_data["FilterBank_minFreq"],
            self.BRIR_data["FilterBank_maxFreq"],
            nargout=4,
        )

        self.BRIR_data, self.HRIR, self.HRIR_sofa = self.eng.read_and_massage_HRTF(
            self.BRIR_data, nargout=3
        )

        return self

    def render_reference_BRIRs(self):
        self.SRIR_data["P_RIR"] = self.reference_P_RIR

        self.reference_BRIRs = self._render_BRIRs()
        return self

    def render_diffuse_BRIRs(self):
        # set p_rir to diffuse one in SRIR_data, it has be cropped by DSonset
        # that should work as long as we don't denoise
        self.SRIR_data["P_RIR"] = self.diffuse.time[:, self.original_DSonset - 1 :].T

        self.diffuse_BRIRs = self._render_BRIRs()
        return self

    def _render_BRIRs(self):
        self.ensure_HRTF_and_BRIR_data()

        brirs_merged, self.SRIR_data_after_rendering = self.eng.render_brirs(
            self.SRIR_data, self.BRIR_data, self.HRIR, nargout=2
        )
        # shape should be (time, 2, directions)

        brirs_transposed = np.array(brirs_merged).T  # reverses axes order
        # shape should be (directions, 2, time)

        # pad zeros to beginning to cancel DSonset removal
        brirs = pf.Signal(brirs_transposed, sampling_rate=self.fs)
        brirs_padded = pf.dsp.pad_zeros(brirs, self.original_DSonset - 1, "beginning")

        # cut end to get wanted length
        brir_duration_samples = int(math.ceil(self.brir_duration * self.fs))
        brir_end_window_length = 512
        brirs_cut = pf.dsp.time_window(
            brirs_padded,
            (
                brir_duration_samples - brir_end_window_length + 1,
                brir_duration_samples - 1,
            ),
            shape="right",
            crop="window",
        )
        assert brirs_cut.n_samples == brir_duration_samples

        # return BRIRs
        return brirs_cut

    def plot_reference_brir(self):
        """plots reference BRIR at azimuth 0"""
        azimuths = np.array(self.BRIR_data["AzOrient"][0]).squeeze()
        i = np.argmin(np.abs(azimuths - 0))
        pf.plot.time(
            self.reference_BRIRs[i], alpha=0.6, label=["reference L", "reference R"]
        )
        pf.plot.time(
            pf.add((self.diffuse_BRIRs[i], 0.02), "time"),
            alpha=0.6,
            label=["diffuse L + .02", "diffuse R + .02"],
        )
        self.set_early_xlim(
            offset_seconds=384 / 2 / self.fs, add_seconds=5e-3
        )  # offset by HRTF delay
        plt.legend()
        plt.title(f"BRIRs (Azimuth {azimuths[i]:.0f}°)")

    def plot_filters(self):
        pf.plot.time_freq(self.filters)
        plt.gcf().legend(
            [
                f"{i:02} src{self.runs[i]['src'][0]}"
                for i in range(self.filters.cshape[0])
            ],
            loc="outside right upper",
        )

    def save_result(
        self,
        path_for_new_folder: Path,
        save_filters=True,
        save_brirs=True,
        save_plots=True,
        save_metrics=True,
    ):
        path_for_new_folder = Path(path_for_new_folder)
        folder_path = (
            path_for_new_folder / f"{self.input_sofa_path.stem}_{self.source_index}SRC"
        )
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        if save_metrics:
            self.runs.write_parquet(folder_path / "runs.parquet")

        if save_plots:
            self.plot_imsource_index_ir(self.pruned_imsource_index)
            plt.savefig(folder_path / "pruned_ir.pdf")
            plt.close()

            self.plot_extraction_history()
            plt.savefig(folder_path / "extraction_history.pdf")
            plt.close()

            self.plot_doa_ism()
            plt.savefig(folder_path / "doa_ism.png", bbox_inches="tight", dpi=1200)
            plt.close()

            self.plot_reference_brir()
            plt.savefig(folder_path / "brirs.pdf")
            plt.close()

            self.plot_extraction_metrics()
            plt.savefig(folder_path / "extraction_metrics.pdf")
            plt.close()

            self.plot_filters()
            plt.savefig(folder_path / "filters.pdf", bbox_inches="tight")
            plt.close()

        if save_brirs:
            # BRIRs
            # convention: SingleRoomMIMOSRIR, data is mrne
            # M: listener orientation
            # R: left/right
            # N: time
            # E: reference & diffuse (same position)
            brirs = sofar.Sofa("SingleRoomMIMOSRIR")
            brirs.Data_IR = np.stack(
                (self.diffuse_BRIRs.time, self.reference_BRIRs.time), -1
            )
            brirs.GLOBAL_Title = "Pre-rendered BRIRs from SdmIsmAnalysis"
            brirs.GLOBAL_AuthorContact = "jcmf.schule@gmail.com"
            brirs.GLOBAL_Organization = "TU Ilmenau"
            brirs.GLOBAL_DatabaseName = (
                "derived from https://doi.org/10.5281/zenodo.10450779"
            )
            src.utils.copy_attributes(
                self.input_sofa,
                brirs,
                [
                    "GLOBAL_RoomShortName",
                    "GLOBAL_RoomDescription",
                    "GLOBAL_RoomLocation",
                    "GLOBAL_RoomType",
                    "ListenerPosition_Type",
                    "ListenerPosition_Units",
                    "GLOBAL_SourceDescription",
                    "GLOBAL_SourceShortName",
                    "SourcePosition_Type",
                    "SourcePosition_Units",
                    "SourceUp",
                    "SourceView_Type",
                    "SourceView_Units",
                ],
            )
            brirs.GLOBAL_ListenerShortName = "KEMAR"
            brirs.GLOBAL_ListenerDescription = (
                "see https://doi.org/10.18154/RWTH-2020-11307"
            )
            brirs.ListenerPosition = np.tile(
                self.input_sofa.ListenerPosition[self.source_index],
                (brirs.Data_IR.shape[0], 1),
            )
            brirs.ListenerView = np.array(
                [[az, 0.0, 1.0] for az in self.BRIR_data["AzOrient"][0]]
            )
            brirs.ListenerUp = np.array(
                np.tile([0.0, 90, 1], (brirs.Data_IR.shape[0], 1))
            )
            brirs.ListenerView_Type = "spherical"
            brirs.ListenerView_Units = "degree, degree, metre"
            brirs.ReceiverDescriptions = np.array(["left", "right"])
            brirs.ReceiverView = np.array([[1.0, 0, 0], [1.0, 0, 0]])
            brirs.ReceiverUp = np.array([[0, 0, 1.0], [0, 0, 1.0]])
            brirs.ReceiverView_Type = "cartesian"
            brirs.ReceiverView_Units = "metre"
            brirs.ReceiverPosition = np.array(
                [[[0.0], [0.07], [0.0]], [[0.0], [-0.07], [0.0]]]
            )
            brirs.ReceiverPosition_Type = "cartesian"
            brirs.ReceiverPosition_Units = "metre"
            brirs.SourcePosition = np.tile(
                self.input_sofa.SourcePosition[self.source_index],
                (brirs.Data_IR.shape[0], 1),
            )
            brirs.SourceView = self.input_sofa.SourceView[self.source_index]
            brirs.EmitterDescriptions = np.array(["reference", "diffuse"])
            brirs.EmitterView = np.tile([1.0, 0, 0], (2, 1))
            brirs.EmitterUp = np.tile([0, 0, 1.0], (2, 1))
            brirs.Data_SamplingRate = self.fs
            brirs.delete("MeasurementDate")
            brirs.Data_Delay = np.tile(self.sample_offset, (1, 2, 1))
            sofar.write_sofa(folder_path / "brirs.sofa", brirs)

        if save_filters:
            # Filter, as GeneralFIR with mrn data
            # M: Each extracted reflection
            # R: 1
            # N: time
            filters = sofar.Sofa("GeneralFIR")
            filters.Data_IR = self.filters.time[:, np.newaxis, :]
            filters.GLOBAL_Title = "Extracted Reflection Filters from SdmIsmAnalysis"
            filters.GLOBAL_AuthorContact = "jcmf.schule@gmail.com"
            filters.GLOBAL_Organization = "TU Ilmenau"
            src.utils.copy_attributes(
                self.input_sofa,
                filters,
                [
                    "GLOBAL_RoomType",
                    "ListenerPosition_Type",
                    "ListenerPosition_Units",
                    "SourcePosition_Type",
                    "SourcePosition_Units",
                ],
            )
            filters.add_attribute(
                "GLOBAL_RoomDescription", self.input_sofa.GLOBAL_RoomDescription
            )
            filters.ListenerPosition = self.input_sofa.ListenerPosition[
                self.source_index
            ]
            filters.SourcePosition = self.input_sofa.SourcePosition[self.source_index]
            filters.Data_SamplingRate = self.fs
            # add custom metadata
            # use E dimension of emitters for planes
            filters.EmitterPosition = np.tile(
                [0.0, 0, 0], (len(self.planes), 1)
            )  # dummy data
            filters.add_variable(
                "PlaneNormals",
                np.array([plane.n for plane in self.planes]),
                "double",
                "EC",
            )
            filters.add_variable(
                "PlaneDistances",
                np.array([plane.p for plane in self.planes]),
                "double",
                "E",
            )
            history_string = json.dumps(
                [self.ism_histories[i] for i in self.runs["src"]]
            )
            filters.add_attribute("GLOBAL_ReflectionHistories", history_string)
            filters.add_variable("SpeedOfSound", self.c, "double", "I")
            sofar.write_sofa(folder_path / "filters.sofa", filters)

        return self
