from pathlib import Path
import logging
import numpy as np
import sofar
import pyfar as pf

from src.SdmIsmAnalysis import Plane, reflection_matrix_from_plane
from .orientations import view_up_to_matrix
import json

from scipy.spatial import cKDTree


class SimpleSynthesis:
    def __init__(self) -> None:
        return

    def read_HRTFs(self, path):
        path = Path(path)
        self.hrtfs, self.hrtf_source_coords, _receiver_coords = pf.io.read_sofa(
            path, verify=False
        )
        self.hrtf_source_coords_tree = cKDTree(self.hrtf_source_coords.cartesian)
        try:
            assert self.hrtfs.sampling_rate == self.fs
        except AttributeError:
            self.fs = self.hrtfs.sampling_rate
        self.hrtfs.time = np.fft.fftshift(self.hrtfs.time, axes=2)
        return self

    def read_source_fr(self, path):
        path = Path(path)
        self.source_fr_sofa = sofar.read_sofa(path)
        self.source_frs = pf.Signal(
            np.squeeze(
                self.source_fr_sofa.Data_Real + 1j * self.source_fr_sofa.Data_Imag
            ),
            sampling_rate=self.source_fr_sofa.N[-1] * 2,
            n_samples=(self.source_fr_sofa.N.shape[0] - 1) * 2,
            domain="freq",
        )

        # shift all source IRs such that abs max is at sample 0 (needs to later be applied with cyclic convolution!)
        # for i in range(self.source_frs.cshape[0]):
        #     max_idx = np.argmax(np.abs(self.source_frs[i].time))
        #     self.source_frs.time[i] = pf.dsp.time_shift(
        #         self.source_frs[i], -max_idx
        #     ).time

        # determine the abs max of each source_fr -> After applying source fr, signal
        # needs to be inserted more to the left by this amount
        self.source_frs_center_idx = np.argmax(np.abs(self.source_frs.time), axis=1)
        try:
            assert self.source_frs.sampling_rate == self.fs
        except AttributeError:
            self.fs = self.source_frs.sampling_rate
        assert self.source_fr_sofa.ListenerPosition_Type == "spherical"
        rad_coords = pf.deg2rad(self.source_fr_sofa.ListenerPosition)
        self.source_fr_listener_coords = pf.Coordinates.from_spherical_elevation(
            rad_coords[:, 0],
            rad_coords[:, 1],
            1,  # use radius 1, later we need same radius everywhere and it's most simple to choose 1
        )
        self.source_fr_listener_coords_tree = cKDTree(
            self.source_fr_listener_coords.cartesian
        )
        return self

    def read_input_sofas(self, folder_path: Path):
        folder_path = Path(folder_path)
        self.brirs_sofa = sofar.read_sofa(folder_path / "brirs.sofa")
        self.filters_sofa = sofar.read_sofa(folder_path / "filters.sofa")
        self._process_sofas()
        return self

    def _process_sofas(self):
        # BRIRS

        brirs_signal, _source_coordinates, _receiver_coordinates = pf.io.convert_sofa(
            self.brirs_sofa
        )
        try:
            assert brirs_signal.sampling_rate == self.fs
        except AttributeError:
            self.fs = brirs_signal.sampling_rate
        self.diffuse_brirs = brirs_signal[0]
        self.reference_brirs = brirs_signal[1]

        self.listener_position = self.brirs_sofa.ListenerPosition[0]  # assert all same

        assert self.brirs_sofa.ListenerView_Type == "spherical"
        self.listener_view_cart = pf.Coordinates.from_spherical_elevation(
            pf.deg2rad(self.brirs_sofa.ListenerView)[:, 0],
            pf.deg2rad(self.brirs_sofa.ListenerView)[:, 1],
            self.brirs_sofa.ListenerView[:, 2],
        ).cartesian
        self.listener_up_cart = pf.Coordinates.from_spherical_elevation(
            pf.deg2rad(self.brirs_sofa.ListenerUp)[:, 0],
            pf.deg2rad(self.brirs_sofa.ListenerUp)[:, 1],
            self.brirs_sofa.ListenerUp[:, 2],
        ).cartesian
        self.brir_listener_orientations = [
            view_up_to_matrix(*t)
            for t in zip(self.listener_view_cart, self.listener_up_cart)
        ]

        self.source_position = self.brirs_sofa.SourcePosition[0]  # assert all same

        assert self.brirs_sofa.SourceView_Type == "cartesian"
        self.source_orientation = view_up_to_matrix(
            self.brirs_sofa.SourceView[0], self.brirs_sofa.SourceUp[0]
        )

        self.sample_offset = int(
            self.brirs_sofa.Data_Delay.flatten()[0]
        )  # assert all same

        # Filters

        self.filters, _, _ = pf.io.convert_sofa(self.filters_sofa)
        self.ism_histories = json.loads(self.filters_sofa.GLOBAL_ReflectionHistories)
        self.planes = [
            Plane(*t)
            for t in zip(
                self.filters_sofa.PlaneDistances, self.filters_sofa.PlaneNormals
            )
        ]
        self.plane_reflections = [
            reflection_matrix_from_plane(plane) for plane in self.planes
        ]
        self.c = self.filters_sofa.SpeedOfSound.flatten()[0]

    def perform_ism(self, source_position, source_orientation):
        imsrc_positions = np.zeros((len(self.ism_histories), 3))
        imsrc_orientations = np.zeros((len(self.ism_histories), 3, 3))
        invalid_imsrc_indices = []
        for i in range(len(self.ism_histories)):
            history = self.ism_histories[i]
            current_position = source_position
            current_orientation = source_orientation
            for plane_idx in history:
                plane = self.planes[plane_idx]
                d = plane.p - np.dot(current_position, plane.n)
                if d < 0:
                    invalid_imsrc_indices.append(i)
                current_position = current_position + 2 * d * plane.n
                current_orientation = (
                    self.plane_reflections[plane_idx] @ current_orientation
                )
            imsrc_positions[i] = current_position
            imsrc_orientations[i] = current_orientation
        return imsrc_positions, imsrc_orientations, invalid_imsrc_indices

    def extrapolating_synthesis(
        self,
        source_position,
        source_orientation,
        listener_position,
        debug=False,
        only_brir_indices=None,
        ignore_imsrcs=None,
        return_invalid_imsrcs=False,
        imsrcs_make_unextrapolated=None,  # put these imsrcs in at the analysis source_position, source_orientation and listener_position, as if they were unextracted
    ):
        if imsrcs_make_unextrapolated is None:
            imsrcs_make_unextrapolated = []
        if ignore_imsrcs is None:
            ignore_imsrcs = []
        assert source_position.shape == (3,)
        assert source_orientation.shape == (3, 3)
        assert listener_position.shape == (3,)

        imsrc_positions, imsrc_orientations, invalid_imsrc_indices = self.perform_ism(
            source_position, source_orientation
        )
        if len(invalid_imsrc_indices) > 0:
            logging.info(
                f"ignoring invalid imsrc indices {invalid_imsrc_indices} in addition to ones in ignore_imsrcs: {ignore_imsrcs}"
            )
            ignore_imsrcs.extend(invalid_imsrc_indices)

        imsrcs_to_listener = listener_position - imsrc_positions
        distances = np.linalg.norm(imsrcs_to_listener, axis=1)
        sample_delays = np.int64(
            np.round(distances / self.c * self.fs)
        )  # zero when right on listener
        normalized_imsrcs_to_listener = (
            imsrcs_to_listener / np.linalg.norm(imsrcs_to_listener, axis=1)[:, None]
        )  # broadcast along axis=0

        # first perform nonextrapolating synthesis with imsrcs_make_unextrapolated as basis
        if len(imsrcs_make_unextrapolated) == 0:
            brirs = self.diffuse_brirs.copy()
        else:
            all_imsrcs = set(range(len(self.ism_histories)))
            imsrcs_extrapolated_or_ignored = (
                all_imsrcs - set(imsrcs_make_unextrapolated)
            ) | set(ignore_imsrcs)
            brirs = self.extrapolating_synthesis(  # nonextrapolating use
                self.source_position,
                self.source_orientation,
                self.listener_position,
                only_brir_indices=only_brir_indices,
                ignore_imsrcs=imsrcs_extrapolated_or_ignored,
            )
            ignore_imsrcs.extend(imsrcs_make_unextrapolated)

        filter_center_idx = self.filters.n_samples // 2

        # DS always needs to end up at 1000 as before, so we'll have to compensate the distance change in time
        original_direct_sound_distance = np.linalg.norm(
            self.listener_position - self.source_position
        )
        original_direct_sound_sample_delay = int(
            round(original_direct_sound_distance / self.c * self.fs)
        )
        extrapolation_sample_offset = (
            original_direct_sound_sample_delay - sample_delays[0]
        )
        original_direct_sound_idx = (
            self.sample_offset + original_direct_sound_sample_delay
        )
        # assert original_direct_sound_idx == 1000

        if only_brir_indices is not None:
            brir_indices = only_brir_indices
        else:
            brir_indices = range(brirs.cshape[0])

        if debug:
            max_of_imsrc_inserted_at = []
            inserted_reflection = []
            hrtf_applied_arr = []
            relative_to_source_arr = []
            source_fr_idx_arr = []

        for brir_idx in brir_indices:
            listener_orientation = self.brir_listener_orientations[brir_idx]
            for imsrc_idx in range(len(self.ism_histories)):
                if imsrc_idx in ignore_imsrcs:
                    continue
                reflection_filter = self.filters[imsrc_idx]
                # distance-level correction
                reflection_filter /= distances[imsrc_idx]
                # HRTF based on listener incidence
                listener_to_imsrc = -normalized_imsrcs_to_listener[imsrc_idx]
                relative_listener_to_imsrc = listener_orientation.T @ listener_to_imsrc
                listener_to_imsrc = pf.Coordinates.from_cartesian(
                    *relative_listener_to_imsrc
                )
                _distances, hrtf_inds = self.hrtf_source_coords_tree.query(
                    (listener_to_imsrc.cartesian,)
                )
                hrtf_idx = hrtf_inds[0][0]
                hrtf_applied = pf.dsp.convolve(
                    reflection_filter, self.hrtfs[hrtf_idx], mode="full"
                )  # we are including the HRTF delay just as we did when using BinauralSDM

                # source fr based on source exit angle
                relative_to_source = pf.Coordinates.from_cartesian(
                    *(
                        imsrc_orientations[imsrc_idx].T
                        @ normalized_imsrcs_to_listener[imsrc_idx]
                    )
                )
                _distances, source_fr_inds = self.source_fr_listener_coords_tree.query(
                    (relative_to_source.cartesian,)
                )
                source_fr_idx = source_fr_inds[0][0]
                source_fr_applied = pf.dsp.convolve(
                    hrtf_applied, self.source_frs[source_fr_idx], mode="cyclic"
                )
                # insert at proper delay based on distance
                # goal: filter center index lands at delay + offset
                first_time_idx = (
                    sample_delays[imsrc_idx]
                    + self.sample_offset
                    + extrapolation_sample_offset
                    - filter_center_idx
                    - self.source_frs_center_idx[source_fr_idx]
                )
                if first_time_idx < 0:
                    source_fr_applied = pf.dsp.time_window(
                        source_fr_applied,
                        (
                            -first_time_idx,
                            -first_time_idx + original_direct_sound_idx * 0.8,
                        ),
                        shape="left",
                        crop="window",
                    )
                    first_time_idx = 0

                brirs[brir_idx].time[
                    :, first_time_idx : first_time_idx + source_fr_applied.n_samples
                ] += source_fr_applied.time

                if debug:
                    max_idx = np.argmax(np.abs(source_fr_applied.time), axis=1)[0]
                    max_of_imsrc_inserted_at.append(first_time_idx + max_idx)
                    inserted_reflection.append(source_fr_applied)
                    hrtf_applied_arr.append(hrtf_applied)
                    relative_to_source_arr.append(relative_to_source)
                    source_fr_idx_arr.append(source_fr_idx)

        if debug:
            return (
                brirs,
                sample_delays,
                max_of_imsrc_inserted_at,
                inserted_reflection,
                hrtf_applied_arr,
                relative_to_source_arr,
                source_fr_idx_arr,
                normalized_imsrcs_to_listener,
                imsrc_orientations,
            )
        elif return_invalid_imsrcs:
            return brirs, invalid_imsrc_indices
        else:
            return brirs
