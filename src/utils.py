import matplotlib.pyplot as plt
import numpy as np
import math
import pyfar as pf
import scipy


def mygrid():
    plt.grid(True, which="major", color="grey", linestyle="-")
    plt.grid(True, which="minor", color="lightgrey", linestyle=":")
    plt.minorticks_on()


def amp2db(x):
    a = np.abs(x)
    a[a == 0] = np.NAN  # make 0 be NaN to avoid error
    return 20 * np.log10(a)


def energy2db(x):
    a = np.abs(x)
    a[a == 0] = np.NAN  # make 0 be NaN to avoid error
    return 10 * np.log10(a)


def sample_amp_repr(ir):
    "Transform IR into sample size on a linear scale from 0 to 1, e.g. for alpha or size in scatter plot"
    alog = amp2db(ir)
    alognorm = alog - np.nanmax(alog)  # 0 dB max
    alognorm_squeeze = np.clip((alognorm / 60) + 1, 0, 1)  # 60 dB is minimum
    return alognorm_squeeze


def oriented_arrow(
    position, orientation, arrow_len=0.5, color=None, label=None, alpha=0.5, *kwargs
):
    """draw an arrow representing position and orientation in 2D. Position is a 2D/3D Vector,
    orientation a 3x3 matrix with the view vector in the first column."""
    return plt.arrow(
        position[0],
        position[1],
        orientation[0, 0] * arrow_len,
        orientation[1, 0] * arrow_len,
        color=color,
        width=arrow_len / 10,
        head_width=arrow_len * 2 / 3,
        head_length=arrow_len,
        length_includes_head=True,
        overhang=1,
        label=label,
        alpha=alpha,
        *kwargs,
    )


def closest_even_integer(f):
    """disambiguates upwards, so for 3.0 it returns 4"""
    ceil = math.floor(f + 1)  # disambiguation
    floor = math.floor(f)
    if ceil % 2 == 0:  # ceil is even
        return ceil
    else:
        return floor


def copy_attributes(source, destination, attributes):
    for attr in attributes:
        setattr(destination, attr, getattr(source, attr))


def smoothed_ir_level(input: pf.Signal):
    """
    Return smoothed IR level of the input signal, with 0.125 ms gaussian window.

    Corresponds to Puomio et al. 2021: "Sound rendering with early reflections extracted
    from a measured spatial room impulse response"
    (doi.org/10.1109/I3DA48870.2021.9610900)
    """
    output = input.copy()
    output.time = np.sqrt(
        scipy.ndimage.gaussian_filter(
            input.time**2, 0.125e-3 * input.sampling_rate, axes=len(output.cshape)
        )
    )
    return output


def cartesian_std(arr):
    arr = arr[~np.isnan(arr).any(axis=1)]  # ignore NaN
    mean = np.mean(arr, axis=0)
    square_distances = np.linalg.norm(arr - mean, axis=1) ** 2
    std = np.sqrt(np.mean(square_distances))
    return std


def peak_rise_from_left(level: pf.Signal, peaks, guard_time=2e-3):
    """Create semi-robust metric comparing peak heights to previous level for direct sound detection (e.g. first result > 10).

    For efficiency, feel free to filter `peaks` for useful sample times."""
    guard = int(round(guard_time * level.sampling_rate))
    result = np.array(
        [
            level.time[0, peaks[i]]
            / np.mean(level.time[0, : max(guard, peaks[i] - guard)])
            for i in range(len(peaks))
        ]
    )
    return result


def find_onset_sample(signal: pf.Signal, guard_time=2e-3, threshold=10):
    level = smoothed_ir_level(signal)
    peaks, _ = scipy.signal.find_peaks(level.time[0])
    peak_rise = peak_rise_from_left(level, peaks)
    onset_peak = np.argmax(peak_rise > threshold)
    return peaks[onset_peak]
