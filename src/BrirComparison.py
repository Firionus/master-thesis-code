import logging
import numpy as np
import scipy
import pyfar as pf
import pyrato as ra

from src.utils import smoothed_ir_level


def brir_comparison(ref: pf.Signal, sig: pf.Signal):
    output = {}

    # in dB, -inf..inf, lower values are better
    output["rmse"] = rmse_brirs(ref, sig)

    d = xcorr_brirs(ref, sig)
    # in samples, -inf..inf, lower absolute values are better
    output["lag"] = d["lag_mean"]
    # in samples, 0..inf, lower values are better
    output["lag_std"] = d["lag_std"]
    # in samples, 0..inf, lower values are better
    output["lag_rms"] = d["lag_rms"]
    # in unit, -1..1, higher values are better
    output["xcorr"] = d["xcorr_mean"]
    # in unit, 0..inf, no clear association with quality
    output["xcorr_std"] = d["xcorr_std"]

    d = level_differences_brirs(ref, sig)
    # in dB, -inf..inf, lower absolute values are better
    output["level_delta"] = d["level_mean"]
    # in dB, 0..inf, lower values are better
    output["level_rms"] = d["level_rms"]

    d = energy_distribution_brirs(ref, sig)
    # in unit, 0..1, lower is better
    output["misdistributed_energy"] = d["level_distribution_error_mean"]

    try:
        d = t30_c20_brirs(ref, sig)
        # in unit, -inf..inf, lower absolute values are better
        output["t30_rel_delta"] = d["delta_t30_rel"]
        # in unit, 0..inf, lower values are better
        output["t30_rel_rms"] = d["rms_t30_rel"]
        # in dB, -inf..inf, lower absolute values are better
        output["c20_delta"] = d["delta_c20"]
        # in dB, 0..inf, lower values are better
        output["c20_rms"] = d["rms_c20"]
    except ValueError:
        logging.exception("encountered error in t30_c20_brirs")

    d = iacc_brirs(ref, sig)
    # in unit, -2..2, lower absolute values are better
    output["iacc_delta"] = d["iacc_mean"]
    # in unit, 0..2, lower values are better
    output["iacc_rms"] = d["iacc_rms"]
    # in samples, -inf..inf, lower absolute values are better
    output["lag_iacc_delta"] = d["lag_mean"]
    # in samples, 0..inf, lower values are better
    output["lag_iacc_rms"] = d["lag_rms"]

    d = ild_brirs(ref, sig)
    # in dB, -inf..inf, lower absolute values are better
    output["ild_delta"] = d["ild_mean"]
    # in dB, 0..inf, lower values are better
    output["ild_rms"] = d["ild_rms"]

    d = spectral_brirs(ref, sig)
    # in unit, 0..inf, lower values are better
    output["spectral_difference"] = d["spectral_difference_mean"]

    return output


def rmse_brirs(ref: pf.Signal, sig: pf.Signal, early_window=0.07, window_width=0.01):
    """Compute the RMSE between the two signals, in dB relative to the RMS of the reference.

    Comparison is performed for each channel, then the energy averaged.

    By default, only the first 70 ms are compared. This can be changed
    with `early_window`. The windowing time can be changed with `window_width`.
    """
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )
    ref_rms = pf.dsp.energy(ref)
    diff_rms = pf.dsp.energy(ref - sig)
    relative = diff_rms / ref_rms
    mean = np.mean(relative)
    if mean == 0:
        return -np.inf
    return 10 * np.log10(mean)


def xcorr_brirs(
    ref: pf.Signal,
    sig: pf.Signal,
    early_window=0.07,
    window_width=0.01,
    raw_output=False,
):
    """Compute the normalized cross correlation and average lag between each channel of the two signals.

    Comparison is performed for each channel, then the mean and standard deviation are returned.

    By default, only the first 70 ms (excl. window) are compared. This can be changed
    with `early_window`. The windowing time can be changed with `window_width`.
    """
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )
    ref_arr = ref.time.reshape((-1, ref.n_samples))
    sig_arr = sig.time.reshape((-1, ref.n_samples))

    xcorr_lags = scipy.signal.correlation_lags(ref_arr.shape[1], sig_arr.shape[1])
    lags = np.full(ref_arr.shape[0], np.nan)
    xcorr_maxs = np.full(ref_arr.shape[0], np.nan)
    for i in range(ref_arr.shape[0]):
        xcorr = scipy.signal.correlate(ref_arr[i], sig_arr[i])
        xcorr_max_ind = np.argmax(np.abs(xcorr))  # abs needed to detect phase inversion
        lags[i] = xcorr_lags[xcorr_max_ind]
        xcorr_maxs[i] = xcorr[xcorr_max_ind] / np.sqrt(
            np.sum(ref_arr[i] ** 2) * np.sum(sig_arr[i] ** 2)
        )

    if raw_output:
        return lags, xcorr_maxs

    return {
        "lag_mean": np.mean(lags),
        "lag_std": np.std(lags),
        "lag_rms": np.sqrt(np.mean(lags**2)),
        "xcorr_mean": np.mean(xcorr_maxs),
        "xcorr_std": np.std(xcorr_maxs),
        "xcorr_abs_mean": np.mean(np.abs(xcorr_maxs)),
        "xcorr_abs_std": np.std(np.abs(xcorr_maxs)),
        "xcorr_min": np.min(xcorr_maxs),
        "xcorr_max": np.max(xcorr_maxs),
        "lags": lags,
        "xcorrs": xcorr_maxs,
    }


def level_differences_brirs(
    ref: pf.Signal,
    sig: pf.Signal,
    early_window=0.07,
    window_width=0.01,
):
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )
    sig_energy = pf.dsp.energy(sig)
    ref_energy = pf.dsp.energy(ref)
    decibel = 10 * np.log10(sig_energy / ref_energy)

    return {
        "level_mean": np.mean(decibel),
        "level_rms": np.sqrt(np.mean(decibel**2)),
        "absolute_ref_levels": 10 * np.log10(ref_energy),
        "absolute_sig_levels": 10 * np.log10(sig_energy),
    }


def energy_distribution_brirs(
    ref: pf.Signal, sig: pf.Signal, early_window=None, window_width=0.01
):
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )
    # normalize to energy 0.5
    ref_level = (
        smoothed_ir_level(ref).time / (np.sqrt(2 * pf.dsp.energy(ref))[..., np.newaxis])
    )
    sig_level = (
        smoothed_ir_level(sig).time / (np.sqrt(2 * pf.dsp.energy(sig))[..., np.newaxis])
    )
    energy_errors = np.sum((sig_level - ref_level) ** 2, axis=len(ref_level.shape) - 1)

    return {
        "level_distribution_error_mean": np.mean(energy_errors),
        "level_distribution_error_std": np.std(energy_errors),
        "level_distribution_errors": energy_errors,
    }


def t30_c20_brirs(ref: pf.Signal, sig: pf.Signal, direct_sound_samples=1000):
    assert ref.sampling_rate == sig.sampling_rate
    ref_t30, sig_t30, ref_c20, sig_c20 = [np.full(ref.cshape, np.nan) for _ in range(4)]
    c20_sample = direct_sound_samples + int(0.02 * ref.sampling_rate)

    for i in np.ndindex(ref.cshape):
        ref_edc = ra.energy_decay_curve_chu(
            ref[i], time_shift=False, channel_independent=True
        )
        assert ref_edc.cshape == (1,)
        ref_t30[i] = ra.reverberation_time_linear_regression(ref_edc, T="T30")[0]
        ref_c20[i] = 10 * np.log10(1 / ref_edc.time[0, c20_sample] - 1)

        sig_edc = ra.energy_decay_curve_chu(
            sig[i], time_shift=False, channel_independent=True
        )
        sig_t30[i] = ra.reverberation_time_linear_regression(sig_edc, T="T30")[0]
        sig_c20[i] = 10 * np.log10(1 / sig_edc.time[0, c20_sample] - 1)

    delta_t30 = np.mean(sig_t30 - ref_t30)
    delta_t30_rel = delta_t30 / np.mean(ref_t30)
    rms_t30 = np.sqrt(np.mean((sig_t30 - ref_t30) ** 2))
    rms_t30_rel = rms_t30 / np.mean(ref_t30)

    delta_c20 = np.mean(sig_c20 - ref_c20)
    rms_c20 = np.sqrt(np.mean((sig_c20 - ref_c20) ** 2))

    return {
        "ref_t30": ref_t30,
        "sig_t30": sig_t30,
        "delta_t30_rel": delta_t30_rel,
        "rms_t30_rel": rms_t30_rel,
        "ref_c20": ref_c20,
        "sig_c20": sig_c20,
        "delta_c20": delta_c20,
        "rms_c20": rms_c20,
    }


def iacc_brirs(ref: pf.Signal, sig: pf.Signal, early_window=0.07, window_width=0.01):
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )

    xcorr_lags = scipy.signal.correlation_lags(ref.n_samples, sig.n_samples)
    # find 1ms limits
    start = np.argmax(xcorr_lags > (-1e-3 * ref.sampling_rate))
    stop = np.argmax(xcorr_lags > (1e-3 * ref.sampling_rate))
    ref_lags, sig_lags, ref_iacc, sig_iacc = [
        np.full(ref.cshape[0], np.nan) for _ in range(4)
    ]

    for i in range(ref.cshape[0]):
        ref_xcorr = scipy.signal.correlate(ref[i, 0].time[0], ref[i, 1].time[0])
        ref_xcorr_max_ind = np.argmax(np.abs(ref_xcorr[start : stop + 1])) + start
        ref_lags[i] = xcorr_lags[ref_xcorr_max_ind]
        ref_iacc[i] = ref_xcorr[ref_xcorr_max_ind] / np.sqrt(
            pf.dsp.energy(ref[i, 0])[0] * pf.dsp.energy(ref[i, 1])[0]
        )

        sig_xcorr = scipy.signal.correlate(sig[i, 0].time[0], sig[i, 1].time[0])
        sig_xcorr_max_ind = np.argmax(np.abs(sig_xcorr[start : stop + 1])) + start
        sig_lags[i] = xcorr_lags[sig_xcorr_max_ind]
        sig_iacc[i] = sig_xcorr[sig_xcorr_max_ind] / np.sqrt(
            pf.dsp.energy(sig[i, 0])[0] * pf.dsp.energy(sig[i, 1])[0]
        )

    return {
        "iacc_mean": np.mean(sig_iacc - ref_iacc),
        "iacc_rms": np.sqrt(np.mean((sig_iacc - ref_iacc) ** 2)),
        "lag_mean": np.mean(sig_lags - ref_lags),
        "lag_rms": np.sqrt(np.mean((sig_lags - ref_lags) ** 2)),
        "ref_lags": ref_lags,
        "sig_lags": sig_lags,
        "ref_iacc": ref_iacc,
        "sig_iacc": sig_iacc,
    }


def ild_brirs(ref: pf.Signal, sig: pf.Signal, early_window=0.07, window_width=0.01):
    assert ref.sampling_rate == sig.sampling_rate
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )

    ref_ild, sig_ild = [np.full(ref.cshape[0], np.nan) for _ in range(2)]

    for i in range(ref.cshape[0]):
        ref_ild[i] = (
            10 * np.log10(pf.dsp.energy(ref[i, 0]) / pf.dsp.energy(ref[i, 1]))[0]
        )
        sig_ild[i] = (
            10 * np.log10(pf.dsp.energy(sig[i, 0]) / pf.dsp.energy(sig[i, 1]))[0]
        )

    return {
        "ild_mean": np.mean(sig_ild - ref_ild),
        "ild_rms": np.sqrt(np.mean((sig_ild - ref_ild) ** 2)),
        "ref_ild": ref_ild,
        "sig_ild": sig_ild,
    }


def spectral_brirs(
    ref: pf.Signal, sig: pf.Signal, early_window=0.07, window_width=0.01
):
    assert ref.sampling_rate == sig.sampling_rate
    assert ref.n_samples == sig.n_samples
    if early_window is not None:
        win_start = early_window * ref.sampling_rate
        win_stop = win_start + window_width * ref.sampling_rate
        ref = pf.dsp.time_window(
            ref, (win_start, win_stop), shape="right", crop="window"
        )
        sig = pf.dsp.time_window(
            sig, (win_start, win_stop), shape="right", crop="window"
        )

    start = np.argmax(ref.frequencies > 40)
    stop = np.argmax(ref.frequencies > 16e3)

    # log weighting
    weights = 1 / ref.frequencies[start:stop]
    weights /= np.sum(weights)  # results in weighted mean after summation

    # normalize power
    ref_normalized = ref / np.sqrt(pf.dsp.energy(ref))
    sig_normalized = sig / np.sqrt(pf.dsp.energy(sig))

    # smoothing
    ref_smoothed, _ = pf.dsp.smooth_fractional_octave(ref_normalized, 3)
    sig_smoothed, _ = pf.dsp.smooth_fractional_octave(sig_normalized, 3)

    # abs values in valid frequency range
    ref_abs = np.abs(ref_smoothed.freq[..., start:stop])
    sig_abs = np.abs(sig_smoothed.freq[..., start:stop])

    # weighted difference of magnitude (linear)
    weighted_difference = weights * (sig_abs - ref_abs)

    overall_differences = np.sum(np.abs(weighted_difference), axis=-1)

    return {
        "spectral_difference_mean": np.mean(overall_differences),
        "spectral_differences": overall_differences,
        "ref_abs": ref_abs,
        "sig_abs": sig_abs,
        "frequencies": ref.frequencies[start:stop],
    }
