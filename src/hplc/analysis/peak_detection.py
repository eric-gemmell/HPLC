from __future__ import annotations

import numpy as np
import scipy.signal
import scipy.optimize
import scipy.special


def _skew_normal(t, amplitude, mu, sigma, alpha):
    """
    Skew-normal peak shape.
    I(t) = 2 * A * phi(t; mu, sigma) * Phi(alpha * (t - mu) / sigma)
    """
    z = (t - mu) / sigma
    phi = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * z ** 2)
    cdf = 0.5 * (1 + scipy.special.erf(alpha * z / np.sqrt(2)))
    return 2 * amplitude * phi * cdf


def _multi_skew_normal(t, *params):
    """Sum of N skew-normal peaks. params = [amp1, mu1, sig1, alpha1, ...]"""
    n_peaks = len(params) // 4
    result = np.zeros_like(t, dtype=float)
    for i in range(n_peaks):
        result += _skew_normal(t, *params[i * 4:(i + 1) * 4])
    return result


def _estimate_noise(signal: np.ndarray) -> float:
    """MAD-based noise estimator from successive differences."""
    diff = np.diff(signal)
    return float(np.median(np.abs(diff)) / 0.6745)


def _peak_properties(amplitude, mu, sigma, alpha):
    """Compute retention_time, max_height, area, start_time, end_time from fitted params.

    Uses the full theoretical extent of the peak (no clipping to data range).
    """
    margin = 5 * sigma * (1 + abs(alpha))
    t_lo = mu - margin
    t_hi = mu + margin
    t_fine = np.linspace(t_lo, t_hi, 100000)
    curve = _skew_normal(t_fine, amplitude, mu, sigma, alpha)

    max_idx = np.argmax(curve)
    center = float(t_fine[max_idx])
    max_height = float(curve[max_idx])

    threshold = max_height * 0.001
    above = np.where(curve > threshold)[0]
    if len(above) > 0:
        start = float(t_fine[above[0]])
        end = float(t_fine[above[-1]])
    else:
        start = center - 3 * sigma
        end = center + 3 * sigma

    area = float(np.trapezoid(curve, t_fine))
    return center, max_height, area, start, end


def _quick_range(amp, mu, sigma, alpha, fraction=0.01):
    """Quick estimate of peak range where signal > fraction * max."""
    margin = 3 * sigma * (1 + abs(alpha))
    t = np.linspace(mu - margin, mu + margin, 1000)
    curve = _skew_normal(t, amp, mu, sigma, alpha)
    max_val = curve.max()
    if max_val <= 0:
        return mu - sigma, mu + sigma
    above = t[curve > max_val * fraction]
    if len(above) == 0:
        return mu - sigma, mu + sigma
    return float(above[0]), float(above[-1])


def _merge_overlapping(ranges):
    """Group indices by overlapping ranges. Returns list of lists of indices."""
    n = len(ranges)
    if n == 0:
        return []

    order = sorted(range(n), key=lambda i: ranges[i][0])
    groups = []
    current = [order[0]]
    _, cur_hi = ranges[order[0]]

    for idx in order[1:]:
        lo, hi = ranges[idx]
        if lo <= cur_hi:
            current.append(idx)
            cur_hi = max(cur_hi, hi)
        else:
            groups.append(current)
            current = [idx]
            cur_hi = hi
    groups.append(current)
    return groups


def detect_peaks(
    time: np.ndarray,
    signal: np.ndarray,
) -> list[dict]:
    """
    Detect and fit skew-normal peaks in a chromatographic signal.

    Algorithm:
    Phase 1 — Iterative detection: find peaks, fit individually, subtract from
              residual, repeat to discover peaks hidden under larger ones.
    Phase 2 — Group overlapping peaks and jointly refit within each group
              for accurate deconvolution.
    Phase 3 — Compute properties from the fitted parameters.
    """
    time = np.asarray(time, dtype=float)
    signal_arr = np.asarray(signal, dtype=float)

    if len(time) < 3:
        return []

    dt = np.mean(np.diff(time)) if len(time) > 1 else 1.0

    sig_range = signal_arr.max() - signal_arr.min()
    if sig_range < 1e-12:
        return []

    original_noise = _estimate_noise(signal_arr)
    noise_floor = max(original_noise, sig_range * 1e-6)

    # ---- Phase 1: iterative peak detection on residual ----
    all_params: list[tuple[float, float, float, float]] = []
    residual = signal_arr.copy()

    for _iteration in range(5):
        shifted = residual - residual.min()
        res_range = float(shifted.max())

        if res_range < noise_floor * 3:
            break

        norm = shifted / res_range
        noise_norm = _estimate_noise(norm)
        prom = max(0.01, 5 * noise_norm)

        peak_idx, _ = scipy.signal.find_peaks(norm, prominence=prom)
        if len(peak_idx) == 0:
            break

        widths, _, _, _ = scipy.signal.peak_widths(shifted, peak_idx, rel_height=0.5)

        found_any = False
        order = np.argsort(-shifted[peak_idx])

        for rank in order:
            pidx = int(peak_idx[rank])
            width = float(widths[rank])

            hwhm = width * dt / 2
            sigma_guess = max(hwhm / np.sqrt(2 * np.log(2)), dt)
            mu_guess = float(time[pidx])
            height_guess = float(residual[pidx])
            if height_guess <= 0:
                continue
            amp_guess = max(height_guess * sigma_guess * np.sqrt(2 * np.pi), dt)

            win_half = max(int(6 * sigma_guess / dt), 15)
            i_lo = max(pidx - win_half, 0)
            i_hi = min(pidx + win_half, len(time) - 1)
            t_win = time[i_lo:i_hi + 1]
            s_win = residual[i_lo:i_hi + 1]

            p0 = [amp_guess, mu_guess, sigma_guess, 0.0]
            lo = [0, time[0], dt * 0.5, -50.0]
            hi = [amp_guess * 100, time[-1], (time[-1] - time[0]) / 2, 50.0]

            try:
                popt, _ = scipy.optimize.curve_fit(
                    _skew_normal, t_win, s_win,
                    p0=p0, bounds=(lo, hi), maxfev=5000,
                )
            except (RuntimeError, ValueError):
                popt = np.array(p0)

            params = tuple(float(x) for x in popt)
            all_params.append(params)
            residual -= _skew_normal(time, *params)
            found_any = True

        if not found_any:
            break

    if len(all_params) == 0:
        return []

    # ---- Phase 2: group overlapping peaks and jointly refit within groups ----
    ranges = [_quick_range(*p) for p in all_params]
    groups = _merge_overlapping(ranges)

    final_params: list[tuple[float, float, float, float]] = []

    for group in groups:
        group_params = [all_params[i] for i in group]

        if len(group) == 1:
            final_params.append(group_params[0])
            continue

        # Determine fitting window from group ranges + buffer
        g_lo = min(ranges[i][0] for i in group)
        g_hi = max(ranges[i][1] for i in group)
        max_sigma = max(all_params[i][2] for i in group)
        buf = max_sigma * 3
        w_lo = max(g_lo - buf, float(time[0]))
        w_hi = min(g_hi + buf, float(time[-1]))

        mask = (time >= w_lo) & (time <= w_hi)
        t_win = time[mask]
        s_win = signal_arr[mask]

        if len(t_win) < 4 * len(group):
            final_params.extend(group_params)
            continue

        n = len(group)
        p0_all: list[float] = []
        lo_all: list[float] = []
        hi_all: list[float] = []

        for amp, mu, sigma, alpha in group_params:
            p0_all.extend([amp, mu, sigma, alpha])
            lo_all.extend([0, float(time[0]), dt * 0.5, -50.0])
            hi_all.extend([
                max(amp * 100, sig_range * 100),
                float(time[-1]),
                (float(time[-1]) - float(time[0])) / 2,
                50.0,
            ])

        try:
            popt, _ = scipy.optimize.curve_fit(
                _multi_skew_normal, t_win, s_win,
                p0=p0_all, bounds=(lo_all, hi_all),
                maxfev=50000 * n,
            )
        except (RuntimeError, ValueError):
            popt = np.array(p0_all)

        for i in range(n):
            final_params.append(tuple(float(x) for x in popt[i * 4:(i + 1) * 4]))

    # ---- Phase 3: compute properties and filter ----
    results = []
    for amp, mu, sigma, alpha in final_params:
        center, max_height, area, start, end = _peak_properties(amp, mu, sigma, alpha)

        if max_height < noise_floor * 2:
            continue

        results.append({
            "retention_time": center,
            "max_height": max_height,
            "area": area,
            "start": start,
            "start_time": start,
            "end": end,
            "end_time": end,
            "mu": mu,
            "amplitude": amp,
            "sigma": sigma,
            "alpha": alpha,
        })

    return results
