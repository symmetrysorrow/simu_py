import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scipy

import general

k_b = 1.381e-23
ptfn_Flink = 0.5
eta = 100.0


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def smooth_log_spectrum(values, window):
    values = np.asarray(values, dtype=float)
    safe = np.maximum(values, np.finfo(float).tiny)
    log_values = np.log(safe)
    pad = window // 2
    padded = np.pad(log_values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.exp(np.convolve(padded, kernel, mode="valid"))


def generate_noise_from_asd(noise_asd, sample, rate, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    noise_asd = np.asarray(noise_asd[: int(sample / 2) + 1], dtype=float)
    df = rate / sample

    spectrum = np.zeros(len(noise_asd), dtype=np.complex128)
    if len(noise_asd) == 0:
        return np.zeros(sample)

    spectrum[0] = noise_asd[0] * sample * np.sqrt(df)

    if len(noise_asd) > 2:
        phases = rng.uniform(0.0, 2.0 * np.pi, len(noise_asd) - 2)
        magnitude = noise_asd[1:-1] * sample * np.sqrt(df / 2.0)
        spectrum[1:-1] = magnitude * np.exp(1j * phases)

    if sample % 2 == 0 and len(noise_asd) > 1:
        spectrum[-1] = noise_asd[-1] * sample * np.sqrt(df)
    elif len(noise_asd) > 1:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        magnitude = noise_asd[-1] * sample * np.sqrt(df / 2.0)
        spectrum[-1] = magnitude * np.exp(1j * phase)

    return np.fft.irfft(spectrum, n=sample)


def asd_from_rfft(noise_fft, sample, rate):
    noise_fft = np.asarray(noise_fft)
    df = rate / sample
    amp_dens = np.zeros(len(noise_fft), dtype=float)

    if len(noise_fft) == 0:
        return amp_dens

    amp_dens[0] = np.abs(noise_fft[0]) / (sample * np.sqrt(df))

    if len(noise_fft) > 2:
        amp_dens[1:-1] = (
            np.sqrt(2.0) * np.abs(noise_fft[1:-1]) / (sample * np.sqrt(df))
        )

    if len(noise_fft) > 1:
        if sample % 2 == 0:
            amp_dens[-1] = np.abs(noise_fft[-1]) / (sample * np.sqrt(df))
        else:
            amp_dens[-1] = (
                np.sqrt(2.0) * np.abs(noise_fft[-1]) / (sample * np.sqrt(df))
            )

    return amp_dens


def make_noise_total(para):
    n_abs = para["n_abs"]
    c_abs = para["C_abs"]
    c_tes = para["C_tes"]
    g_abs_abs = float(para["G_abs-abs"])
    g_abs_tes = para["G_abs-tes"]
    g_tes_bath = para["G_tes-bath"]
    r_tes = para["R"]
    r_load = para["R_l"]
    t_c = para["T_c"]
    t_bath = para["T_bath"]
    alpha = para["alpha"]
    beta = para["beta"]
    inductance = para["L"]
    n_dim = para["n"]
    rate = int(para["rate"])
    samples = int(para["samples"])

    frequency = np.arange(0, rate, rate / samples)
    current = np.sqrt(
        (g_tes_bath * t_c * (1 - ((t_bath / t_c) ** n_dim))) / (n_dim * r_tes)
    )

    t_el = inductance / (r_load + r_tes * (1 + beta))
    loop_gain = (alpha * (current**2) * r_tes) / (g_tes_bath * t_c)
    t_i = c_tes / ((1 - loop_gain) * g_tes_bath)

    ptfn_tes_bath = np.sqrt(4 * k_b * t_c**2 * g_tes_bath * ptfn_Flink)
    ptfn_abs_tes = np.sqrt(4 * k_b * t_c**2 * g_abs_tes * ptfn_Flink)
    ptfn_abs_abs = np.sqrt(4 * k_b * t_c**2 * g_abs_abs * ptfn_Flink)
    enj = np.sqrt(4 * k_b * t_c * r_tes * (1 + 2 * beta + beta**2))
    enj_r = np.sqrt(4 * k_b * t_bath * r_load)

    n_matrix = np.zeros((5, 9), dtype=np.complex128)
    n_matrix[0, 0] = -enj / inductance
    n_matrix[0, 1] = enj_r / inductance
    n_matrix[1, 0] = current * enj / c_tes
    n_matrix[1, 2] = ptfn_tes_bath / c_tes
    n_matrix[1, 3] = ptfn_abs_tes / c_tes
    n_matrix[1, 4] = ptfn_abs_abs / c_tes
    n_matrix[2, 4] = -ptfn_abs_tes / c_abs
    n_matrix[2, 5] = -2 * ptfn_abs_abs / c_abs
    n_matrix[2, 6] = -ptfn_abs_tes / c_abs
    n_matrix[3, 4] = ptfn_abs_abs / c_tes
    n_matrix[3, 5] = ptfn_abs_tes / c_tes
    n_matrix[3, 6] = ptfn_tes_bath / c_tes
    n_matrix[3, 8] = current * enj / c_tes
    n_matrix[4, 7] = enj_r / inductance
    n_matrix[4, 8] = -enj / inductance

    def matrix_m(omega):
        x = np.zeros((5, 5), dtype=np.complex128)
        x[0, 0] = 1 / t_el + omega * 1.0j
        x[0, 1] = loop_gain * g_tes_bath / (current * inductance)
        x[1, 0] = -current * r_tes * (2 + beta) / c_tes
        x[1, 1] = 1 / t_i + (g_abs_tes / c_tes) + omega * 1.0j
        x[1, 2] = -g_abs_tes / c_tes
        x[2, 1] = -g_abs_tes / c_abs
        x[2, 2] = 2 * g_abs_abs / c_abs + omega * 1.0j
        x[2, 3] = -g_abs_tes / c_abs
        x[3, 2] = -g_abs_tes / c_tes
        x[3, 3] = 1 / t_i + (g_abs_tes / c_tes) + omega * 1.0j
        x[3, 4] = -current * r_tes * (2 + beta) / c_tes
        x[4, 3] = loop_gain * g_tes_bath / (current * inductance)
        x[4, 4] = 1 / t_el + omega * 1.0j
        return x

    total_noise = np.zeros(samples, dtype=float)
    for idx, freq in enumerate(frequency):
        m_inv = np.linalg.inv(matrix_m(freq * 2 * np.pi))
        total_noise[idx] = np.sum(np.abs(m_inv[0, :] @ n_matrix))
    return total_noise


def save_noise_asd(para, noise_spe_dens, histories, seed):
    sample = int(para["samples"])
    rate = float(para["rate"])
    noise_spe_dens = np.asarray(noise_spe_dens[: int(sample / 2) + 1], dtype=float)
    power_model = np.zeros(len(noise_spe_dens))
    rng = np.random.default_rng(seed)
    for _ in range(histories):
        noise_time = generate_noise_from_asd(noise_spe_dens, sample, rate, rng=rng)
        noise_time = general.Bessel(noise_time, rate, 10000)
        noise_time = general.Bessel(noise_time, rate, para["cutoff"])
        noise_fft = np.fft.rfft(noise_time)
        power_model += np.abs(noise_fft) ** 2

    power_model /= histories
    amp_dens = np.sqrt(power_model)
    amp_dens = asd_from_rfft(amp_dens, sample, rate) * eta * 1e6
    if len(amp_dens) > 1:
        amp_dens = amp_dens[:-1]
    return amp_dens


def make_reference(pre_values, smooth_window):
    return smooth_log_spectrum(pre_values, smooth_window)


def build_weights(freq):
    safe_freq = np.maximum(freq, 1.0)
    logf = np.log10(safe_freq)
    center = np.log10(1.0e4)
    sigma = 0.55
    mid = np.exp(-0.5 * ((logf - center) / sigma) ** 2)
    base = np.ones_like(freq)
    weights = 0.35 * base + 0.65 * mid
    weights[freq < 150.0] *= 0.2
    weights[freq > 8.0e4] *= 0.35
    return weights


def compare_spectra(model, reference, freq):
    n = min(len(model), len(reference), len(freq))
    model = np.asarray(model[:n], dtype=float)
    reference = np.asarray(reference[:n], dtype=float)
    freq = np.asarray(freq[:n], dtype=float)

    mask = freq > 0
    model = model[mask]
    reference = reference[mask]
    freq = freq[mask]

    safe_model = np.maximum(model, np.finfo(float).tiny)
    safe_ref = np.maximum(reference, np.finfo(float).tiny)
    log_model = np.log(safe_model)
    log_ref = np.log(safe_ref)
    weights = build_weights(freq)

    amp_term = np.average((log_model - log_ref) ** 2, weights=weights)

    slope_weights = weights[1:]
    slope_term = np.average((np.diff(log_model) - np.diff(log_ref)) ** 2, weights=slope_weights)

    center_mask = (freq >= 2.0e3) & (freq <= 3.0e4)
    center_term = np.mean(np.abs(log_model[center_mask] - log_ref[center_mask]))

    return amp_term + 0.7 * slope_term + 0.8 * center_term


def parameter_specs(base_para):
    return [
        ("C_abs", float(base_para["C_abs"]) * 0.03, float(base_para["C_abs"]) * 4.0),
        ("C_tes", float(base_para["C_tes"]) * 0.03, float(base_para["C_tes"]) * 4.0),
        ("G_abs-abs", float(base_para["G_abs-abs"]) * 0.03, float(base_para["G_abs-abs"]) * 4.0),
        ("G_abs-tes", float(base_para["G_abs-tes"]) * 0.03, float(base_para["G_abs-tes"]) * 4.0),
        ("G_tes-bath", float(base_para["G_tes-bath"]) * 0.03, float(base_para["G_tes-bath"]) * 4.0),
        ("T_bath", float(base_para["T_bath"]) * 0.9, float(base_para["T_bath"]) * 1.1),
        ("R", float(base_para["R"]) * 0.9, float(base_para["R"]) * 1.1),
        ("R_l", float(base_para["R_l"]) * 0.9, float(base_para["R_l"]) * 1.1),
        ("L", float(base_para["L"]) * 0.9, float(base_para["L"]) * 1.1),
        ("alpha", max(0.0, float(base_para["alpha"]) * 0.1), 180.0),
        ("beta", 0.0, 5.0),
    ]


def tune_parameters(input_path, pre_path, output_path, report_path, histories, maxiter, seed, smooth_window):
    base_para = load_json(input_path)
    pre_values = np.loadtxt(pre_path)
    reference = make_reference(pre_values, smooth_window)
    sample = int(base_para["samples"])
    rate = float(base_para["rate"])
    freq = np.fft.rfftfreq(sample, d=1 / rate)
    if len(freq) > 1:
        freq = freq[:-1]

    specs = parameter_specs(base_para)
    keys = [item[0] for item in specs]
    bounds = [(item[1], item[2]) for item in specs]
    x0 = np.array([float(base_para[key]) for key in keys], dtype=float)

    cache = {}
    history = []

    def objective(x):
        rounded = tuple(float(f"{value:.12g}") for value in x)
        if rounded in cache:
            return cache[rounded]

        para = copy.deepcopy(base_para)
        for key, value in zip(keys, x):
            para[key] = float(value)

        try:
            total_noise = make_noise_total(para)
            model = save_noise_asd(para, total_noise, histories=histories, seed=seed)
            score = compare_spectra(model, reference, freq)
        except Exception:
            score = 1e12

        cache[rounded] = score
        history.append({"score": float(score), "params": {key: float(value) for key, value in zip(keys, x)}})
        print(f"score={score:.6f} " + " ".join(f"{key}={value:.6g}" for key, value in zip(keys, x)))
        return score

    initial_score = objective(x0)

    result = scipy.optimize.minimize(
        objective,
        x0,
        method="Powell",
        bounds=bounds,
        options={"maxiter": maxiter, "disp": True},
    )

    best_para = copy.deepcopy(base_para)
    for key, value in zip(keys, result.x):
        best_para[key] = float(value)

    total_noise = make_noise_total(best_para)
    tuned_amp = save_noise_asd(best_para, total_noise, histories=histories, seed=seed)
    verification_score = compare_spectra(tuned_amp, reference, freq)

    save_json(output_path, best_para)
    np.savetxt(output_path.with_name("noise_total_tuned.dat"), total_noise)
    np.savetxt(output_path.with_name("noise_total-bessel100k-tuned.dat"), tuned_amp)

    report = {
        "input_path": str(input_path),
        "pre_path": str(pre_path),
        "output_path": str(output_path),
        "initial_score": float(initial_score),
        "search_score": float(result.fun),
        "verification_score": float(verification_score),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "histories": histories,
        "maxiter": maxiter,
        "seed": seed,
        "smooth_window": smooth_window,
        "initial_params": {key: float(base_para[key]) for key in keys},
        "best_params": {key: float(best_para[key]) for key in keys},
        "evaluations": history,
    }
    save_json(report_path, report)
    return report


def build_parser():
    parser = argparse.ArgumentParser(description="Tune input.json against noise_total-bessel100k-pre.dat")
    parser.add_argument("--input", type=Path, required=True, help="Path to input.json")
    parser.add_argument("--pre", type=Path, required=True, help="Path to noise_total-bessel100k-pre.dat")
    parser.add_argument("--output", type=Path, help="Path to write tuned input.json")
    parser.add_argument("--report", type=Path, help="Path to write tuning report JSON")
    parser.add_argument("--histories", type=int, default=24, help="Monte Carlo histories per evaluation")
    parser.add_argument("--maxiter", type=int, default=30, help="Powell iterations")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for repeatable SaveNoise sampling")
    parser.add_argument("--smooth-window", type=int, default=41, help="Smoothing window for the pre spectrum")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_path = args.output or args.input.with_name("input_tuned.json")
    report_path = args.report or args.input.with_name("input_tuned_report.json")

    report = tune_parameters(
        input_path=args.input,
        pre_path=args.pre,
        output_path=output_path,
        report_path=report_path,
        histories=args.histories,
        maxiter=args.maxiter,
        seed=args.seed,
        smooth_window=args.smooth_window,
    )
    print("initial score:", report["initial_score"])
    print("search score:", report["search_score"])
    print("verification score:", report["verification_score"])
    print("tuned input:", output_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
