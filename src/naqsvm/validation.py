"""Self-checks: simulator correctness, statistical model, mathematical claims, leakage.

Every check returns (name, passed, blocking, detail). Core checks are blocking: the notebook
stops if one fails. Qiskit-dependent checks are non-blocking: they are skipped (and reported)
when Qiskit/Aer is not importable, and if one fails the notebook switches the grid to the
independently verified NumPy reference simulator and records that decision.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from . import kernels as kx
from .config import NoiseSpec
from .shots import largest_remainder, measure_two_phase, measure_uniform
from .simulators import (NumpyDensityMatrixSimulator, apply_readout_all_zero,
                         exact_all_zero_probability, feature_statevectors, ideal_kernel,
                         kernel_circuit_ops, op_angles, qiskit_available, zz_feature_map_ops)

# ---------------------------------------------------------------------------
# Independent dense-matrix implementation (kron products, Kraus/Pauli forms)
# ---------------------------------------------------------------------------
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_Hd = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def _embed(ops_by_qubit, n):
    M = np.array([[1.0 + 0j]])
    for q in range(n):
        M = np.kron(M, ops_by_qubit.get(q, _I))
    return M


def _cx_dense(c, t, n):
    D = 2 ** n
    M = np.zeros((D, D), dtype=complex)
    for i in range(D):
        bits = [(i >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[c]:
            bits[t] ^= 1
        j = sum(b << (n - 1 - q) for q, b in enumerate(bits))
        M[j, i] = 1
    return M


def _gate_dense(g, qs, theta, n):
    if g == "h":
        return _embed({qs[0]: _Hd}, n)
    if g == "p":
        return _embed({qs[0]: np.diag([1, np.exp(1j * theta)])}, n)
    return _cx_dense(qs[0], qs[1], n)


def _dense_depol(rho, qs, p, n):
    paulis = [_I, _X, _Y, _Z]
    k = len(qs)
    acc = np.zeros_like(rho)
    for combo in itertools.product(paulis, repeat=k):
        P = _embed(dict(zip(qs, combo)), n)
        acc += P @ rho @ P.conj().T
    return (1 - p) * rho + p * acc / 4 ** k


def _dense_thermal(rho, q, t, t1, t2, n):
    g = 1 - np.exp(-t / t1)
    lam_pd = 1 - np.exp(-2 * t * (1 / t2 - 1 / (2 * t1)))
    kraus = [np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])]
    rho = sum(_embed({q: K}, n) @ rho @ _embed({q: K}, n).conj().T for K in kraus)
    kraus = [np.array([[1, 0], [0, np.sqrt(1 - lam_pd)]]), np.array([[0, 0], [0, np.sqrt(lam_pd)]])]
    return sum(_embed({q: K}, n) @ rho @ _embed({q: K}, n).conj().T for K in kraus)


def dense_probabilities(x, y, reps, ent, noise: NoiseSpec):
    n = len(x)
    rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
    rho[0, 0] = 1
    X, Y = x[None], y[None]
    for (g, qs, spec) in kernel_circuit_ops(n, reps, ent):
        th = float(op_angles(spec, X, Y)[0]) if spec else 0.0
        U = _gate_dense(g, qs, th, n)
        rho = U @ rho @ U.conj().T
        if noise.kind == "depolarizing":
            if g == "h" and noise.p1 > 0:
                rho = _dense_depol(rho, qs, noise.p1, n)
            if g == "cx" and noise.p2 > 0:
                rho = _dense_depol(rho, qs, noise.p2, n)
        elif noise.kind == "thermal":
            tt = noise.t_1q if g == "h" else noise.t_2q
            if g in ("h", "cx"):
                for q in qs:
                    rho = _dense_thermal(rho, q, tt, noise.t1, noise.t2, n)
    return np.real(np.diag(rho))


# ---------------------------------------------------------------------------
def run_all(cfg, include_qiskit: bool = True, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []

    def add(name, ok, detail, blocking=True):
        out.append(dict(check=name, passed=bool(ok) if ok is not None else None, blocking=blocking,
                        detail=detail))

    n, reps, ent = cfg.n_qubits, cfg.reps, cfg.entanglement
    X = rng.uniform(0, np.pi / 2, (6, n))
    Y = rng.uniform(0, np.pi / 2, (6, n))

    # 1. statevector vs independent dense unitary
    S = feature_statevectors(X, reps, ent)
    err = 0.0
    for b in range(len(X)):
        psi = np.zeros(2 ** n, dtype=complex); psi[0] = 1
        for (g, qs, spec) in zz_feature_map_ops(n, reps, ent):
            th = float(op_angles(spec, X[b:b + 1], X[b:b + 1])[0]) if spec else 0.0
            psi = _gate_dense(g, qs, th, n) @ psi
        err = max(err, np.abs(psi - S[b]).max())
    add("statevector == dense kron simulation", err < 1e-10, f"max |diff| = {err:.2e}")

    # 2. noisy density matrix vs dense (depolarizing, thermal), incl. readout
    for nz in [NoiseSpec("dep", p1=0.01, p2=0.05, readout=0.03),
               NoiseSpec("thermal", kind="thermal", t1=40e-6, t2=30e-6, t_1q=50e-9, t_2q=500e-9, readout=0.02)]:
        p_np, _, _ = exact_all_zero_probability(X, Y, reps, ent, nz, backend="numpy")
        Pd = np.array([dense_probabilities(X[b], Y[b], reps, ent, nz) for b in range(len(X))])
        p_dense = apply_readout_all_zero(Pd, n, nz.readout)
        e = np.abs(p_np - p_dense).max()
        add(f"NumPy DM simulator == dense Kraus/Pauli ({nz.kind})", e < 1e-10, f"max |diff| = {e:.2e}")

    # 3. noise-free DM probability == ideal kernel; diagonal == 1
    p0, _, _ = exact_all_zero_probability(X, Y, reps, ent, NoiseSpec("none", kind="none"), backend="numpy")
    K = ideal_kernel(X, Y, reps, ent)
    add("noise-free Pr[0..0] == |<psi(y)|psi(x)>|^2", np.abs(p0 - np.diag(K)).max() < 1e-10,
        f"max |diff| = {np.abs(p0 - np.diag(K)).max():.2e}")
    Kxx = ideal_kernel(X, X, reps, ent)
    add("ideal kernel PSD and unit diagonal", kx.is_psd(Kxx) and np.allclose(np.diag(Kxx), 1),
        f"min eig = {kx.min_eig(Kxx):.2e}")

    # 4. binomial estimator unbiased with the stated variance
    p = np.array([0.02, 0.2, 0.5, 0.9])
    Sh = 64
    est = np.array([measure_uniform(p, Sh * len(p), rng).estimate for _ in range(20000)])
    bias = np.abs(est.mean(0) - p).max()
    vr = (est.var(0) / (p * (1 - p) / Sh))
    add("k_hat = n/S unbiased, Var = p(1-p)/S", bias < 0.003 and np.all(np.abs(vr - 1) < 0.05),
        f"max bias {bias:.4f}; var ratio {np.round(vr, 3).tolist()}")

    # 5. allocation integrality and Neyman variance reduction
    w = rng.uniform(0, 1, 37)
    a = largest_remainder(w, 1000)
    add("largest-remainder allocation sums exactly", a.sum() == 1000 and a.min() >= 0, f"sum={a.sum()}")
    p = np.concatenate([np.full(200, 0.01), np.full(200, 0.5)])
    B = 64 * len(p)
    mse_u = np.mean([np.sum((measure_uniform(p, B, rng).estimate - p) ** 2) for _ in range(400)])
    mse_n = np.mean([np.sum((measure_two_phase(p, B, rng, 0.25).estimate - p) ** 2) for _ in range(400)])
    add("pilot+Neyman lowers Frobenius MSE on heterogeneous p", mse_n < mse_u,
        f"MSE uniform {mse_u:.4f} vs Neyman {mse_n:.4f}")

    # 6. mitigation is exact under a global depolarizing channel
    d = 2 ** n
    lam = 0.37
    Kobs = lam * Kxx + (1 - lam) / d
    lam_hat = kx.survival_from_diag(np.diag(Kobs), d, 0.0)
    Kc = kx.mitigate(Kobs, lam_hat.mean(), d)
    add("depolarizing inversion recovers ideal kernel exactly", np.abs(Kc - Kxx).max() < 1e-12,
        f"max |diff| = {np.abs(Kc - Kxx).max():.2e}")

    # 7. SVM invariance to K -> s*K (with C -> C/s) and K -> K + c*11^T  (binary)
    from sklearn.svm import SVC
    Z = rng.normal(size=(40, 3)); yb = (Z[:, 0] + 0.3 * rng.normal(size=40) > 0).astype(int)
    Kz = np.exp(-0.5 * ((Z[:, None] - Z[None]) ** 2).sum(-1))
    m1 = SVC(kernel="precomputed", C=1.0, tol=1e-8).fit(Kz, yb)
    m2 = SVC(kernel="precomputed", C=1.0 / 0.3, tol=1e-8).fit(0.3 * Kz, yb)
    m3 = SVC(kernel="precomputed", C=1.0, tol=1e-8).fit(Kz + 0.7, yb)
    f1, f2, f3 = m1.decision_function(Kz), m2.decision_function(0.3 * Kz), m3.decision_function(Kz + 0.7)
    add("SVM decision invariant to kernel scaling (C rescaled) and constant offset",
        np.abs(f1 - f2).max() < 1e-4 and np.abs(f1 - f3).max() < 1e-4,
        f"max |df| scale {np.abs(f1 - f2).max():.1e}, offset {np.abs(f1 - f3).max():.1e}")

    # 8. ridge >= ||E||_2 guarantees PSD; 2*sigma scale vs realised ||E||_2
    N = 60
    Zs = rng.uniform(0, np.pi * cfg.angle_scale, (N, n))
    K0 = ideal_kernel(Zs, Zs, reps, ent)
    V = np.full((N, N), 0.004); np.fill_diagonal(V, 0)
    E = np.triu(rng.normal(0, np.sqrt(V)), 1); E = E + E.T
    nrm = np.linalg.norm(E, 2)
    from .methods import noise_spectral_scale
    rho = noise_spectral_scale(V)
    add("K + E + r I is PSD when r >= ||E||_2", kx.is_psd(K0 + E + nrm * np.eye(N)),
        f"min eig without ridge {kx.min_eig(K0 + E):.3f}, with r=||E||: {kx.min_eig(K0 + E + nrm * np.eye(N)):.2e}")
    add("noise scale 2*sigma tracks realised ||E||_2 (Gaussian test)", 0.7 < rho / nrm < 1.4,
        f"2 sigma = {rho:.3f}, ||E||_2 = {nrm:.3f}")

    # 9. PSD repair operators
    A = rng.normal(size=(20, 20)); A = A + A.T
    add("clip/flip/shift produce PSD matrices",
        all(kx.is_psd(f(A)) for f in (kx.psd_clip, kx.psd_flip, kx.psd_shift)), "")

    # 10. leakage: preprocessing statistics come from training rows only
    from .data import load_dataset, make_split, prepare
    Xd, yd = load_dataset("breast_cancer")
    sp = make_split(yd, 60, 30, 60, 3)
    pr = prepare(Xd, yd, sp, n, 0.1, 3)
    ok = (np.allclose(pr.F_train.mean(0), 0, atol=1e-10) and not np.allclose(pr.F_test.mean(0), 0, atol=1e-3)
          and np.isclose(pr.A_train.min(), 0) and np.isclose(pr.A_train.max(), 0.1 * np.pi))
    add("scalers/PCA fitted on train only; splits disjoint", ok,
        "train features standardised exactly; test only approximately (as expected)")

    # 10b. hidden-ridge identity is exact under a global depolarizing channel (Sec. 3.4)
    from .hidden_ridge import check_one
    Zt = rng.uniform(0, np.pi * cfg.angle_scale, (50, n))
    Ztest = rng.uniform(0, np.pi * cfg.angle_scale, (20, n))
    Kt, Kte = ideal_kernel(Zt, Zt, reps, ent), ideal_kernel(Ztest, Zt, reps, ent)
    yt = (Zt[:, 0] + Zt[:, 1] > np.median(Zt[:, 0] + Zt[:, 1])).astype(int)
    yte = (Ztest[:, 0] + Ztest[:, 1] > np.median(Zt[:, 0] + Zt[:, 1])).astype(int)
    lam = 0.4
    r = check_one(lam * Kt + (1 - lam) / d, lam * Kte + (1 - lam) / d, Kt, Kte, yt, yte, lam, d, 1.0, 0, rng)
    add("hidden-ridge identity exact under global depolarizing (SVM(K_raw,C) == SVM(K+rI, ΛC))",
        r["B_decision_rel_diff"] < 1e-5 and r["B_pred_agreement"] == 1.0,
        f"rel. decision diff {r['B_decision_rel_diff']:.1e} (control without ridge: {r['Z_decision_rel_diff']:.2f})")

    # 10c. prior-work allocation rules spend exactly the budget
    from .methods import measure_aqka_svm, measure_margin_neyman
    iu = np.triu_indices(50, 1)
    pk = np.clip(lam * Kt + (1 - lam) / d, 0, 1)[iu]
    m1 = measure_aqka_svm(pk, iu, 50, yt, 64 * len(pk), rng)
    m2 = measure_margin_neyman(pk, iu, 50, yt, 64 * len(pk), rng)
    add("AQKA-SVM and margin-Neyman allocations spend exactly the budget",
        m1.total_shots == 64 * len(pk) and m2.total_shots == 64 * len(pk) and m1.shots.min() > 0,
        f"AQKA {m1.total_shots}, margin-Neyman {m2.total_shots}, budget {64 * len(pk)}")

    # 11. Qiskit / Aer cross-checks
    if include_qiskit and qiskit_available():
        try:
            from .simulators import aer_basis_probabilities, aer_shot_counts_all_zero, build_qiskit_circuit
            for nz in [NoiseSpec("none", kind="none"), NoiseSpec("dep", p1=0.01, p2=0.05, readout=0.03),
                       NoiseSpec("thermal", kind="thermal", t1=40e-6, t2=30e-6, t_1q=50e-9, t_2q=500e-9)]:
                pa = aer_basis_probabilities(X, Y, reps, ent, nz)
                pn = NumpyDensityMatrixSimulator(n, reps, ent, nz).basis_probabilities(X, Y)
                e = np.abs(pa[:, 0] - pn[:, 0]).max()
                add(f"Qiskit Aer density matrix == NumPy reference ({nz.kind})", e < 1e-8, f"max |diff| = {e:.2e}", blocking=False)
            try:
                from qiskit.circuit.library import ZZFeatureMap
                from qiskit.quantum_info import Statevector
                fm = ZZFeatureMap(feature_dimension=n, reps=reps, entanglement=ent)
                fid = []
                for b in range(len(X)):
                    sv = Statevector(fm.assign_parameters(X[b])).data
                    mine = feature_statevectors(X[b:b + 1], reps, ent)[0]
                    # NumPy state uses qubit 0 as most-significant bit; Qiskit as least-significant
                    mine_q = mine.reshape((2,) * n).transpose(list(range(n))[::-1]).reshape(-1)
                    fid.append(abs(np.vdot(sv, mine_q)) ** 2)
                add("circuit == qiskit.circuit.library.ZZFeatureMap (state fidelity)",
                    min(fid) > 1 - 1e-8, f"min fidelity = {min(fid):.10f}", blocking=False)
            except Exception as exc:
                add("circuit == qiskit ZZFeatureMap", None, f"skipped: {exc!r}", blocking=False)
            nz = NoiseSpec("dep", p1=0.01, p2=0.05, readout=0.03)
            p_exact, _, _ = exact_all_zero_probability(X, Y, reps, ent, nz, backend="aer")
            shots = 20000
            cnt = aer_shot_counts_all_zero(X, Y, reps, ent, nz, shots, seed=11)
            z = (cnt - shots * p_exact) / np.sqrt(shots * p_exact * (1 - p_exact))
            add("Aer shot sampling (with readout error) consistent with Binomial(S, p_exact)",
                np.all(np.abs(z) < 4.0), f"z-scores {np.round(z, 2).tolist()}", blocking=False)
        except Exception as exc:
            add("Qiskit/Aer cross-checks", False, f"error: {exc!r}", blocking=False)
    else:
        add("Qiskit/Aer cross-checks", None, "skipped: qiskit / qiskit-aer not importable", blocking=False)
    return pd.DataFrame(out)
