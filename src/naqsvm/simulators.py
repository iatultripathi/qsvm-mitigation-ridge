"""Fidelity quantum-kernel circuits and exact (infinite-shot) noisy probabilities.

Kernel definition (established; Havlicek et al. 2019, Schuld & Killoran 2019):
    k(x, y) = |<psi(y)|psi(x)>|^2 = Pr[ measure 0...0 on U(y)^dagger U(x)|0...0> ]
estimated on hardware with the compute-uncompute circuit.

Two interchangeable backends compute p = Pr[0...0] *exactly* (density matrix):
  * 'aer'   : Qiskit Aer AerSimulator(method='density_matrix') + NoiseModel
  * 'numpy' : a vectorised NumPy density-matrix simulator of the *same* gate list
              and the *same* noise channels (reference implementation; used when
              Qiskit cannot be installed, and cross-checked against Aer otherwise).

Finite-shot measurement is simulated separately (shots.py) as Binomial(S, p).
This is exact, not an approximation: S i.i.d. shots of the same circuit each
return 0...0 with probability p, so the number of 0...0 outcomes ~ Binomial(S, p).
The notebook validates this against Aer shot-based sampling.

Readout error (symmetric, per qubit, flip prob e) is applied analytically:
    p_obs(0...0) = sum_s P(s) (1-e)^(n-|s|) e^|s|      (|s| = Hamming weight)
which is what Aer's ReadoutError produces for the all-zero outcome.
"""
from __future__ import annotations

import time
import warnings
from typing import List, Tuple

import numpy as np

from .config import NoiseSpec

# ----------------------------------------------------------------------------
# Circuit template: one source of truth for both backends
# ----------------------------------------------------------------------------
# Each op: (gate, qubits, angle_spec). angle_spec is None for h/cx, else
# ('single', side, q)        -> angle = sign * 2 * v[q]
# ('pair',   side, i, j)     -> angle = sign * 2 * (pi - v[i]) * (pi - v[j])
# where side='x' (forward, sign=+1) or 'y' (inverse, sign=-1).


def _pairs(n: int, entanglement: str) -> List[Tuple[int, int]]:
    if entanglement == "linear":
        return [(i, i + 1) for i in range(n - 1)]
    if entanglement == "full":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    raise ValueError(f"unknown entanglement {entanglement!r}")


def zz_feature_map_ops(n: int, reps: int, entanglement: str, side: str = "x"):
    """Gate list of the ZZ feature map U(v) (same structure as Qiskit's ZZFeatureMap)."""
    ops = []
    for _ in range(reps):
        for q in range(n):
            ops.append(("h", (q,), None))
        for q in range(n):
            ops.append(("p", (q,), ("single", side, q)))
        for (i, j) in _pairs(n, entanglement):
            ops.append(("cx", (i, j), None))
            ops.append(("p", (j,), ("pair", side, i, j)))
            ops.append(("cx", (i, j), None))
    return ops


def kernel_circuit_ops(n: int, reps: int, entanglement: str):
    """U(y)^dagger U(x): forward map on x, then the inverse map on y."""
    fwd = zz_feature_map_ops(n, reps, entanglement, side="x")
    inv = []
    for (g, qs, spec) in reversed(zz_feature_map_ops(n, reps, entanglement, side="y")):
        inv.append((g, qs, spec))  # H and CX are self-inverse; P angles get sign -1 via side='y'
    return fwd + inv


def op_angles(spec, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorised angles for a batch of (x, y) pairs."""
    kind, side = spec[0], spec[1]
    V = X if side == "x" else Y
    sign = 1.0 if side == "x" else -1.0
    if kind == "single":
        return sign * 2.0 * V[:, spec[2]]
    return sign * 2.0 * (np.pi - V[:, spec[2]]) * (np.pi - V[:, spec[3]])


def count_gates(n: int, reps: int, entanglement: str):
    ops = kernel_circuit_ops(n, reps, entanglement)
    return {g: sum(1 for o in ops if o[0] == g) for g in ("h", "p", "cx")}


# ----------------------------------------------------------------------------
# Ideal kernel (statevector, exact)
# ----------------------------------------------------------------------------
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def feature_statevectors(X: np.ndarray, reps: int, entanglement: str) -> np.ndarray:
    """|psi(x)> for each row of X, shape (B, 2**n). Vectorised NumPy statevector."""
    B, n = X.shape
    psi = np.zeros((B,) + (2,) * n, dtype=complex)
    psi[(slice(None),) + (0,) * n] = 1.0
    for (g, qs, spec) in zz_feature_map_ops(n, reps, entanglement, side="x"):
        if g == "h":
            psi = np.moveaxis(np.tensordot(psi, _H, axes=([1 + qs[0]], [1])), -1, 1 + qs[0])
        elif g == "p":
            th = op_angles(spec, X, X)
            idx = [slice(None)] * (n + 1)
            idx[1 + qs[0]] = 1
            psi[tuple(idx)] *= np.exp(1j * th).reshape((B,) + (1,) * (n - 1))
        elif g == "cx":
            c, t = qs
            idx = [slice(None)] * (n + 1)
            idx[1 + c] = 1
            sub = psi[tuple(idx)]                      # control = 1 block
            tax = 1 + t - (1 if t > c else 0)          # target axis inside the block
            psi[tuple(idx)] = np.flip(sub, axis=tax).copy()
    return psi.reshape(B, 2 ** n)


def ideal_kernel(XA: np.ndarray, XB: np.ndarray, reps: int, entanglement: str) -> np.ndarray:
    SA = feature_statevectors(XA, reps, entanglement)
    SB = feature_statevectors(XB, reps, entanglement)
    return np.abs(SA.conj() @ SB.T) ** 2


# ----------------------------------------------------------------------------
# NumPy density-matrix reference simulator (batched over circuit instances)
# ----------------------------------------------------------------------------
class NumpyDensityMatrixSimulator:
    """Exact density-matrix simulation of the kernel circuit for many (x, y) pairs.

    Noise channels (applied after the gate, as in Qiskit Aer noise models):
      depolarizing (Qiskit convention): E(rho) = (1-p) rho + p Tr_q(rho) (x) I/2^k
      thermal relaxation (T1, T2, gate time t, zero excited-state population):
          rho00 += g rho11, rho11 *= (1-g), rho01 *= exp(-t/T2), g = 1-exp(-t/T1)
    """

    def __init__(self, n: int, reps: int, entanglement: str, noise: NoiseSpec):
        self.n, self.reps, self.ent, self.noise = n, reps, entanglement, noise
        self.ops = kernel_circuit_ops(n, reps, entanglement)
        if noise.kind == "thermal" and noise.t2 > 2 * noise.t1:
            raise ValueError("thermal relaxation requires T2 <= 2*T1")

    # rho has shape (B, 2,..,2 [rows n], 2,..,2 [cols n])
    def _move(self, rho, qs):
        n = self.n
        src = [1 + q for q in qs] + [1 + n + q for q in qs]
        dst = list(range(rho.ndim - 2 * len(qs), rho.ndim))
        return np.moveaxis(rho, src, dst), src, dst

    def _back(self, rho, src, dst):
        return np.moveaxis(rho, dst, src)

    def _h(self, rho, q):
        r, s, d = self._move(rho, (q,))
        r = np.einsum("ab,...bc,dc->...ad", _H, r, _H.conj(), optimize=True)
        return self._back(r, s, d)

    def _p(self, rho, q, th):
        r, s, d = self._move(rho, (q,))
        r = r.copy()
        ph = np.exp(1j * th).reshape((-1,) + (1,) * (r.ndim - 3))
        r[..., 1, 0] *= ph
        r[..., 0, 1] *= ph.conj()
        return self._back(r, s, d)

    def _cx(self, rho, c, t):
        r, s, d = self._move(rho, (c, t))     # last axes: [rc, rt, cc, ct]
        r = r.copy()
        r[..., 1, :, :, :] = r[..., 1, ::-1, :, :].copy()
        r[..., :, :, 1, :] = r[..., :, :, 1, ::-1].copy()
        return self._back(r, s, d)

    def _depol1(self, rho, q, p):
        if p <= 0:
            return rho
        r, s, d = self._move(rho, (q,))
        tr = r[..., 0, 0] + r[..., 1, 1]
        out = (1 - p) * r
        out[..., 0, 0] += 0.5 * p * tr
        out[..., 1, 1] += 0.5 * p * tr
        return self._back(out, s, d)

    def _depol2(self, rho, a, b, p):
        if p <= 0:
            return rho
        r, s, d = self._move(rho, (a, b))     # [ra, rb, ca, cb]
        tr = sum(r[..., i, j, i, j] for i in (0, 1) for j in (0, 1))
        out = (1 - p) * r
        for i in (0, 1):
            for j in (0, 1):
                out[..., i, j, i, j] += 0.25 * p * tr
        return self._back(out, s, d)

    def _thermal1(self, rho, q, t):
        nz = self.noise
        g = 1.0 - np.exp(-t / nz.t1)
        f = np.exp(-t / nz.t2)
        r, s, d = self._move(rho, (q,))
        out = r.copy()
        out[..., 0, 0] = r[..., 0, 0] + g * r[..., 1, 1]
        out[..., 1, 1] = (1 - g) * r[..., 1, 1]
        out[..., 0, 1] = f * r[..., 0, 1]
        out[..., 1, 0] = f * r[..., 1, 0]
        return self._back(out, s, d)

    def _noise_after(self, rho, g, qs):
        nz = self.noise
        if nz.kind == "depolarizing":
            if g == "h":
                rho = self._depol1(rho, qs[0], nz.p1)
            elif g == "cx":
                rho = self._depol2(rho, qs[0], qs[1], nz.p2)
        elif nz.kind == "thermal":
            if g == "h":
                rho = self._thermal1(rho, qs[0], nz.t_1q)
            elif g == "cx":
                rho = self._thermal1(rho, qs[0], nz.t_2q)
                rho = self._thermal1(rho, qs[1], nz.t_2q)
        return rho

    def basis_probabilities(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        B, n = X.shape
        D = 2 ** n
        rho = np.zeros((B,) + (2,) * (2 * n), dtype=complex)
        rho[(slice(None),) + (0,) * (2 * n)] = 1.0
        for (g, qs, spec) in self.ops:
            if g == "h":
                rho = self._h(rho, qs[0])
            elif g == "p":
                rho = self._p(rho, qs[0], op_angles(spec, X, Y))
            elif g == "cx":
                rho = self._cx(rho, qs[0], qs[1])
            rho = self._noise_after(rho, g, qs)
        diag = np.real(np.diagonal(rho.reshape(B, D, D), axis1=1, axis2=2))
        return np.clip(diag, 0.0, 1.0)


def apply_readout_all_zero(P: np.ndarray, n: int, e: float) -> np.ndarray:
    """Pr[observe 0...0] given true basis probabilities P (B, 2**n), symmetric flip e."""
    if e <= 0:
        return P[:, 0].copy()
    w = np.array([bin(s).count("1") for s in range(2 ** n)])
    weights = (1 - e) ** (n - w) * e ** w
    return P @ weights


# ----------------------------------------------------------------------------
# Qiskit / Aer backend
# ----------------------------------------------------------------------------
def qiskit_available() -> bool:
    try:
        import qiskit  # noqa: F401
        import qiskit_aer  # noqa: F401
        return True
    except Exception:
        return False


def build_qiskit_circuit(x: np.ndarray, y: np.ndarray, reps: int, entanglement: str,
                         measure: bool = False):
    """The same kernel circuit as a qiskit.QuantumCircuit (gates h, p, cx)."""
    from qiskit import QuantumCircuit
    n = len(x)
    qc = QuantumCircuit(n)
    X, Y = x[None, :], y[None, :]
    for (g, qs, spec) in kernel_circuit_ops(n, reps, entanglement):
        if g == "h":
            qc.h(qs[0])
        elif g == "p":
            qc.p(float(op_angles(spec, X, Y)[0]), qs[0])
        elif g == "cx":
            qc.cx(qs[0], qs[1])
    if measure:
        qc.measure_all()
    return qc


def build_aer_noise_model(noise: NoiseSpec, include_readout: bool = False):
    """Qiskit Aer NoiseModel matching NumpyDensityMatrixSimulator."""
    if noise.kind == "none" and not (include_readout and noise.readout > 0):
        return None
    from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                                  thermal_relaxation_error, ReadoutError)
    nm = NoiseModel(basis_gates=["h", "p", "cx"])
    if noise.kind == "depolarizing":
        if noise.p1 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise.p1, 1), ["h"])
        if noise.p2 > 0:
            nm.add_all_qubit_quantum_error(depolarizing_error(noise.p2, 2), ["cx"])
    elif noise.kind == "thermal":
        e1 = thermal_relaxation_error(noise.t1, noise.t2, noise.t_1q)
        e2 = thermal_relaxation_error(noise.t1, noise.t2, noise.t_2q)
        nm.add_all_qubit_quantum_error(e1, ["h"])
        nm.add_all_qubit_quantum_error(e2.expand(e2), ["cx"])
    if include_readout and noise.readout > 0:
        e = noise.readout
        nm.add_all_qubit_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]))
    return nm


def aer_basis_probabilities(X, Y, reps, entanglement, noise: NoiseSpec, chunk=1000):
    from qiskit_aer import AerSimulator
    nm = build_aer_noise_model(noise, include_readout=False)
    sim = AerSimulator(method="density_matrix", noise_model=nm) if nm is not None \
        else AerSimulator(method="density_matrix")
    B, n = X.shape
    out = np.empty((B, 2 ** n))
    for s in range(0, B, chunk):
        circs = []
        for b in range(s, min(B, s + chunk)):
            qc = build_qiskit_circuit(X[b], Y[b], reps, entanglement)
            qc.save_probabilities(label="probs")
            circs.append(qc)
        res = sim.run(circs, shots=1).result()
        for k in range(len(circs)):
            out[s + k] = np.asarray(res.data(k)["probs"], dtype=float)
    return np.clip(out, 0.0, 1.0)


def aer_shot_counts_all_zero(X, Y, reps, entanglement, noise: NoiseSpec, shots: int, seed: int):
    """Real shot-based Aer sampling (with readout error) -> counts of '0...0'."""
    from qiskit_aer import AerSimulator
    nm = build_aer_noise_model(noise, include_readout=True)
    kw = dict(method="density_matrix", seed_simulator=seed)
    sim = AerSimulator(noise_model=nm, **kw) if nm is not None else AerSimulator(**kw)
    circs = [build_qiskit_circuit(X[b], Y[b], reps, entanglement, measure=True) for b in range(len(X))]
    res = sim.run(circs, shots=shots).result()
    n = X.shape[1]
    key = "0" * n
    return np.array([res.get_counts(k).get(key, 0) for k in range(len(circs))])


# ----------------------------------------------------------------------------
# Unified front end
# ----------------------------------------------------------------------------
def resolve_backend(requested: str) -> str:
    if requested == "numpy":
        return "numpy"
    ok = qiskit_available()
    if requested == "aer" and not ok:
        raise RuntimeError("sim_backend='aer' requested but qiskit/qiskit-aer are not importable")
    return "aer" if ok else "numpy"


def exact_all_zero_probability(X, Y, reps, entanglement, noise: NoiseSpec,
                               backend: str = "numpy", chunk: int = 1500):
    """Exact Pr[observe 0...0] (incl. readout error) for each pair (X[b], Y[b]).

    Returns (p, backend_used, seconds). If the Aer backend raises, falls back to
    the NumPy reference simulator and reports it (never silently).
    """
    t0 = time.perf_counter()
    n = X.shape[1]
    if noise.kind == "global":
        # synthetic GLOBAL depolarizing channel on the whole kernel circuit (analytic):
        # p = Lam * k_ideal + (1 - Lam)/d. Used only to verify the hidden-ridge identity exactly.
        S1 = feature_statevectors(X, reps, entanglement)
        S2 = feature_statevectors(Y, reps, entanglement)
        k = np.abs(np.sum(S2.conj() * S1, axis=1)) ** 2
        return np.clip(noise.p1 * k + (1 - noise.p1) / 2 ** n, 0, 1), "analytic", time.perf_counter() - t0
    used = backend
    if backend == "aer":
        try:
            P = aer_basis_probabilities(X, Y, reps, entanglement, noise, chunk=chunk)
        except Exception as exc:  # pragma: no cover - depends on installed qiskit
            warnings.warn(f"Aer simulation failed ({exc!r}); falling back to NumPy reference simulator.")
            used = "numpy(fallback)"
            backend = "numpy"
    if backend == "numpy":
        sim = NumpyDensityMatrixSimulator(n, reps, entanglement, noise)
        P = np.vstack([sim.basis_probabilities(X[s:s + chunk], Y[s:s + chunk])
                       for s in range(0, len(X), chunk)]) if len(X) else np.zeros((0, 2 ** n))
    p = apply_readout_all_zero(P, n, noise.readout if noise.kind != "none" else 0.0)
    return np.clip(p, 0.0, 1.0), used, time.perf_counter() - t0
