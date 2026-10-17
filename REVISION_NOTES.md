# Revision v3: how the four weaknesses were addressed

Run on 17–18 Oct 2026. Simulator: the package's NumPy density-matrix reference simulator, which the v2 self-checks
validated against Qiskit Aer to ~1e-16. Qiskit could not be installed in this run's environment. The new device-noise
simulator was checked against the v2 depolarizing simulator (max difference 4.9e-15) and against its own slow
reference implementation (max difference 1.7e-16).
Auto-generated numbers: `REVISION_FINDINGS_raw.md`. Theory text for the paper: `REVISION_THEORY.md`.

| Run | Design | Rows |
|---|---|---|
| R1 mechanism study | original v2 design: 4 datasets × 10 seeds (2000–2009), N=60, 4 qubits; 9 noise settings (global Λ=0.8/0.5/0.25, low/medium/high, thermal, device A, device B) × 5 shot budgets (exact, 1024, 256, 64, 16) | 1,800 cells |
| R2-A scale-up | 4 qubits, **N_train=200** (val 60, test 100); breast cancer, synthetic (1000 pts), digits 3-vs-8, quantum-structured; medium, high, device A, device B; shots 64/256/1024; seeds 3000–3002 | 144 cells |
| R2-B scale-up | **6 qubits**, N_train=80; breast cancer, synthetic, quantum-structured; device A, device B; same shots and seeds | 54 cells |

---

## W1: "C1 does not hold at low shot counts" → solved (theory + exact test)

**New result (Corollary 1, proof in `REVISION_THEORY.md`).** Global ("average") diagonal mitigation is a scalar affine
map. So for *any* measured kernel, *any* Λ̂ and *any* noise:

SVM(K_raw, C) ≡ SVM(K_mit + r̂·I, Λ̂·C), with r̂ = (1−Λ̂)(1−1/d)/Λ̂.

Mitigation **is** exactly "remove the ridge r̂ and rescale C". Theorem 1 is the special case of global depolarizing noise
with infinite shots.

**Test (Table R1/R1b, Fig. R1).** 1,998 cells across all three runs:
- All **1,537** cells whose raw kernel has smallest eigenvalue above −0.49 agree to solver precision (max relative
  decision difference **1.8e-7**, identical predictions). This covers every noise model, including the device-like
  model with coherent error.
- 51 cells (**2.6%**) differ. All of them have strongly indefinite kernels (16 shots, or 64 shots at N=200). There the
  SVM dual is non-convex, and libsvm's non-PSD safeguard is not scale-invariant, so the two equivalent problems stop at
  different stationary points. State this in the paper as a numerical caveat; it is not a failure of the identity.
- **Why v2 weakened at 64 shots.** v2 compared the finite-shot raw SVM with the *noise-free* K + rI. The entire v2
  discrepancy equals the measured-noise term E/Λ. Mean relative difference at 64 shots: 0.346 for Theorem 1, and
  exactly 0.346 for E/Λ alone.
- **Corollary 2.** Mitigation leaves the kernel's signal-to-noise ratio unchanged (median SNR 4.63 raw vs 4.63
  mitigated). Drop the v2 statement "mitigation amplifies shot noise" as a classification argument.

## W2: "mitigation can hurt was weakly supported" → replaced by a sharp, confirmed prediction

From Corollary 1 and Proposition 3 (a ridge is an extra squared-hinge soft-margin term, so it overlaps with C):
- (P2a) at **fixed C**, mitigation changes accuracy, more strongly at higher noise.
- (P2b) with **C tuned**, the change vanishes.

R1 results (Table R2, R2b, Fig. R2; units = dataset × seed, n = 40; differences are mitigated − raw):

| C setting | mitigated − raw | mitigated+clip − raw+clip | Spearman(\|effect\|, r̂) |
|---|---|---|---|
| fixed C = 0.1 | **+0.044** [+0.023, +0.069], Holm p = 0.003 | +0.044, p = 0.013 | **+0.98** (p = 2e-32) |
| fixed C = 1 | −0.002 [−0.005, +0.001] | −0.003 | +0.80 (p = 7e-11) |
| fixed C = 10 | **−0.012** [−0.016, −0.007], p = 4e-5 | **−0.034**, p = 9e-7 | +0.81 (p = 1e-11) |
| fixed C = 100 | **−0.027** [−0.034, −0.019], p = 8e-7 | **−0.054**, p = 5e-10 | +0.59 (p = 2e-5) |
| **C tuned** | −0.002 [−0.006, +0.001], **equivalent ±0.01 (TOST p = 6.5e-5)** | −0.001, **equivalent (TOST p = 1.4e-5)** | +0.58 (mean ≈ 0) |

**Revised claim:**
- At fixed C, mitigation is *not* neutral. It helps when C is small (an over-regularised SVM loses its extra ridge) and
  hurts when C is large. Either way, the size of the effect grows with noise.
- With C tuned on validation, mitigation is equivalent within ±0.01.
- "Mitigation can hurt" is now supported, but only for fixed (large) C. That is the practically relevant case, because
  many pipelines use library defaults.

Scale-up replication (fixed C = 100): −0.020 [−0.029, −0.010] at N=200; −0.015 [−0.029, −0.004] at 6 qubits.
With tuned C: −0.004 [−0.011, +0.003] and −0.007 [−0.020, +0.004]. These are close to zero, but with 12 and 9 units the
TOST equivalence test is not conclusive (p = 0.07 and 0.34).

## W3: "hidden-ridge accuracy test used fixed C = 1" → re-run with tuned C

Table R3, R3b, R3c; Fig. R3; exact kernels, n = 40:
- **The hidden ridge on the ideal kernel does not help once C is tuned.** Ideal + rI minus ideal: −0.014
  [−0.028, −0.002], Holm p = 0.29. The ridge sweep with tuned C falls steadily, from 0.867 (no ridge) to 0.823 (4r).
  The v2 gap (0.832 vs 0.798) came from the fixed C.
- **Where the noisy QSVM actually loses accuracy is the measured noise E, not the ridge.** Raw noisy minus ideal + rI,
  with tuned C: −0.037 [−0.045, −0.030], Holm p = 2e-11. Global mitigation cannot remove E (Corollary 2).

This gives the paper one consistent story. Diagonal mitigation only moves a ridge that C can absorb. The real loss comes
from shot noise and non-global noise, which the affine correction cannot touch.

## W4: "small problems / simulation only / RBF beats QSVM" → addressed (scale-up + a small hardware check)

- **Larger N and more qubits (Table R4, R4b, R4c; Fig. R4).** Corollary 1 holds at N=200 and at 6 qubits. The pattern
  "fixed C hurts, tuned C is about neutral" replicates. NA-QSVM is still not better than QSVM-clip (−0.005 and +0.005,
  not significant).
- **Device-like noise.** Heterogeneous per-qubit depolarizing, T1/T2 relaxation, asymmetric readout, and a *coherent*
  ZZ over-rotation (which the depolarizing correction cannot remove). Two profiles, "device A" (low-error class) and
  "device B" (higher-error class); the parameters are illustrative, not a specific backend. All conclusions hold.
- **Quantum-structured data** (Havlíček et al. 2019 "ad hoc" construction, labels defined through the same ZZ feature
  map):
  - Ideal QSVM 0.893 vs RBF 0.523 (4 qubits, N=200).
  - Ideal QSVM 0.722 vs RBF 0.533 (6 qubits).
  - The noisy QSVMs reach 0.79 and 0.69, so the study now includes a regime where the quantum kernel is the right model.
    The mitigation conclusions are the same there.
- **Classical baseline still wins on every classical dataset**, at N=200 and at 6 qubits. Keep reporting this and keep
  "no quantum advantage" in the abstract. The paper's claims do not depend on advantage.
- **The scale-up uses 3 seeds per dataset** (12 and 9 units), fewer than the 10 seeds of the main study.

## W4b: real-hardware check on IBM `ibm_fez` (18 Oct 2026)

Setup: Heron r2, 156 qubits; physical qubits 0–3; about 21 two-qubit gates per kernel circuit; no twirling, dynamical
decoupling or readout mitigation. Breast cancer data, 3 splits (seeds 5000–5002), N_train=30, N_val=10, N_test=30.
That is 5,115 circuits × 256 shots, with a 64-shot exact sub-sample of the same counts. The script estimated about 7 of the 10 free Open Plan
minutes; the actual usage is shown on the IBM dashboard. Results: `hardware/hw_qpu/HARDWARE_RESULTS.md`; LaTeX: `hardware/hw_qpu/tableR5_hardware.tex`.

- **Corollary 1 holds on real device data.** Relative decision difference 8.6e-9 to 5.2e-8 in all 6 (split × shots)
  cells, with 100% identical predictions. This is the same solver-precision level as in simulation.
- **The device's global survival is high and stable.** Λ̂ = 0.923–0.930 across splits and shot budgets (jobs submitted
  about 13 minutes apart), so the hidden ridge is small: r̂ = 0.070–0.078. That is below the smallest r̂ in the
  simulations (≈0.18, device A).
- **Mitigation effect.** With C tuned by 5-fold CV + validation, mitigated equals raw in 5 of 6 cells; the 6th
  (split 5000, 256 shots) is −0.13, a different C choice on one small split. At fixed C the differences are −0.03 to
  +0.05. This agrees with the dose-response in Fig. R2 (the effect shrinks as r̂ → 0), but with 3 splits and 30 test
  points it is a consistency check, not a statistical test.
- **Accuracy.** Mean over splits, tuned C: raw 0.73 at 64 shots / 0.66 at 256 shots; ideal noise-free kernel 0.78;
  RBF-SVM 0.95. The hardware QSVM is limited by the small N=30 and by E (shot noise and non-global noise), as the
  simulations predict. The 64 vs 256-shot difference is within split-to-split noise.
- **Protocol deviation (state it in the paper).** With only 10 validation points, the simulation protocol for tuning C
  (validation only, ties → smallest C) often picked a one-class predictor. For the hardware split, C was therefore tuned
  by 5-fold CV on the training kernel + validation, with ties → median C. The same rule is used for raw, mitigated and
  ideal.

---

## Changes to make in the manuscript

1. **Theory section.** Add Corollary 1, Corollary 2 and Proposition 3 (text ready in `REVISION_THEORY.md`). Remove
   consequence (c) "mitigation amplifies shot noise".
2. **Replace the §8b hidden-ridge figure/table** with Fig. R1 and Table R1, and add the indefinite-kernel caveat.
3. **Mitigation section.** Replace "does not help / can hurt" with the fixed-vs-tuned result (Table R2, Fig. R2).
   Suggested sentence: *"Diagonal-based mitigation changes a QSVM only through a ridge; at fixed C it shifts balanced
   accuracy by several points (up to about 7 at the highest noise), in a direction set by C, while with C tuned on validation it is equivalent
   within ±0.01."*
4. **Add Fig. R3** (ridge sweep) and the finding that the accuracy loss comes from E, not the ridge.
5. **Add a scale-up subsection** (Fig. R4, Table R4c) with the quantum-structured dataset and the device-like noise.
6. **Add a short hardware subsection** (Table R5, `hardware/hw_qpu/tableR5_hardware.tex`): Corollary 1 confirmed on
   ibm_fez, Λ̂ ≈ 0.93, r̂ ≈ 0.07, mitigation ≈ neutral, as predicted for small r̂.
7. **Limitations.** The hardware check is small (1 device, 4 qubits, 3 splits, N=30); the scale-up has fewer seeds; the
   device profiles are illustrative.
8. **Abstract and title** still fit. A more precise title option: *"Error Mitigation Only Moves a Ridge: An Exact
   Analysis of Quantum-Kernel SVMs under Noise and Finite Shots"*.

## Files added (all inside `noise_aware_qsvm_research/`)

| Path | What |
|---|---|
| `src/naqsvm/revision.py` | Corollary-1 test, fixed/tuned-C mechanism block, device-like noise simulator, quantum-structured dataset |
| `run_revision_mechanism.py` | R1 (about 65 min on 2 CPU cores) |
| `run_revision_scaleup.py A\|B --nseeds=3` | R2-A / R2-B (about 25 / 40 min) |
| `analyze_revision.py` | all revision tables, figures and `REVISION_FINDINGS_raw.md` |
| `results/revision/*.csv` | raw results of R1 and R2 |
| `tables/revision/` | Tables R1–R4 (CSV; LaTeX for R1, R2, R4c) |
| `figures/revision/` | Figs. R1–R4 (PNG and PDF) |
| `REVISION_THEORY.md`, `REVISION_NOTES.md`, `REVISION_FINDINGS_raw.md` | theory text, this summary, raw numbers |
| `hardware/run_ibm_hardware.py`, `hardware/analyze_hardware.py`, `hardware/README_HARDWARE.md` | IBM hardware run and analysis |
| `hardware/hw_qpu/seed500*/` | raw hardware counts and job metadata (job IDs, backend, layout) |
| `hardware/hw_qpu/HARDWARE_RESULTS.md`, `*.csv`, `tableR5_hardware.tex` | hardware results and LaTeX table |
