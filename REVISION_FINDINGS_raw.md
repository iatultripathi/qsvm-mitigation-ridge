# Revision findings (auto-generated numbers)

## W1 - C1 at low shot counts: resolved by Corollary 1 (exact finite-shot identity)

Test of Corollary 1, SVM(K_raw, C) vs SVM(K_mit + r_hat I, Lam_hat C), over 1998 cells from all three runs (R1: 4 datasets x 10 seeds x 9 noise settings x 5 shot budgets incl. exact; R2-A: N=200; R2-B: 6 qubits). All 1537 cells whose raw kernel has smallest eigenvalue above -0.49 agree to solver precision (max relative decision difference 1.8e-07). Of the 461 strongly indefinite cells (smallest eigenvalue <= -0.49; few shots and/or large N), 410 also agree; 51 (2.6% of all cells) do not (max rel. diff 0.41, min prediction agreement 0.833). For a strongly indefinite kernel the SVM dual is non-convex and its solution is not unique; libsvm's safeguard for non-positive curvature is not scale-invariant, so two mathematically equivalent problems can stop at different stationary points. These cells occur at ['16', '64'] shots per entry. The identity holds under every noise model (global, local depolarizing, thermal, device-like with coherent error).
- shots=exact: ideal+ridge (Theorem 1) rel. diff 0.032; the part explained by the measured-kernel deviation E/Lam alone (same ridge, same C; shot noise + departure from global depolarizing) is 0.032; no-ridge control 0.169.
- shots=1024: ideal+ridge (Theorem 1) rel. diff 0.125; the part explained by the measured-kernel deviation E/Lam alone (same ridge, same C; shot noise + departure from global depolarizing) is 0.125; no-ridge control 0.206.
- shots=64: ideal+ridge (Theorem 1) rel. diff 0.346; the part explained by the measured-kernel deviation E/Lam alone (same ridge, same C; shot noise + departure from global depolarizing) is 0.346; no-ridge control 0.369.
- shots=16: ideal+ridge (Theorem 1) rel. diff 0.518; the part explained by the measured-kernel deviation E/Lam alone (same ridge, same C; shot noise + departure from global depolarizing) is 0.518; no-ridge control 0.523.
Interpretation: the v2 '64-shot failure' was the shot-noise term E/Lam, not a failure of the ridge account. Global mitigation leaves the signal-to-noise ratio unchanged (mean SNR raw 4.63 vs mitigated 4.63, median over cells).

## W2 - does mitigation hurt? Sharpened prediction and test

Prediction from Corollary 1: mitigation = removing the ridge r_hat (C rescaled). A ridge acts as an extra soft-margin term that overlaps with C, so (P2a) at FIXED C mitigation changes accuracy and the size of the change grows with r_hat; (P2b) with C TUNED on validation the change vanishes.
- mitigated - raw [C0.1]: +0.0444 [+0.0228, +0.0686], mean |cell effect| 0.0450, Holm p = 0.003, TOST(±0.01) p = 1 -> not shown equivalent
- mitigated - raw [C1]: -0.0020 [-0.0052, +0.0011], mean |cell effect| 0.0234, Holm p = 0.25, TOST(±0.01) p = 7.7e-06 -> equivalent
- mitigated - raw [C10]: -0.0117 [-0.0162, -0.0072], mean |cell effect| 0.0432, Holm p = 3.5e-05, TOST(±0.01) p = 0.77 -> not shown equivalent
- mitigated - raw [C100]: -0.0265 [-0.0338, -0.0188], mean |cell effect| 0.0604, Holm p = 8.2e-07, TOST(±0.01) p = 1 -> not shown equivalent
- mitigated - raw [tuned]: -0.0024 [-0.0058, +0.0011], mean |cell effect| 0.0298, Holm p = 0.25, TOST(±0.01) p = 6.5e-05 -> equivalent
- mitigated+clip - raw+clip [C0.1]: +0.0441 [+0.0225, +0.0683], mean |cell effect| 0.0468, Holm p = 0.013, TOST(±0.01) p = 1 -> not shown equivalent
- mitigated+clip - raw+clip [C1]: -0.0029 [-0.0065, +0.0007], mean |cell effect| 0.0236, Holm p = 0.14, TOST(±0.01) p = 2.2e-04 -> equivalent
- mitigated+clip - raw+clip [C10]: -0.0340 [-0.0411, -0.0266], mean |cell effect| 0.0518, Holm p = 8.8e-07, TOST(±0.01) p = 1 -> not shown equivalent
- mitigated+clip - raw+clip [C100]: -0.0544 [-0.0649, -0.0441], mean |cell effect| 0.0704, Holm p = 5.0e-10, TOST(±0.01) p = 1 -> not shown equivalent
- mitigated+clip - raw+clip [tuned]: -0.0008 [-0.0044, +0.0030], mean |cell effect| 0.0290, Holm p = 0.35, TOST(±0.01) p = 1.4e-05 -> equivalent
- dose-response, C=C0.1: mean |effect| per cell 0.0450; Spearman(|effect|, r_hat) = +0.98 (p = 1.9e-32)
- dose-response, C=C1: mean |effect| per cell 0.0234; Spearman(|effect|, r_hat) = +0.80 (p = 6.5e-11)
- dose-response, C=C10: mean |effect| per cell 0.0432; Spearman(|effect|, r_hat) = +0.81 (p = 1.3e-11)
- dose-response, C=C100: mean |effect| per cell 0.0604; Spearman(|effect|, r_hat) = +0.59 (p = 2.3e-05)
- dose-response, C=tuned: mean |effect| per cell 0.0298; Spearman(|effect|, r_hat) = +0.58 (p = 3.2e-05)

## W3 - hidden-ridge accuracy test with tuned C

- ideal + r I  minus  ideal (no ridge) [C0.1]: +0.0045 [+0.0018, +0.0076], Holm p = 0.026, TOST p = 3.3e-04
- raw noisy  minus  ideal + r I [C0.1]: -0.0535 [-0.0801, -0.0298], Holm p = 8.5e-05, TOST p = 1
- ideal + r I  minus  ideal (no ridge) [C1]: +0.0049 [-0.0016, +0.0110], Holm p = 0.29, TOST p = 0.064
- raw noisy  minus  ideal + r I [C1]: -0.0393 [-0.0458, -0.0331], Holm p = 1.8e-11, TOST p = 1
- ideal + r I  minus  ideal (no ridge) [C10]: -0.0125 [-0.0225, -0.0021], Holm p = 0.093, TOST p = 0.68
- raw noisy  minus  ideal + r I [C10]: -0.0709 [-0.0798, -0.0618], Holm p = 1.8e-11, TOST p = 1
- ideal + r I  minus  ideal (no ridge) [C100]: -0.0093 [-0.0281, +0.0087], Holm p = 0.35, TOST p = 0.47
- raw noisy  minus  ideal + r I [C100]: -0.0789 [-0.0882, -0.0696], Holm p = 1.8e-11, TOST p = 1
- ideal + r I  minus  ideal (no ridge) [tuned]: -0.0143 [-0.0283, -0.0021], Holm p = 0.29, TOST p = 0.73
- raw noisy  minus  ideal + r I [tuned]: -0.0374 [-0.0449, -0.0303], Holm p = 1.8e-11, TOST p = 1
Ridge sweep on the ideal kernel (mean balanced accuracy, exact kernels): C1: 0r→0.847, 0.25r→0.854, 0.5r→0.855, 1r→0.852, 2r→0.847, 4r→0.828; C100: 0r→0.875, 0.25r→0.882, 0.5r→0.877, 1r→0.865, 2r→0.853, 4r→0.830; tuned: 0r→0.867, 0.25r→0.866, 0.5r→0.861, 1r→0.853, 2r→0.844, 4r→0.823

## W4 - scale-up (N = 200 at 4 qubits; 6 qubits; device-like noise; quantum-structured data)

References (ideal quantum kernel vs classical RBF, mean balanced accuracy):
- 4q, N=200, adhoc: ideal QSVM 0.893; RBF same input 0.523; RBF all features nan
- 4q, N=200, breast_cancer: ideal QSVM 0.916; RBF same input 0.949; RBF all features 0.947
- 4q, N=200, digits_3v8: ideal QSVM 0.935; RBF same input 0.962; RBF all features 0.993
- 4q, N=200, synthetic_large: ideal QSVM 0.752; RBF same input 0.789; RBF all features 0.866
- 6q, N=80, adhoc: ideal QSVM 0.722; RBF same input 0.533; RBF all features nan
- 6q, N=80, breast_cancer: ideal QSVM 0.894; RBF same input 0.929; RBF all features 0.925
- 6q, N=80, synthetic: ideal QSVM 0.778; RBF same input 0.828; RBF all features 0.828
Mitigation effect at scale (QSVM-mitig(avg)+clip minus QSVM-clip, tuned C, fair SV test protocol):
- 4q, N=200: QSVM-mitig(avg)+clip - QSVM-clip: -0.0069 [-0.0122, -0.0018], n=12, Holm p = 0.31, TOST p = 0.14
- 4q, N=200: NA-QSVM (proposed) - QSVM-clip: -0.0049 [-0.0149, +0.0032], n=12, Holm p = 1, TOST p = 0.16
- 4q, N=200: QSVM-mitig(avg)+clip - QSVM-raw: -0.0008 [-0.0047, +0.0029], n=12, Holm p = 1, TOST p = 3.6e-04
- 6q, N=80: QSVM-mitig(avg)+clip - QSVM-clip: +0.0020 [-0.0051, +0.0108], n=9, Holm p = 1, TOST p = 0.052
- 6q, N=80: NA-QSVM (proposed) - QSVM-clip: +0.0054 [-0.0043, +0.0148], n=9, Holm p = 1, TOST p = 0.2
- 6q, N=80: QSVM-mitig(avg)+clip - QSVM-raw: +0.0012 [-0.0083, +0.0125], n=9, Holm p = 1, TOST p = 0.08
- stage A: Corollary 1 max rel. decision difference 1.2e-01 over 144 cells; Lam range 0.23-0.84.
  stage A, mitigated - raw [C100]: -0.0195 [-0.0290, -0.0102], mean |cell| 0.0417, TOST p = 0.96
  stage A, mitigated - raw [tuned]: -0.0037 [-0.0111, +0.0034], mean |cell| 0.0231, TOST p = 0.068
- stage B: Corollary 1 max rel. decision difference 2.0e-07 over 54 cells; Lam range 0.53-0.82.
  stage B, mitigated - raw [C100]: -0.0153 [-0.0292, -0.0036], mean |cell| 0.0419, TOST p = 0.76
  stage B, mitigated - raw [tuned]: -0.0072 [-0.0195, +0.0037], mean |cell| 0.0249, TOST p = 0.34
