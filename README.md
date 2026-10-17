# Error Mitigation Only Moves a Ridge: code and data

Code, configurations, raw results, figures, tables and IBM hardware counts for the article
*"Error Mitigation Only Moves a Ridge: An Exact Analysis of Quantum-Kernel Support Vector Machines Under Noise and
Finite Shots"* by A. Tripathi, A. Rasool, N. Khare and M. K. Jhariya (IEEE Access, under review).
License: MIT (see `LICENSE`). How to cite: see `CITATION.cff`.

Quick start: `pip install -r requirements.txt`, then open `noise_aware_qsvm_research.ipynb` (benchmark), or run
the revision scripts listed below. Hardware scripts and the IBM `ibm_fez` counts with job IDs are in `hardware/`.

The sections below describe the package history (v2 → v3).

## v3 revision (17–18 Oct 2026)
New analyses that address four weaknesses of v2: C1 at low shot counts, weak support for "mitigation can hurt", the
fixed-C hidden-ridge test, and small problem size. Start with **`REVISION_NOTES.md`** (summary and manuscript changes),
then `REVISION_THEORY.md` (Corollaries 1–2, Proposition 3). Code: `src/naqsvm/revision.py` and
`run_revision_*.py` / `analyze_revision.py`. Outputs: `results/revision/`, `tables/revision/`, `figures/revision/`.
Reproduce: `python run_revision_mechanism.py; python run_revision_scaleup.py A --nseeds=3;
python run_revision_scaleup.py B --nseeds=3; python analyze_revision.py`.

## What is in this folder
| Path | Content |
|---|---|
| `src/naqsvm/` | complete source code (importable package) |
| `configs/experiment_config.json` | every setting of this run (datasets, seeds, noise, shots, grids, bandwidths) |
| `configs/method_variants.json` | options of the proposed method and its ablations |
| `results/raw_results.csv` | one row per dataset × seed × noise × shots × method (all metrics) |
| `results/hidden_ridge_check.csv` | per split × noise × shots results of the Theorem-1 test (§8b) |
| `results/simulation_log.csv` | per-block simulator backend, time, exact Λ, Neyman oracle efficiency |
| `results/split_indices.csv` | exact train/val/test row indices (reproducibility, leakage audit) |
| `results/self_checks.csv` | outcome of all correctness checks |
| `results/summary_*.csv`, `results/bandwidth_calibration.csv` | aggregated results |
| `tables/` | Tables 1–6 and statistical tests incl. mitigation TOST (CSV; LaTeX for Tables 1–2) |
| `figures/` | all figures (PNG and PDF) |
| `FINDINGS.md` | auto-generated findings (numbers only) |
| `LITERATURE_CHECKLIST.md` | novelty audit: closest prior work, search protocol, decision rules |
| `environment.json`, `requirements.txt` | software and hardware environment of this run |

## Contributions and their status
* C1 Hidden-ridge theorem (exact under global depolarizing noise): fixing diag(K)=1 ≡ ideal kernel + ridge (1−Λ)(1−1/d)/Λ with box ΛC;
  mitigation removes this ridge. Sharpens Heyraud et al. (PRA 2022).
* C2 Direct test of C1 under global, local, non-unital noise and finite shots (`results/hidden_ridge_check.csv`).
* C3 Equal-shot-budget benchmark, fair SV-restricted test protocol for every method, mitigation equivalence tests (TOST).
* C4 NA-QSVM: explicit ridge calibrated to shot noise after mitigation (close to ShofaR in spirit; not a headline claim).
* Established: fidelity-kernel QSVM, binomial shot model, PSD clip/flip/shift, depolarizing mitigation (Hubregtsen et al. 2022).
* Prior work re-implemented as baselines: AQKA-Bernoulli and AQKA-SVM (Xu et al. 2026); margin-aware Neyman approximating Miroszewski (2026).

## How to reproduce
1. Open `noise_aware_qsvm_research.ipynb` in Google Colab or Jupyter and run all cells
   (`RUN_MODE="QUICK"` for a smoke test). Python ≥ 3.10.
2. Or, from Python: `sys.path.insert(0, "src")`, then use `naqsvm.experiment.run_grid`.
Seeds: evaluation [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009]; bandwidth pilot [100, 101, 102]. All random streams are derived from these seeds.

## Findings of this run
*Run: FULL mode, simulator `aer`, 4 datasets × 10 seeds 2000–2009 × 5 noise settings × 4 shot budgets; 13720 result rows; 0 logged failures.*

**C1/C2 – Hidden-ridge test** (relative decision difference to the raw noisy SVM; B = ideal kernel + hidden ridge, Z = no-ridge control):
- global Λ=0.8, shots=exact: Λ=0.800, r=0.234; B 0.0000 vs Z 0.0848; prediction agreement B 1.000 vs Z 0.991; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.8, shots=1024: Λ=0.800, r=0.234; B 0.0934 vs Z 0.1269; prediction agreement B 0.975 vs Z 0.973; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.8, shots=64: Λ=0.800, r=0.234; B 0.3263 vs Z 0.3230; prediction agreement B 0.919 vs Z 0.915; B closer in 35% of splits (Holm p = 0.11)
- global Λ=0.5, shots=exact: Λ=0.500, r=0.938; B 0.0000 vs Z 0.2317; prediction agreement B 1.000 vs Z 0.975; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.5, shots=1024: Λ=0.500, r=0.938; B 0.1036 vs Z 0.2540; prediction agreement B 0.969 vs Z 0.960; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.5, shots=64: Λ=0.500, r=0.938; B 0.3661 vs Z 0.3907; prediction agreement B 0.900 vs Z 0.895; B closer in 70% of splits (Holm p = 0.000378)
- global Λ=0.25, shots=exact: Λ=0.250, r=2.812; B 0.0000 vs Z 0.4195; prediction agreement B 1.000 vs Z 0.941; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.25, shots=1024: Λ=0.250, r=2.812; B 0.1312 vs Z 0.4514; prediction agreement B 0.967 vs Z 0.933; B closer in 100% of splits (Holm p = 3.82e-11)
- global Λ=0.25, shots=64: Λ=0.250, r=2.812; B 0.4278 vs Z 0.5749; prediction agreement B 0.859 vs Z 0.857; B closer in 98% of splits (Holm p = 3.82e-11)
- low, shots=exact: Λ=0.770, r=0.280; B 0.0079 vs Z 0.0979; prediction agreement B 0.998 vs Z 0.990; B closer in 100% of splits (Holm p = 3.82e-11)
- low, shots=1024: Λ=0.770, r=0.280; B 0.0940 vs Z 0.1338; prediction agreement B 0.978 vs Z 0.975; B closer in 100% of splits (Holm p = 3.82e-11)
- low, shots=64: Λ=0.770, r=0.280; B 0.3510 vs Z 0.3579; prediction agreement B 0.905 vs Z 0.902; B closer in 60% of splits (Holm p = 0.115)
- medium, shots=exact: Λ=0.483, r=1.006; B 0.0240 vs Z 0.2406; prediction agreement B 0.993 vs Z 0.975; B closer in 100% of splits (Holm p = 3.82e-11)
- medium, shots=1024: Λ=0.483, r=1.006; B 0.1154 vs Z 0.2685; prediction agreement B 0.968 vs Z 0.962; B closer in 100% of splits (Holm p = 3.82e-11)
- medium, shots=64: Λ=0.483, r=1.006; B 0.3771 vs Z 0.4186; prediction agreement B 0.893 vs Z 0.895; B closer in 90% of splits (Holm p = 1.88e-09)
- high, shots=exact: Λ=0.234, r=3.064; B 0.0586 vs Z 0.4448; prediction agreement B 0.988 vs Z 0.936; B closer in 100% of splits (Holm p = 3.82e-11)
- high, shots=1024: Λ=0.234, r=3.064; B 0.1501 vs Z 0.4743; prediction agreement B 0.964 vs Z 0.922; B closer in 100% of splits (Holm p = 3.82e-11)
- high, shots=64: Λ=0.234, r=3.064; B 0.4422 vs Z 0.5794; prediction agreement B 0.867 vs Z 0.853; B closer in 98% of splits (Holm p = 3.82e-11)
- thermal (mismatch), shots=exact: Λ=0.590, r=0.655; B 0.0489 vs Z 0.1912; prediction agreement B 0.985 vs Z 0.982; B closer in 100% of splits (Holm p = 3.82e-11)
- thermal (mismatch), shots=1024: Λ=0.590, r=0.655; B 0.1122 vs Z 0.2165; prediction agreement B 0.968 vs Z 0.963; B closer in 98% of splits (Holm p = 3.82e-11)
- thermal (mismatch), shots=64: Λ=0.590, r=0.655; B 0.3556 vs Z 0.3753; prediction agreement B 0.910 vs Z 0.908; B closer in 72% of splits (Holm p = 0.00181)

**C3 – Effect of diagonal-based mitigation** (paired, units = dataset × seed):
- accuracy, QSVM-mitig(avg)+clip minus QSVM-clip: -0.0019 [-0.0060, +0.0023] → no significant difference (Holm p = 0.512); equivalent within ±0.010 (TOST p = 0.000309)
- kernel error, QSVM-mitig(avg)+clip minus QSVM-clip: -0.2229 [-0.2288, -0.2163] (Holm p = 5.46e-12)
- accuracy, QSVM-mitig(split)+clip minus QSVM-clip: -0.0037 [-0.0075, -0.0000] → no significant difference (Holm p = 0.275); equivalent within ±0.010 (TOST p = 0.00124)
- kernel error, QSVM-mitig(split)+clip minus QSVM-clip: -0.1984 [-0.2042, -0.1919] (Holm p = 5.46e-12)
- accuracy, NA-QSVM (proposed) minus Abl: no mitigation: +0.0018 [-0.0011, +0.0049] → no significant difference (Holm p = 0.512); equivalent within ±0.010 (TOST p = 1.95e-06)
- kernel error, NA-QSVM (proposed) minus Abl: no mitigation: -0.1998 [-0.2064, -0.1931] (Holm p = 5.46e-12)

**C4 – NA-QSVM vs established and prior-adaptive methods** (balanced accuracy, NA-QSVM minus other; fair SV test protocol):
- vs QSVM-raw: +0.0002 [-0.0044, +0.0048], W/T/L = 20/1/19 → no significant difference (Holm p = 1)
- vs QSVM-clip: +0.0014 [-0.0032, +0.0059], W/T/L = 20/0/20 → no significant difference (Holm p = 1)
- vs QSVM-flip: +0.0008 [-0.0034, +0.0050], W/T/L = 20/1/19 → no significant difference (Holm p = 1)
- vs QSVM-shift: +0.0024 [-0.0025, +0.0073], W/T/L = 20/1/19 → no significant difference (Holm p = 1)
- vs QSVM-mitig(avg)+clip: +0.0033 [-0.0018, +0.0084], W/T/L = 25/1/14 → no significant difference (Holm p = 1)
- vs QSVM-mitig(split)+clip: +0.0051 [+0.0010, +0.0093], W/T/L = 26/1/13 → no significant difference (Holm p = 0.184)
- vs AQKA-Bernoulli+clip: +0.0018 [-0.0031, +0.0065], W/T/L = 23/0/17 → no significant difference (Holm p = 1)
- vs AQKA-SVM+clip: +0.0025 [-0.0016, +0.0063], W/T/L = 24/0/16 → no significant difference (Holm p = 0.866)
- vs MarginNeyman+clip (Miroszewski-style approx.): -0.0012 [-0.0058, +0.0036], W/T/L = 17/0/23 → no significant difference (Holm p = 1)

**Ablations (full NA-QSVM minus variant):**
- Abl: uniform shots: +0.0005 [-0.0033, +0.0044] → no significant difference (Holm p = 1)
- Abl: no noise-aware ridge: +0.0030 [-0.0008, +0.0066] → no significant difference (Holm p = 0.183)
- Abl: no mitigation: +0.0018 [-0.0011, +0.0049] → no significant difference (Holm p = 1)
- Abl: no SV-restricted test: +0.0078 [+0.0044, +0.0111] → significantly **higher** (Holm p = 0.001)
- Abl: ridge c=1 (untuned): -0.0025 [-0.0058, +0.0006] → no significant difference (Holm p = 0.295)
- Abl: SV-weighted train alloc: +0.0016 [-0.0021, +0.0054] → no significant difference (Holm p = 1)
- Variant: shot-saving (SV test, S/entry): +0.0084 [+0.0051, +0.0119] → significantly **higher** (Holm p = 0.000324)

**Kernel-matrix error (rel. off-diagonal Frobenius, mean over units):** QSVM-raw: 0.375, QSVM-clip: 0.367, QSVM-flip: 0.367, QSVM-shift: 0.375, QSVM-mitig(avg)+clip: 0.144, QSVM-mitig(split)+clip: 0.169, AQKA-Bernoulli+clip: 0.367, AQKA-SVM+clip: 0.377, MarginNeyman+clip (Miroszewski-style approx.): 0.374, NA-QSVM (proposed): 0.171

**References:** ideal QSVM (exact, noise-free) balanced accuracy 0.873; RBF-SVM on the same 4-D input 0.902; best finite-shot noisy quantum method on average: MarginNeyman+clip (Miroszewski-style approx.) (0.817). The classical RBF-SVM matches or exceeds the ideal quantum kernel on these data, so **no quantum advantage is indicated**.

**Friedman test** over 80 blocks: p = 0.45; mean ranks: MarginNeyman+clip (Miroszewski-style approx.) 4.96, NA-QSVM (proposed) 5.08, QSVM-raw 5.13, QSVM-clip 5.39, QSVM-flip 5.59, QSVM-shift 5.69, QSVM-mitig(split)+clip 5.75, QSVM-mitig(avg)+clip 5.78, AQKA-Bernoulli+clip 5.78, AQKA-SVM+clip 5.84

## Limitations
Simulation only (parametrised noise, 4 qubits, N = 60 training points); no hardware runs; no claim of quantum advantage;
novelty must be confirmed with `LITERATURE_CHECKLIST.md`. See Section 14 of the notebook for the full list.
