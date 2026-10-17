# Literature-review checklist and novelty audit (v2, 15 Oct 2026)

**Status of this audit.** Items marked [read in full] were read completely. Items marked [checked online] were checked
through their arXiv abstract or HTML pages on 15 Oct 2026; some HTML pages were incomplete, as noted. Other bibliographic
details were written from memory. **Verify every entry against the publisher or arXiv record before citing it.**
No experimental result in this package depends on these references.

## A. Closest prior work and how this paper relates to it

| # | Reference | What it does | Overlap with this paper | How to position |
|---|---|---|---|---|
| A1 | J. Xu, C. Li, D. Zeng, J. Paisley, Q. Zhao, "AQKA: Active Quantum Kernel Acquisition Under a Shot Budget", arXiv:2605.14672 (2026) [checked online] | Adaptive shot allocation over kernel entries. Closed-form rules for KRR and SVM: Prop. 4, $s_{ij}\propto\|\eta_i\eta_j\|\sqrt{K_{ij}(1-K_{ij})}$. Its "bernoulli-only" baseline is $s\propto\sqrt{K(1-K)}$. IBM hardware runs. No error mitigation. | **Very high** for allocation. NA-QSVM's pilot + Neyman step *is* AQKA's bernoulli-only baseline. | Do not claim allocation. Use AQKA-Bernoulli and AQKA-SVM as baselines (done, re-implemented). Ask the authors for code. |
| A2 | A. Miroszewski, "Adaptive Measurement Allocation for Learning Kernelized SVMs Under Noisy Observations", arXiv:2605.22275 (2026), with the ShotWise package [checked online; algorithm sections not retrievable] | Neyman-type allocation for kernel SVMs combining margin sensitivity $\alpha_i\alpha_j$, margin-violation probability $P_i$, and variance including a physical-noise term. Early stopping on dual-coefficient stability. IBM hardware runs. | **Very high** for SVM-aware allocation. | Baseline (an approximation from the published equations is included). Replace it with official ShotWise before submission. |
| A3 | A. Shastry, A. Jayakumar, A. Patel, C. Bhattacharyya, "Shot-frugal and Robust quantum kernel classifiers", arXiv:2210.06971 [checked online] | Chance-constrained SOCP (ShofaR). The shot-noise-derived diagonal shift $(c/2\sqrt N)\kappa(\delta)I$ in the regulariser avoids PSD clipping. Notes that prediction needs only kernels to the support vectors. | **High** for the noise-calibrated ridge (C4) and the SV-restricted test. | C4 differs only in calibrating the ridge *after* depolarizing mitigation (variance ×1/Λ²). Present C4 as a practical variant, not a headline. Consider ShofaR as an extra baseline. |
| A4 | V. Heyraud, Z. Li, Z. Denis, A. Le Boité, C. Ciuti, "Noisy quantum kernel machines", Phys. Rev. A 106, 052421 (2022) [read in full] | Dissipative spin-chain kernels Tr[ρρ′] with least-squares training. Decoherence lowers the kernel trace (average purity), which tightens a Rademacher bound, giving "implicit regularization". Gaussian measurement noise ≡ λ→λ+σ². An intercept ≡ centred kernel. | **Medium–high** in spirit for C1. | They give no exact equivalence, do not fix the diagonal to 1, and use no fidelity kernel, hinge loss or mitigation. C1 is an exact sharpening for the standard QSVM pipeline. Cite prominently. |
| A5 | T. Hubregtsen, D. Wierichs, E. Gil-Fuster, P.-J. H. S. Derks, P. K. Faehrmann, J. J. Meyer, "Training quantum embedding kernels on near-term quantum computers", Phys. Rev. A 106, 042431 (2022) | Diagonal-based depolarizing mitigation (single / average / split channel) and PSD regularisation. | Established component and baselines. | Cite as the source of the mitigation that C1 analyses. |
| A6 | S. Govender, I. Sinayskiy, "The Cross-Kernel Margin: A Robustness Measure for Quantum Kernel Methods", arXiv:2601.23084 (2026) [checked online] | Local vs global depolarizing models for QSVMs; Tikhonov-stabilised SVM dual; margin-based robustness bounds. | Medium (Tikhonov term; local vs global noise). | Read Section on Tikhonov stabilisation; check that it does not state C1. |
| A7 | G. Lo Monaco et al., "Who can sample forever? Shot noise effects in quantum linear regression", arXiv:2610.12015 (2026) [checked online] | Shot noise in fidelity-kernel / ELM regression; notes an implicit Tikhonov term from shot noise. | Low–medium (regression, shot noise only). | Cite in related work. |
| A8 | V. Scavino, "Decision Kernels for Quantum Error Mitigation", arXiv:2607.02888 (2026) [checked online] | Argues that mitigation accuracy gains need not improve downstream decisions; shift-invariant decisions. Not about kernels or SVMs. | Low (same moral, different object). | Cite as supporting context for "accuracy ≠ decision". |
| A9 | G. Gentinetta, A. Thomsen, D. Sutter, S. Woerner, "The complexity of quantum support vector machines", Quantum 8, 1225 (2024) [checked online] | Shot complexity of QSVM training (dual and primal). | Background. | Cite for shot-cost motivation. |

## B. Foundational and established methods (cite for the parts labelled "established")
- V. Havlíček et al., Nature (2019): ZZ feature map and quantum-kernel SVM.
- M. Schuld, N. Killoran, Phys. Rev. Lett. (2019): quantum feature Hilbert spaces.
- S. Thanasilp, S. Wang, M. Cerezo, Z. Holmes, Nature Communications (2024): exponential concentration and shot cost.
- X. Wang, Y. Du, Y. Luo, D. Tao, Quantum (2021): quantum kernels in the NISQ era (noise and finite shots).
- R. Shaydulin, S. M. Wild, Phys. Rev. A (2022): kernel bandwidth.
- N. J. Higham, Linear Algebra Appl. (1988): nearest PSD matrix (eigenvalue clipping).
- Y. Chen, E. K. Garcia, M. R. Gupta, A. Rahimi, L. Cazzanti, JMLR (2009): clip/flip/shift for indefinite kernels.
- C. Cortes, M. Mohri, A. Rostamizadeh, JMLR (2012): centered kernel alignment.
- J. Neyman, J. R. Stat. Soc. (1934): optimal allocation.
- A. S. Bandeira, R. van Handel, Ann. Probab. (2016): spectral norm of random matrices with independent entries.
- J. Demšar, JMLR (2006): statistical comparison of classifiers.
- D. Lakens, Social Psychological and Personality Science (2017): equivalence testing (TOST).

## C. Component status after this audit

| Component | Status |
|---|---|
| Fidelity kernel, binomial shot model, PSD repair, diagonal mitigation | Established |
| Adaptive shot allocation (pilot + Neyman / Bernoulli) | **Published** (A1, A2): baseline only |
| SV-restricted test measurement | Standard SVM property (A3 notes it): applied to all methods, not claimed |
| Noise-calibrated ridge (C4) | **Largely covered** by A3; only the post-mitigation calibration is new; minor claim |
| Hidden-ridge theorem (C1) | **Not found in A1–A9**; closest is A4 (implicit regularisation via a bound). Claimable as an exact sharpening, after reading A6 fully |
| Direct test of C1 (C2), fair equal-budget benchmark with TOST (C3) | Not found; claimable as empirical contributions |

## D. Remaining checks before submission
1. Read A6 (cross-kernel margin) in full: does its Tikhonov-stabilised dual coincide with the hidden ridge?
2. Obtain the full A2 text and the ShotWise code; replace the approximation.
3. Ask the A1 authors for code or confirm the AQKA-SVM re-implementation details (solver, rounding).
4. Search again just before submission using the queries below (the field is moving fast: A1, A2, A6–A8 all appeared in 2026).

Query strings (arXiv quant-ph / cs.LG, Google Scholar, IEEE Xplore, Scopus):
1. "quantum kernel" AND (mitigation OR "depolarizing") AND (SVM OR "support vector") AND (regularization OR ridge)
2. "quantum kernel" AND "diagonal" AND (depolarizing OR "noise rate")
3. "quantum kernel" AND ("shot allocation" OR "measurement allocation" OR "shot budget")
4. "noisy quantum kernel" AND ("implicit regularization" OR Tikhonov)
5. Forward citations of A4, A5, A1, A2.

## E. Decision rules
- Claim C1 only as an exact result *for the stated noise model*, cite A4 as the conceptual predecessor, and report §8b's
  approximation quality under local noise.
- Claim C3's "mitigation does not help" only where the TOST equivalence test (or a significantly negative effect) supports it.
- Do not claim allocation or the SV-restricted test as contributions.
- Never claim quantum advantage; the RBF-SVM reference is reported for that reason.
