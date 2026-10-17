# Revised theory section (v3) — drop-in text for the manuscript

Notation as in the notebook §3: $d=2^n$, measured all-zero probabilities $p_{ij}$, raw kernel $K_{\text{raw}}$ with
off-diagonal entries $\hat p_{ij}$ (finite-shot estimates) and diagonal fixed to 1; validation/test rows
$\hat p(\mathbf x,\mathbf x_i)$ are used as measured.

**Lemma (SVM invariances, unchanged).** For a soft-margin SVM with intercept,
(i) $K\to K+c\,\mathbf 1\mathbf 1^\top$ (train and test rows shifted by the same $c$) leaves the decision function
unchanged; (ii) $\mathrm{SVM}(sK,C)\equiv\mathrm{SVM}(K,sC)$ for $s>0$. Both hold per one-vs-one sub-problem.

**Theorem 1 (hidden ridge, unchanged).** Under global depolarizing noise $p_{ij}=\Lambda k_{ij}+(1-\Lambda)/d$ with exact
probabilities, $\mathrm{SVM}(K_{\text{raw}},C)\equiv\mathrm{SVM}(K+rI,\Lambda C)$, $r=(1-\Lambda)(1-1/d)/\Lambda$.

**Corollary 1 (finite-shot, model-free form).** Let $\hat\Lambda\in(0,1]$ be *any* scalar — in practice the
diagonal-based estimate $\hat\Lambda=(\overline{\hat p_{ii}}-1/d)/(1-1/d)$ — and let global ("average") mitigation be
$K_{\text{mit}}=(K_{\text{raw}}-1/d)/\hat\Lambda+1/d$ off the diagonal, $\operatorname{diag}K_{\text{mit}}=1$, applied with
the same $\hat\Lambda$ to validation and test rows. Then, for every shot budget and every noise channel,
$$
K_{\text{raw}}=\hat\Lambda\,(K_{\text{mit}}+\hat r I)+\tfrac{1-\hat\Lambda}{d}\mathbf 1\mathbf 1^\top,\qquad
\hat r=\frac{(1-\hat\Lambda)(1-1/d)}{\hat\Lambda},
$$
every test row satisfies $\hat k_{\text{raw}}=\hat\Lambda\,\hat k_{\text{mit}}+(1-\hat\Lambda)/d$, and therefore
$$\mathrm{SVM}(K_{\text{raw}},C)\;\equiv\;\mathrm{SVM}(K_{\text{mit}}+\hat rI,\;\hat\Lambda C).$$
*Proof.* Off the diagonal the right side is $\hat\Lambda[(\hat p_{ij}-1/d)/\hat\Lambda+1/d]+(1-\hat\Lambda)/d=\hat p_{ij}$;
on the diagonal it is $\hat\Lambda(1+\hat r)+(1-\hat\Lambda)/d=1$. The test rows follow in the same way. Apply the Lemma. ∎

*Remarks.* (a) No noise model is used: global mitigation **is**, exactly, "remove the ridge $\hat r$ and rescale $C$".
Theorem 1 is the special case in which $K_{\text{mit}}=K$ (global depolarizing, infinite shots). With finite shots,
$K_{\text{mit}}=K+E/\hat\Lambda$ where $E$ collects shot noise and any departure from the global model, so comparing
the finite-shot raw SVM with the *noise-free* $K+rI$ (as in v2, §8b) mixes the ridge with $E$; that is why the v2 test
weakened at 64 shots. (b) Errors in $\hat\Lambda$ do not break the identity; they only change $\hat r$.
(c) The identity is exact for the optimisation problem. When $K_{\text{raw}}$ is strongly indefinite (very few shots),
the SVM dual is non-convex and its solution is not unique; a numerical solver may then stop at different stationary
points for the two equivalent problems (observed only at 16 shots/entry). (d) The "split-channel" mitigation
($\Lambda_{ij}=\sqrt{\Lambda_i\Lambda_j}$) is a congruence $D^{-1}KD^{-1}$, not a scalar affine map, and is not covered.

**Corollary 2 (mitigation does not change the signal-to-noise ratio).** Under the global model, the raw off-diagonal
entries are $\Lambda k_{ij}+(1-\Lambda)/d+e_{ij}$ and the mitigated entries are $k_{ij}+e_{ij}/\hat\Lambda$; the affine
calibration changes, the ratio of informative variation to noise does not. The statement in v2 ("mitigation amplifies
shot noise, $\operatorname{Var}(K_c)=\operatorname{Var}(\hat k)/\Lambda^2$") is true for the absolute variance but has no
consequence for the classifier, because the SVM is invariant to the rescaling (Lemma (ii)). **Replace consequence (c) of
v2 §3.4 by Corollary 2.**

**Proposition 3 (why the ridge is largely absorbed by C).** The dual of $\mathrm{SVM}(K+rI,C)$ is
$$\max_{0\le\alpha\le C,\;y^\top\alpha=0}\;\mathbf 1^\top\alpha-\tfrac12\alpha^\top Y K Y\alpha-\tfrac r2\|\alpha\|^2 ,$$
i.e. the hinge-loss (box $C$) dual plus the dual penalty of a squared-hinge (L2-SVM) slack term of weight $1/(2r)$
(Cortes & Vapnik 1995; the L2-SVM is a hard-margin SVM on $K+I/C'$). Both $C$ and $r$ therefore trade margin width
against margin violations. Removing $r$ (= mitigation, Cor. 1) changes the classifier at a fixed $C$, but re-tuning $C$
on validation can largely compensate. **Testable predictions:** (P2a) at fixed $C$, the accuracy change caused by
mitigation is non-zero and grows with $\hat r$ (i.e. with noise); (P2b) with $C$ tuned on validation it shrinks
towards zero. Both are tested in `results/revision/` (Table R2, Fig. R2).

## Revised claims

| v2 claim | v3 status |
|---|---|
| C1 Theorem 1 (global depolarizing, exact) | kept; now stated together with **Corollary 1** (exact for any noise and any shot budget) and Corollary 2 |
| C2 direct test of C1 (weakened at 64 shots) | replaced by the Corollary-1 test (exact to solver precision at every shot budget ≥ 64; Table R1, Fig. R1) plus a decomposition of the v2 discrepancy into the measured-noise term $E/\Lambda$ |
| C3 "mitigation does not help" (equivalence, tuned C) | sharpened into predictions P2a/P2b and tested at fixed and tuned $C$ (Table R2, Fig. R2) |
| "mitigation can hurt" | only claimed where the data support it: at fixed $C$ (see Table R2 for the effect sizes); not claimed for tuned $C$ |
| C4 NA-QSVM | unchanged (practical variant, no headline claim) |
| scale | extended: N=200 (4 qubits), 6 qubits, device-like noise with coherent error, quantum-structured data (Table R4, Fig. R4); hardware still listed as a limitation |
