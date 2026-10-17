"""naqsvm: Noise-aware, shot-adaptive quantum kernel SVM research code.

Modules
-------
config      experiment configuration dataclasses
simulators  fidelity-kernel circuits; exact noisy probabilities via Qiskit Aer
            (density matrix) or a NumPy density-matrix reference simulator
data        datasets, leakage-safe splits and preprocessing
shots       finite-shot (binomial) measurement model and shot-allocation rules
kernels     kernel post-processing (PSD repair, depolarizing mitigation) and metrics
methods     end-to-end classifiers: baselines, proposed method, ablations
experiment  experiment grid runner with caching, logging and error handling
stats       confidence intervals and paired statistical tests
plots       figures
"""
__version__ = "1.0.0"
