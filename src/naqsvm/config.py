"""Experiment configuration (all knobs in one serialisable place)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class NoiseSpec:
    """A noise level for the kernel circuits.

    kind = 'none'          : no gate noise, no readout error (finite-shot noise only)
    kind = 'depolarizing'  : local depolarizing error after every H (p1) and CX (p2),
                             plus symmetric per-qubit readout error (readout)
    kind = 'global'        : analytic GLOBAL depolarizing channel with survival probability p1
                             (p = p1*k + (1-p1)/d). Used only in the hidden-ridge check.
    kind = 'thermal'       : T1/T2 thermal relaxation after H (t_1q) and CX (t_2q)
                             on every involved qubit, plus readout error.
                             Used to test model mismatch (non-unital noise).
    """
    name: str
    kind: str = "depolarizing"
    p1: float = 0.0
    p2: float = 0.0
    readout: float = 0.0
    t1: float = 50e-6
    t2: float = 50e-6
    t_1q: float = 50e-9
    t_2q: float = 400e-9


@dataclass
class ExperimentConfig:
    mode: str = "FULL"                       # 'QUICK' (smoke test) or 'FULL'
    datasets: List[str] = field(default_factory=lambda: ["iris", "wine", "breast_cancer", "synthetic"])
    # v1 evaluation used seeds 1000-1009; this v2 study uses FRESH seeds 2000-2009 (replication).
    seeds: List[int] = field(default_factory=lambda: list(range(2000, 2010)))
    shots_per_entry: List[int] = field(default_factory=lambda: [16, 64, 256, 1024])
    noise_levels: List[NoiseSpec] = field(default_factory=lambda: [
        NoiseSpec("noise-free", kind="none"),
        NoiseSpec("low", p1=0.001, p2=0.01, readout=0.01),
        NoiseSpec("medium", p1=0.002, p2=0.03, readout=0.02),
        NoiseSpec("high", p1=0.005, p2=0.06, readout=0.04),
        NoiseSpec("thermal (mismatch)", kind="thermal", t1=40e-6, t2=30e-6,
                  t_1q=50e-9, t_2q=500e-9, readout=0.02),
    ])
    # data
    n_train: int = 60
    n_val: int = 30
    n_test: int = 60
    n_qubits: int = 4                        # = number of PCA components
    angle_scale: float = 0.1                 # default bandwidth: features mapped to [0, angle_scale*pi]
    # bandwidth calibration (Sec. 5 of the notebook): chosen per dataset on VALIDATION splits of
    # pilot seeds that are disjoint from the evaluation seeds, using the ideal kernel only.
    bandwidth_grid: List[float] = field(default_factory=lambda: [0.025, 0.05, 0.1, 0.2, 0.5])
    pilot_seeds: List[int] = field(default_factory=lambda: [100, 101, 102])
    angle_scale_by_dataset: Dict[str, float] = field(default_factory=dict)
    # feature map (Havlicek-style ZZ map, linear entanglement)
    reps: int = 2
    entanglement: str = "linear"
    # hyper-parameter grids (selected on the validation split only)
    C_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0])
    ridge_grid: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5, 1.0, 2.0])
    rbf_gamma_grid: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0, "scale"])
    # proposed method
    pilot_fraction: float = 0.25             # fraction of off-diagonal budget spent on the pilot
    lambda_floor: float = 0.05               # lower clip for estimated depolarizing survival prob.
    # v2 fair protocol: every method measures its test kernel only on its support-vector columns
    sv_test_all: bool = True
    # hidden-ridge experiment (Sec. 8b): global-depolarizing survival levels and shot settings
    hidden_ridge_global_lams: List[float] = field(default_factory=lambda: [0.8, 0.5, 0.25])
    hidden_ridge_shots: List[int] = field(default_factory=lambda: [0, 1024, 64])   # 0 = exact
    hidden_ridge_C: float = 1.0
    equivalence_margin: float = 0.01          # TOST margin (balanced accuracy) for "no improvement" claims
    # simulation
    sim_backend: str = "auto"                # 'auto' | 'aer' | 'numpy'
    sim_chunk: int = 1500
    # bookkeeping
    out_dir: str = "noise_aware_qsvm_research"
    selection_metric: str = "balanced_accuracy"

    def to_dict(self) -> Dict:
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def quick_config(**overrides) -> ExperimentConfig:
    """Small grid that runs in a couple of minutes (smoke test)."""
    cfg = ExperimentConfig(
        mode="QUICK",
        datasets=["iris", "breast_cancer"],
        seeds=[0, 1],
        hidden_ridge_shots=[0, 256],
        shots_per_entry=[32, 256],
        noise_levels=[NoiseSpec("noise-free", kind="none"),
                      NoiseSpec("medium", p1=0.002, p2=0.03, readout=0.02)],
        n_train=40, n_val=20, n_test=40,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
