"""Run the kernel circuits of one (or a few) data splits on an IBM Quantum computer.

STEP 0 (once):   python run_ibm_hardware.py --save-token YOUR_API_KEY
STEP 1 (free):   python run_ibm_hardware.py --dry-run          # full pipeline on a local fake IBM backend
STEP 2 (QPU):    python run_ibm_hardware.py                     # asks for confirmation before submitting
Later, more splits:   python run_ibm_hardware.py --first-seed 5001   (results are kept per seed folder)
If the connection drops after submission:  python run_ibm_hardware.py --retrieve [--first-seed N]
Then:            python analyze_hardware.py [--dry-run]

What is measured (defaults: breast_cancer, 1 split, 4 qubits, N_train=30, N_val=10, N_test=30, 256 shots/circuit):
every kernel circuit U(y)^dag U(x) followed by measurement; we keep the number of '0000' outcomes.
Blocks: train off-diagonal (435), train diagonal (30), validation (300 + 10 diagonal), test (900 + 30 diagonal)
= 1,705 circuits per split. All circuits are transpiled onto the SAME physical qubits so that a single
survival factor Lambda is meaningful. No error mitigation, twirling or dynamical decoupling is switched on:
we want the raw device kernel.

Open Plan budget: 10 QPU-minutes per 28-day window. The script prints an estimate and asks before submitting.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from naqsvm.data import load_dataset, make_split, prepare          # noqa: E402
from naqsvm.simulators import build_qiskit_circuit                 # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--save-token", default=None, help="save your IBM Quantum API key once, then exit")
ap.add_argument("--instance", default=None, help="optional instance CRN/name (if you have several)")
ap.add_argument("--dry-run", action="store_true", help="run locally on a fake IBM backend (free)")
ap.add_argument("--retrieve", action="store_true", help="download results of already-submitted jobs")
ap.add_argument("--backend", default=None, help="QPU name, e.g. ibm_torino (default: least busy)")
ap.add_argument("--dataset", default="breast_cancer", help="breast_cancer | iris | wine | synthetic | adhoc")
ap.add_argument("--splits", type=int, default=1, help="number of data splits in this run")
ap.add_argument("--first-seed", type=int, default=5000, help="seed of the first split (use 5001, 5002, ... for later runs)")
ap.add_argument("--n-train", type=int, default=30)
ap.add_argument("--n-val", type=int, default=10)
ap.add_argument("--n-test", type=int, default=30)
ap.add_argument("--shots", type=int, default=256)
ap.add_argument("--batch", type=int, default=300, help="circuits per job")
ap.add_argument("--yes", action="store_true", help="do not ask for confirmation")
args = ap.parse_args()

from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2       # noqa: E402

if args.save_token:
    QiskitRuntimeService.save_account(channel="ibm_quantum_platform", token=args.save_token,
                                      instance=args.instance, overwrite=True, set_as_default=True)
    print("Token saved. Next: python run_ibm_hardware.py --dry-run")
    sys.exit(0)

TAG = "dryrun" if args.dry_run else "qpu"
OUT = os.path.join(HERE, f"hw_{TAG}", f"seed{args.first_seed}")
os.makedirs(OUT, exist_ok=True)
META = os.path.join(OUT, "metadata.json")
REPS, ENT, NQ = 2, "linear", 4
ANGLE = {"breast_cancer": 0.2, "iris": 0.1, "wine": 0.1, "synthetic": 0.1}   # v2 label-free bandwidths


# ------------------------------------------------------------------------------------------
# 1) build the circuit list (deterministic: same seeds -> same circuits)
# ------------------------------------------------------------------------------------------
def load(ds):
    if ds == "adhoc":
        from naqsvm.revision import load_adhoc
        return load_adhoc(NQ, 600, 0.1, REPS, ENT)
    return load_dataset(ds)


def build_jobs():
    X, y = load(args.dataset)
    circs, index = [], []
    for k in range(args.splits):
        seed = args.first_seed + k
        sp = make_split(y, args.n_train, args.n_val, args.n_test, seed)
        if args.dataset == "adhoc":
            A_tr, A_va, A_te = X[sp.idx_train], X[sp.idx_val], X[sp.idx_test]
        else:
            pr = prepare(X, y, sp, NQ, ANGLE[args.dataset], seed)
            A_tr, A_va, A_te = pr.A_train, pr.A_val, pr.A_test
        N = len(A_tr)
        pairs = [("tr_off", i, j, A_tr[i], A_tr[j]) for i in range(N) for j in range(i + 1, N)]
        pairs += [("tr_diag", i, i, A_tr[i], A_tr[i]) for i in range(N)]
        pairs += [("va", i, j, A_va[i], A_tr[j]) for i in range(len(A_va)) for j in range(N)]
        pairs += [("va_diag", i, i, A_va[i], A_va[i]) for i in range(len(A_va))]
        pairs += [("te", i, j, A_te[i], A_tr[j]) for i in range(len(A_te)) for j in range(N)]
        pairs += [("te_diag", i, i, A_te[i], A_te[i]) for i in range(len(A_te))]
        for blk, i, j, a, b in pairs:
            circs.append(build_qiskit_circuit(a, b, REPS, ENT, measure=True))   # U(b)^dag U(a), measure_all
            index.append(dict(split_seed=seed, block=blk, i=i, j=j))
    return circs, pd.DataFrame(index)


def get_backend():
    if args.dry_run:
        try:
            from qiskit_ibm_runtime.fake_provider import FakeTorino as Fake
        except ImportError:
            from qiskit_ibm_runtime.fake_provider import FakeSherbrooke as Fake
        return Fake(), None
    service = QiskitRuntimeService()
    bk = service.backend(args.backend) if args.backend else service.least_busy(operational=True, simulator=False,
                                                                               min_num_qubits=NQ)
    return bk, service


def make_sampler(backend):
    try:
        return SamplerV2(mode=backend)
    except TypeError:                     # older qiskit-ibm-runtime
        return SamplerV2(backend=backend)


def zeros_from(pub_result):
    data = pub_result.data
    reg = getattr(data, "meas", None)
    if reg is None:                       # fall back to the first classical register
        reg = getattr(data, list(data.keys())[0])
    counts = reg.get_counts()
    return int(counts.get("0" * NQ, 0)), int(sum(counts.values()))


# ------------------------------------------------------------------------------------------
# 2) retrieve mode
# ------------------------------------------------------------------------------------------
if args.retrieve:
    meta = json.load(open(META))
    service = QiskitRuntimeService()
    index = pd.read_csv(os.path.join(OUT, "circuit_index.csv"))
    zeros, shots = [], []
    for jid in meta["job_ids"]:
        job = service.job(jid)
        print(jid, job.status())
        res = job.result()
        for r in res:
            z, s = zeros_from(r)
            zeros.append(z); shots.append(s)
    index["zeros"], index["shots"] = zeros, shots
    index.to_csv(os.path.join(OUT, "counts.csv"), index=False)
    print("saved", os.path.join(OUT, "counts.csv"))
    sys.exit(0)

# ------------------------------------------------------------------------------------------
# 3) transpile on one fixed set of physical qubits
# ------------------------------------------------------------------------------------------
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager   # noqa: E402

circs, index = build_jobs()
backend, service = get_backend()
print(f"backend: {backend.name}   circuits: {len(circs)}   shots/circuit: {args.shots}")
pm0 = generate_preset_pass_manager(optimization_level=1, backend=backend, seed_transpiler=7)
first = pm0.run(circs[0])
layout = None
try:
    layout = list(first.layout.initial_index_layout(filter_ancillas=True))
except Exception as exc:
    print("could not read the layout of the first circuit; transpiling without a fixed layout:", exc)
print("physical qubits:", layout)
pm = generate_preset_pass_manager(optimization_level=1, backend=backend, initial_layout=layout, seed_transpiler=7)
t0 = time.time()
isa = pm.run(circs)
ops = pd.Series([sum(v for k, v in c.count_ops().items() if k in ("cz", "ecr", "cx")) for c in isa])
print(f"transpiled in {time.time() - t0:.0f}s; two-qubit gates per circuit: mean {ops.mean():.1f}, max {ops.max()}")

est_s = len(circs) * args.shots * 300e-6 + 2.0 * int(np.ceil(len(circs) / args.batch))
print(f"rough QPU-time estimate: {est_s / 60:.1f} min (Open Plan: 10 min per 28 days; true usage is shown on the dashboard)")
if not args.dry_run and not args.yes:
    if input("Submit to the real QPU? [y/N] ").strip().lower() != "y":
        print("cancelled"); sys.exit(0)

# ------------------------------------------------------------------------------------------
# 4) run in batches, store job ids first (so --retrieve works if the session drops)
# ------------------------------------------------------------------------------------------
index.to_csv(os.path.join(OUT, "circuit_index.csv"), index=False)
sampler = make_sampler(backend)
meta = dict(backend=backend.name, dry_run=args.dry_run, dataset=args.dataset, splits=args.splits,
            n_train=args.n_train, n_val=args.n_val, n_test=args.n_test, shots=args.shots, layout=layout,
            reps=REPS, entanglement=ENT, n_qubits=NQ, angle_scale=ANGLE.get(args.dataset), job_ids=[],
            submitted=time.strftime("%Y-%m-%d %H:%M:%S"), mean_2q_gates=float(ops.mean()))
jobs = []
for s in range(0, len(isa), args.batch):
    job = sampler.run(isa[s:s + args.batch], shots=args.shots)
    jobs.append(job)
    meta["job_ids"].append(job.job_id())
    json.dump(meta, open(META, "w"), indent=2)
    print(f"submitted job {len(jobs)}: {job.job_id()}", flush=True)
zeros, shots = [], []
for k, job in enumerate(jobs):
    res = job.result()
    for r in res:
        z, n = zeros_from(r)
        zeros.append(z); shots.append(n)
    print(f"job {k + 1}/{len(jobs)} done", flush=True)
index["zeros"], index["shots"] = zeros, shots
index.to_csv(os.path.join(OUT, "counts.csv"), index=False)
meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(meta, open(META, "w"), indent=2)
print("saved", os.path.join(OUT, "counts.csv"), "-> now run: python analyze_hardware.py" + (" --dry-run" if args.dry_run else ""))
