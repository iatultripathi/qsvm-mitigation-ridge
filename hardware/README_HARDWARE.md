# Running the hardware check on an IBM quantum computer

Goal: measure the kernel circuits of one or two data splits on a real IBM QPU, then check three things on real device
data. First, that Corollary 1 holds. Second, what Λ̂ and the hidden ridge r̂ the device actually has. Third, whether
the fixed-C vs tuned-C mitigation pattern from the simulations appears. This closes the "simulation only" gap with a
hardware sanity check.

**Budget.** The free Open Plan gives 10 QPU-minutes per 28-day rolling window, and Open Plan instances are created in
the us-east region. Since March 2026, active Open Plan users can also opt in to 180 extra minutes over 12 months; look
for the offer on your dashboard. One split (1,705 circuits × 256 shots) is estimated at roughly 2–3 QPU-minutes, but
this is a rough figure.

## 1. Create the account and an API key (once)
1. Go to https://quantum.cloud.ibm.com and sign up with an IBMid. Your university e-mail is fine.
2. Create an instance on the **Open Plan** (region us-east).
3. On the Platform home dashboard, create an **API key** and copy it.

## 2. Install Qiskit on your Mac (once)
Open Terminal:
```bash
cd ~/Library/CloudStorage/OneDrive-manit.ac.in/IEEE_ACCESS_2/noise_aware_qsvm_research/hardware
python3 -m venv ~/qhw
source ~/qhw/bin/activate
pip install qiskit qiskit-ibm-runtime qiskit-aer scikit-learn pandas scipy matplotlib tabulate
python run_ibm_hardware.py --save-token PASTE_YOUR_API_KEY_HERE
```
(Each new Terminal session: `source ~/qhw/bin/activate` and `cd` into this folder again.)

## 3. Free dry run on a fake IBM backend (do this first)
```bash
python run_ibm_hardware.py --dry-run --n-train 10 --n-val 5 --n-test 10     # quick: checks the whole pipeline
python run_ibm_hardware.py --dry-run                                           # full size, still free (local)
python analyze_hardware.py --dry-run
```
This simulates a real IBM device model on your Mac and costs no QPU time. If anything fails, copy the error message
and send it to Claude before going to step 4.

## 4. Real QPU run
```bash
python run_ibm_hardware.py
```
The script picks the least-busy QPU (or use `--backend ibm_torino`, for example). It prints the number of circuits,
the two-qubit gate count and a time estimate, then asks **"Submit to the real QPU? [y/N]"**. Queueing can take
minutes to hours. You can close the laptop: the job IDs are saved, and later you run
```bash
python run_ibm_hardware.py --retrieve
```
After the first run, check "usage" on the IBM dashboard. If enough minutes are left, add a second split:
```bash
python run_ibm_hardware.py --first-seed 5001
```

## 5. Analyse
```bash
python analyze_hardware.py
```
This writes `hw_qpu/HARDWARE_RESULTS.md`, `hardware_results.csv` and `hardware_accuracy.csv`. Send these back to
Claude to turn them into a paper table and figure.

## Notes
* All circuits are placed on the same 4 physical qubits, with no twirling, dynamical decoupling or readout
  mitigation. The point is to measure the raw device kernel, which the diagonal-based mitigation then corrects.
* Default dataset: breast_cancer with the v2 bandwidth. `--dataset adhoc` runs the quantum-structured dataset
  instead. Use the same dataset for every split you combine.
* These scripts were checked here only partly. The analysis was tested end to end on simulated device counts, but
  the IBM-specific calls (account, transpiler, SamplerV2) could not be run in this environment because Qiskit could not
  be downloaded. That is why step 3 exists.
