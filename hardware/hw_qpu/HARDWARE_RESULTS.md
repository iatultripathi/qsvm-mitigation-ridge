# Hardware results (real QPU: ibm_fez)

Dataset breast_cancer, 3 split(s), N_train=30, N_val=10, N_test=30, 256 shots/circuit, physical qubits [0, 1, 2, 3], mean two-qubit gates per circuit 21.1.

## Device kernel and Corollary 1

|   split_seed |   shots |   lambda_hat |   r_hat |   corollary1_rel_diff |   corollary1_agreement |   min_eig_raw |   kernel_rel_err_raw |   ideal_qsvm |    rbf |
|-------------:|--------:|-------------:|--------:|----------------------:|-----------------------:|--------------:|---------------------:|-------------:|-------:|
|         5000 |     256 |       0.9293 | 0.07132 |             8.602e-09 |                      1 |        0.2142 |               0.3759 |       0.8636 | 0.9737 |
|         5000 |      64 |       0.9294 | 0.07117 |             3.266e-08 |                      1 |        0.066  |               0.3893 |       0.8636 | 0.9737 |
|         5001 |     256 |       0.9301 | 0.07041 |             4.844e-08 |                      1 |        0.2008 |               0.4253 |       0.6555 | 0.9091 |
|         5001 |      64 |       0.925  | 0.07601 |             3.278e-08 |                      1 |        0.1523 |               0.4422 |       0.6555 | 0.9091 |
|         5002 |     256 |       0.9247 | 0.07632 |             5.209e-08 |                      1 |        0.2778 |               0.3823 |       0.5    | 0.9545 |
|         5002 |      64 |       0.9228 | 0.07845 |             1.16e-08  |                      1 |        0.1637 |               0.4006 |       0.5    | 0.9545 |

## Balanced accuracy: raw vs mitigated, fixed and tuned C (mean over splits)

|                           |   bacc_C0.1 |   bacc_C1 |   bacc_C10 |   bacc_C100 |   bacc_tuned |   bacc_tuned_robust |
|:--------------------------|------------:|----------:|-----------:|------------:|-------------:|--------------------:|
| (64, 'ideal (no ridge)')  |         0.5 |     0.695 |      0.749 |       0.74  |        0.725 |               0.779 |
| (64, 'mitigated')         |         0.5 |     0.573 |      0.738 |       0.738 |        0.731 |               0.731 |
| (64, 'mitigated+clip')    |         0.5 |     0.573 |      0.738 |       0.738 |        0.731 |               0.731 |
| (64, 'raw')               |         0.5 |     0.565 |      0.746 |       0.746 |        0.731 |               0.731 |
| (64, 'raw+clip')          |         0.5 |     0.565 |      0.746 |       0.746 |        0.731 |               0.731 |
| (256, 'ideal (no ridge)') |         0.5 |     0.695 |      0.749 |       0.74  |        0.725 |               0.779 |
| (256, 'mitigated')        |         0.5 |     0.589 |      0.699 |       0.699 |        0.522 |               0.619 |
| (256, 'mitigated+clip')   |         0.5 |     0.589 |      0.699 |       0.699 |        0.522 |               0.619 |
| (256, 'raw')              |         0.5 |     0.589 |      0.683 |       0.683 |        0.522 |               0.662 |
| (256, 'raw+clip')         |         0.5 |     0.589 |      0.683 |       0.683 |        0.522 |               0.662 |

`bacc_tuned` uses the simulation protocol (validation only, ties -> smallest C); with a 10-point validation set it often collapses to a one-class predictor (0.5). `bacc_tuned_robust` uses 5-fold CV on the training kernel + validation, ties -> median C; report this one for the hardware split.

Expected from the simulations: Corollary 1 relative difference ~1e-7 unless the kernel is strongly indefinite; mitigated minus raw positive at small fixed C, negative at large fixed C, near zero with tuned C. With one split this is a sanity check, not a statistical test.
