"""Datasets, leakage-safe splits and preprocessing.

Leakage rules enforced here:
  * train / validation / test index sets are disjoint (asserted);
  * every fitted transform (StandardScaler -> PCA -> MinMaxScaler) is fitted on the
    TRAINING split only and then applied to validation and test;
  * the test split is touched exactly once per method (final evaluation).
All datasets ship with scikit-learn (no download required).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn import datasets as skd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(name: str):
    if name == "iris":
        d = skd.load_iris()
        return d.data, d.target
    if name == "wine":
        d = skd.load_wine()
        return d.data, d.target
    if name == "breast_cancer":
        d = skd.load_breast_cancer()
        return d.data, d.target
    if name == "synthetic":
        # fixed generator seed: the *dataset* is fixed, the split varies with the seed
        X, y = skd.make_classification(n_samples=400, n_features=8, n_informative=4,
                                       n_redundant=2, n_classes=2, class_sep=1.0,
                                       flip_y=0.03, random_state=12345)
        return X, y
    if name == "digits_3v8":
        d = skd.load_digits()
        m = np.isin(d.target, [3, 8])
        return d.data[m], (d.target[m] == 8).astype(int)
    raise ValueError(f"unknown dataset {name!r}")


@dataclass
class Split:
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def make_split(y: np.ndarray, n_train: int, n_val: int, n_test: int, seed: int) -> Split:
    """Stratified, seed-controlled train/val/test sub-sampling without overlap."""
    idx = np.arange(len(y))
    tr, rest = train_test_split(idx, train_size=n_train, stratify=y, random_state=seed)
    n_val = min(n_val, len(rest) - 2)
    va, rest2 = train_test_split(rest, train_size=n_val, stratify=y[rest], random_state=seed + 1000)
    if n_test < len(rest2):
        te, _ = train_test_split(rest2, train_size=n_test, stratify=y[rest2], random_state=seed + 2000)
    else:
        te = rest2
    s = Split(np.sort(tr), np.sort(va), np.sort(te))
    assert not (set(s.idx_train) & set(s.idx_val)), "train/val overlap"
    assert not (set(s.idx_train) & set(s.idx_test)), "train/test overlap"
    assert not (set(s.idx_val) & set(s.idx_test)), "val/test overlap"
    return s


@dataclass
class Prepared:
    # quantum inputs (angles)
    A_train: np.ndarray
    A_val: np.ndarray
    A_test: np.ndarray
    # classical inputs
    Z_train: np.ndarray      # standardized + PCA (same information as the quantum input)
    Z_val: np.ndarray
    Z_test: np.ndarray
    F_train: np.ndarray      # standardized full feature set
    F_val: np.ndarray
    F_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    explained_variance: float


def prepare(X, y, split: Split, n_components: int, angle_scale: float, seed: int) -> Prepared:
    Xtr, Xva, Xte = X[split.idx_train], X[split.idx_val], X[split.idx_test]
    sc = StandardScaler().fit(Xtr)                         # fit on train only
    Ftr, Fva, Fte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    k = min(n_components, Xtr.shape[1])
    pca = PCA(n_components=k, random_state=seed).fit(Ftr)  # fit on train only
    Ztr, Zva, Zte = pca.transform(Ftr), pca.transform(Fva), pca.transform(Fte)
    if k < n_components:                                   # pad (not used by default datasets)
        pad = lambda Z: np.hstack([Z, np.zeros((len(Z), n_components - k))])
        Ztr, Zva, Zte = pad(Ztr), pad(Zva), pad(Zte)
    mm = MinMaxScaler(feature_range=(0.0, angle_scale * np.pi)).fit(Ztr)  # fit on train only
    return Prepared(mm.transform(Ztr), mm.transform(Zva), mm.transform(Zte),
                    Ztr, Zva, Zte, Ftr, Fva, Fte,
                    y[split.idx_train], y[split.idx_val], y[split.idx_test],
                    float(np.sum(pca.explained_variance_ratio_)))
