# src/classification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA, TruncatedSVD
from scipy import sparse

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def evaluate(y_true, y_pred, labels=None) -> Dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, labels=labels, zero_division=0),
        "cm": confusion_matrix(y_true, y_pred, labels=labels),
    }


def train_random_forest(split: SplitData, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",     # <- dodato radi nejednakih klasa
        n_jobs=-1,
        random_state=random_state,
    )
    rf.fit(split.X_train, split.y_train)
    pred = rf.predict(split.X_test)
    labels = np.unique(np.concatenate([split.y_train, split.y_test]))
    return rf, evaluate(split.y_test, pred, labels=labels)


def train_svm(split: SplitData) -> Tuple[Any, Dict[str, Any]]:
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        probability=True,
        class_weight="balanced",
    )
    svm.fit(split.X_train, split.y_train)
    pred = svm.predict(split.X_test)
    labels = np.unique(np.concatenate([split.y_train, split.y_test]))
    return svm, evaluate(split.y_test, pred, labels=labels)

def train_naive_bayes(split: SplitData) -> Tuple[Any, Dict[str, Any]]:
    nb = GaussianNB()
    nb.fit(split.X_train, split.y_train)
    pred = nb.predict(split.X_test)
    labels = np.unique(np.concatenate([split.y_train, split.y_test]))
    return nb, evaluate(split.y_test, pred, labels=labels)


    
def train_xgboost(split: SplitData, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    if XGBClassifier is None:
        raise ImportError("xgboost nije instaliran. Instaliraj: pip install xgboost")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(split.y_train)
    y_test_enc = le.transform(split.y_test)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train_enc)),
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=-1,
    )
    xgb.fit(split.X_train, y_train_enc)
    pred_enc = xgb.predict(split.X_test)
    pred = le.inverse_transform(pred_enc)
    labels = np.unique(np.concatenate([split.y_train, split.y_test]))
    metrics = evaluate(split.y_test, pred, labels=labels)
    metrics["label_encoder"] = le
    return xgb, metrics

def train_lightgbm(split: SplitData, random_state: int = 42) -> Tuple[Any, Dict[str, Any]]:
    if LGBMClassifier is None:
        raise ImportError("lightgbm nije instaliran. Instaliraj: pip install lightgbm")

    le = LabelEncoder()
    y_train_enc = le.fit_transform(split.y_train)
    y_test_enc = le.transform(split.y_test)

    lgbm = LGBMClassifier(
        objective="multiclass",
        num_class=len(np.unique(y_train_enc)),
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=50,
        min_data_in_leaf=5,
        class_weight="balanced",
        verbose=-1,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )
    lgbm.fit(split.X_train, y_train_enc)
    pred_enc = lgbm.predict(split.X_test)
    pred = le.inverse_transform(pred_enc)
    labels = np.unique(np.concatenate([split.y_train, split.y_test]))
    metrics = evaluate(split.y_test, pred, labels=labels)
    metrics["label_encoder"] = le
    return lgbm, metrics


def _fit_transform_dimred(
    X_train,
    X_test,
    n_components: int = 50,
    random_state: int = 42,
):
    """
    Fit dim-reduction samo na train, pa transformiši train i test.
    - Dense -> PCA
    - Sparse -> TruncatedSVD (PCA nije praktična direktno na sparse)
    """
    if sparse.issparse(X_train):
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
    else:
        reducer = PCA(n_components=n_components, random_state=random_state)

    X_train_red = reducer.fit_transform(X_train)
    X_test_red = reducer.transform(X_test)
    return reducer, X_train_red, X_test_red


def cv_evaluate_with_pca(
    X,
    y,
    trainer_fn,
    n_splits: int = 5,
    n_components: int = 50,
    random_state: int = 42,
    return_fold_metrics: bool = False,
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    accs = []
    f1s = []
    fold_rows = []

    y = np.asarray(y)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        reducer, X_train_red, X_test_red = _fit_transform_dimred(
            X_train, X_test, n_components=n_components, random_state=random_state
        )

        split = SplitData(
            X_train=X_train_red,
            X_test=X_test_red,
            y_train=y_train,
            y_test=y_test,
        )

        model, m = trainer_fn(split)  # m sadrži accuracy/macro_f1/...
        accs.append(m["accuracy"])
        f1s.append(m["macro_f1"])

        if return_fold_metrics:
            fold_rows.append({
                "fold": fold,
                "accuracy": m["accuracy"],
                "macro_f1": m["macro_f1"],
            })

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        "n_splits": int(n_splits),
        "n_components": int(n_components),
    }

    if return_fold_metrics:
        return summary, pd.DataFrame(fold_rows)

    return summary


def cv_compare_models_with_pca(
    X,
    y,
    trainers: Dict[str, Any],
    n_splits: int = 5,
    n_components: int = 50,
    random_state: int = 42,
):
    rows = []
    for name, fn in trainers.items():
        try:
            s = cv_evaluate_with_pca(
                X=X,
                y=y,
                trainer_fn=fn,
                n_splits=n_splits,
                n_components=n_components,
                random_state=random_state,
            )
            rows.append({"model": name, **s})
        except Exception as e:
            rows.append({"model": name, "error": str(e)})

    df = pd.DataFrame(rows)
    if "macro_f1_mean" in df.columns:
        df = df.sort_values("macro_f1_mean", ascending=False, na_position="last")
    return df