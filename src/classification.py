# src/classification.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sympy import rf
from torch import le

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


def make_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.3, random_state: int = 42) -> SplitData:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return SplitData(X_train, X_test, y_train, y_test)



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
