from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from .config import RANDOM_STATE

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

# Navodim sve modele koje cu koristiti
DEFAULT_MODELS: List[str] = ["RandomForest", "SVM", "NaiveBayes", "XGBoost", "LightGBM"]
SCORING = ["accuracy", "f1_macro"]

# Ova funkcija mi definise modele i njihove hiperparametre koje cu koristiti
def make_model(name: str, random_state: int = RANDOM_STATE) -> Any:
    key = name.lower()

    if key in ("randomforest", "rf"):
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        )

    if key in ("svm", "svc"):
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced")

    if key in ("naivebayes", "nb"):
        return GaussianNB()

    if key in ("xgboost", "xgb"):
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed.")
        return XGBClassifier(
            objective="multi:softprob",
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
        )

    if key in ("lightgbm", "lgbm"):
        if LGBMClassifier is None:
            raise ImportError("lightgbm is not installed.")
        return LGBMClassifier(
            objective="multiclass",
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

    raise ValueError(f"Unknown model name: {name!r}")

# U odnosu na to kog je tipa matrica korsitim razlicite redukcije dimenzioalnost jer PCA ne radi dobro sa sparse matricama
# TruncatedSVD je analog PCA samo bez centriranja
def _make_reducer(X, n_components: int, random_state: int):
    if sparse.issparse(X):
        return TruncatedSVD(n_components=n_components, random_state=random_state)
    return PCA(n_components=n_components, random_state=random_state)


def cv_evaluate_with_pca(
    X,
    y,
    model_name: str,
    n_splits: int = 5,
    n_components: int = 50,
    random_state: int = RANDOM_STATE,
) -> dict:
    y = LabelEncoder().fit_transform(np.asarray(y))

    pipe = Pipeline(
        [
            ("reduce", _make_reducer(X, n_components, random_state)),
            ("clf", make_model(model_name, random_state)),
        ]
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) # Stratifikovana podela zbog klasifikacije
    scores = cross_validate(pipe, X, y, cv=cv, scoring=SCORING) # Ovo zapravo vraca ocene na svakom test delu tokom treniranja modela

    return {
        "accuracy_mean": float(scores["test_accuracy"].mean()),
        "accuracy_std": float(scores["test_accuracy"].std(ddof=1)),
        "macro_f1_mean": float(scores["test_f1_macro"].mean()),
        "macro_f1_std": float(scores["test_f1_macro"].std(ddof=1)),
        "n_splits": int(n_splits),
        "n_components": int(n_components),
    }

# Ova funkcjia je implementirana da ako jedan model "pukne" to ne bude slucaj sa celim pipeline-om
def cv_compare_models_with_pca(
    X,
    y,
    model_names: List[str] = DEFAULT_MODELS,
    n_splits: int = 5,
    n_components: int = 50,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    rows = []
    for name in model_names:
        try:
            rows.append(
                {"model": name, **cv_evaluate_with_pca(X, y, name, n_splits, n_components, random_state)}
            )
        except Exception as e:
            rows.append({"model": name, "error": str(e)})

    df = pd.DataFrame(rows)
    if "macro_f1_mean" in df.columns:
        df = df.sort_values("macro_f1_mean", ascending=False, na_position="last")
    return df