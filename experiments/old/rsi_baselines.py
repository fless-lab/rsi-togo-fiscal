"""
rsi_baselines.py
================
Baseline models for comparison with RSI.

Three baselines representative of the current state of practice:

    RuleBasedSystem : Deterministic threshold rules. No uncertainty
                      quantification. Fragile to missing data.

    XGBoostBaseline : Gradient boosting, fully supervised. Requires
                      complete retraining on regulatory changes.

    MLPBaseline     : Multi-layer perceptron, fully supervised. Same
                      supervision constraints as XGBoost.

Note: RSI operates zero-shot (no labeled examples). These baselines
require fully labeled training data, which reflects the real-world
deployment gap in low-resource regulatory environments.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
)
import time
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct feature matrix from observable columns only.

    Uses only obs_* columns, mirroring what a real system would observe.
    Ground truth gt_* columns are never used.
    """
    X = pd.DataFrame()

    X["ca_declare"] = df["obs_ca_declare"].fillna(0)
    X["tva_declaree"] = df["obs_tva_declaree"].fillna(0)
    X["is_declare"] = df["obs_is_declare"].fillna(0)
    X["benefice_declare"] = df["obs_benefice_declare"].fillna(0)
    X["delay_days"] = df["obs_retard_paiement_jours"].fillna(0)
    X["n_employees"] = df["obs_n_employes_declare"].fillna(1)
    X["underdeclaration_ratio"] = df["obs_ratio_sous_declaration"].fillna(1)

    X["has_bank_account"] = df["obs_has_compte_bancaire"].astype(int)
    X["uses_einvoicing"] = df["obs_utilise_facturation_electronique"].astype(int)
    X["was_audited"] = df["obs_a_ete_audite"].astype(int)
    X["vat_missing"] = df["obs_tva_missing"].astype(int)
    X["cit_missing"] = df["obs_is_missing"].astype(int)

    X["vat_over_ca"] = X["tva_declaree"] / (X["ca_declare"] + 1)
    X["cit_over_ca"] = X["is_declare"] / (X["ca_declare"] + 1)
    X["profit_over_ca"] = X["benefice_declare"] / (X["ca_declare"] + 1)
    X["log_ca"] = np.log1p(X["ca_declare"])
    X["log_vat"] = np.log1p(X["tva_declaree"])
    X["above_vat_threshold_60m"] = (X["ca_declare"] >= 60e6).astype(int)
    X["above_vat_threshold_100m"] = (X["ca_declare"] >= 100e6).astype(int)
    X["above_cit_threshold"] = (X["ca_declare"] >= 100e6).astype(int)

    sector_dummies = pd.get_dummies(df["sector"], prefix="sector")
    X = pd.concat([X, sector_dummies], axis=1)

    return X.fillna(0)


# =============================================================================
# RULE-BASED SYSTEM
# =============================================================================

class RuleBasedSystem:
    """
    Deterministic rule-based compliance detector.

    Applies hard threshold rules directly to declared values.
    Represents the current approach in most African tax administrations.

    Limitations:
        - No uncertainty quantification
        - Fragile to missing data (missing = non-compliant by default)
        - Requires manual parameter update on regulatory changes
        - Cannot learn from behavioral patterns
    """

    def __init__(self, vat_threshold: float = 60_000_000):
        self.vat_threshold = vat_threshold
        self.name = "Rule-Based System"

    def predict(self, X: pd.DataFrame, df_orig: pd.DataFrame) -> np.ndarray:
        predictions = np.zeros(len(X), dtype=int)

        for i, (_, row) in enumerate(X.iterrows()):
            ca = row["ca_declare"]
            vat = row["tva_declaree"]
            cit = row["is_declare"]
            delay = row["delay_days"]
            non_compliant = False

            if not row.get("vat_missing", 0):
                if ca >= self.vat_threshold and vat == 0:
                    non_compliant = True
                elif ca < self.vat_threshold and vat > 0:
                    non_compliant = True

            if not row.get("cit_missing", 0):
                if ca >= 100e6 and cit == 0:
                    non_compliant = True

            if delay > 90:
                non_compliant = True

            predictions[i] = int(non_compliant)

        return predictions

    def update_regulatory_params(self, new_threshold: float) -> dict:
        """Update VAT threshold. Deterministic, no uncertainty quantification."""
        old = self.vat_threshold
        self.vat_threshold = new_threshold
        return {
            "old_threshold": old,
            "new_threshold": new_threshold,
            "cost": "O(1) — deterministic, no uncertainty quantification",
        }


# =============================================================================
# XGBOOST BASELINE
# =============================================================================

class XGBoostBaseline:
    """
    Gradient boosting baseline (fully supervised).

    Strong predictive performance under full supervision, but:
        - Requires labeled compliance examples
        - Requires full retraining on regulatory changes
        - Black-box output, not interpretable by auditors
        - Degrades under missing data without explicit imputation
    """

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.name = "XGBoost (Gradient Boosting)"
        self.train_time = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = X.columns.tolist()
        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.train_time = time.time() - t0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict(self.scaler.transform(X_aligned))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict_proba(self.scaler.transform(X_aligned))[:, 1]


# =============================================================================
# MLP BASELINE
# =============================================================================

class MLPBaseline:
    """
    Multi-layer perceptron baseline (fully supervised).

    Same supervision constraints as XGBoost, with additional limitations:
        - Less interpretable than tree-based methods
        - More sensitive to hyperparameter choices
        - Requires sufficient labeled data to converge
    """

    def __init__(self):
        self.model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.name = "MLP (Neural Network)"
        self.train_time = None

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_names = X.columns.tolist()
        t0 = time.time()
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.train_time = time.time() - t0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict(self.scaler.transform(X_aligned))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_aligned = X.reindex(columns=self.feature_names, fill_value=0)
        return self.model.predict_proba(self.scaler.transform(X_aligned))[:, 1]


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    model_name: str = "",
) -> dict:
    """Compute standard classification metrics."""
    results = {
        "model": model_name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_proba is not None:
        try:
            results["auc_roc"] = round(roc_auc_score(y_true, y_proba), 4)
        except Exception:
            results["auc_roc"] = None
    return results