import argparse
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

warnings.filterwarnings('ignore')


def train_models(
    features_path: str | Path,
    *,
    model_tag: str,
    models_dir: str | Path = "models",
    artifacts_dir: Optional[str | Path] = None,
    contamination: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    features_path = Path(features_path)
    models_path = Path(models_dir)
    artifacts_path = Path(artifacts_dir) if artifacts_dir else models_path
    models_path.mkdir(parents=True, exist_ok=True)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    prefix = f"model_{model_tag}"

    X = pd.read_csv(features_path)
    if verbose:
        print(f"data_shape={X.shape[0]}x{X.shape[1]}")

    if "body_bytes_rolling_mean" in X.columns:
        X = X.drop("body_bytes_rolling_mean", axis=1)
        if verbose:
            print("dropped_feature=body_bytes_rolling_mean")

    X_train, X_val = train_test_split(
        X, test_size=test_size, random_state=random_state
    )
    if verbose:
        print(f"split={X_train.shape[0]}/{X_val.shape[0]}")

    n_train = X_train.shape[0]
    min_positive = max(1.0 / max(n_train, 1), 1e-5)
    requested_contamination = float(contamination)
    if not np.isfinite(requested_contamination):
        requested_contamination = 0.0
    if requested_contamination <= 0:
        effective_contamination = min_positive
    else:
        effective_contamination = max(requested_contamination, min_positive)
    effective_contamination = min(effective_contamination, 0.5)
    if verbose:
        print(
            "contamination="
            f"{effective_contamination:.6f} "
            f"(min={min_positive:.6f}, requested={requested_contamination:.6f})"
        )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    scaler_path = artifacts_path / f"{prefix}_scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler, f)
    if verbose:
        print(f"scaler_path={scaler_path}")

    if_model = IsolationForest(
        contamination=effective_contamination,
        n_estimators=100,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    if_model.fit(X_train)

    if_train_scores = if_model.decision_function(X_train)
    if_val_scores = if_model.decision_function(X_val)
    if_train_preds = if_model.predict(X_train)
    if_val_preds = if_model.predict(X_val)
    if_train_anomalies = int((if_train_preds == -1).sum())
    if_val_anomalies = int((if_val_preds == -1).sum())
    if_model_path = models_path / f"{prefix}_isolation_forest.pkl"
    with if_model_path.open("wb") as f:
        pickle.dump(if_model, f)

    lof_model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=effective_contamination,
        novelty=True,
        n_jobs=-1,
    )
    lof_model.fit(X_train_scaled)
    lof_train_scores = lof_model.score_samples(X_train_scaled)
    lof_val_scores = lof_model.score_samples(X_val_scaled)
    lof_train_preds = lof_model.predict(X_train_scaled)
    lof_val_preds = lof_model.predict(X_val_scaled)
    lof_train_anomalies = int((lof_train_preds == -1).sum())
    lof_val_anomalies = int((lof_val_preds == -1).sum())
    lof_model_path = models_path / f"{prefix}_lof.pkl"
    with lof_model_path.open("wb") as f:
        pickle.dump(lof_model, f)

    ocsvm_model = OneClassSVM(
        kernel="rbf",
        gamma="auto",
        nu=effective_contamination,
    )
    ocsvm_model.fit(X_train_scaled)
    ocsvm_train_scores = ocsvm_model.decision_function(X_train_scaled)
    ocsvm_val_scores = ocsvm_model.decision_function(X_val_scaled)
    ocsvm_train_preds = ocsvm_model.predict(X_train_scaled)
    ocsvm_val_preds = ocsvm_model.predict(X_val_scaled)
    ocsvm_train_anomalies = int((ocsvm_train_preds == -1).sum())
    ocsvm_val_anomalies = int((ocsvm_val_preds == -1).sum())
    ocsvm_model_path = models_path / f"{prefix}_ocsvm.pkl"
    with ocsvm_model_path.open("wb") as f:
        pickle.dump(ocsvm_model, f)

    val_results = pd.DataFrame(
        {
            "if_pred": if_val_preds,
            "if_score": if_val_scores,
            "lof_pred": lof_val_preds,
            "lof_score": lof_val_scores,
            "ocsvm_pred": ocsvm_val_preds,
            "ocsvm_score": ocsvm_val_scores,
        }
    )
    val_results["if_anomaly"] = (val_results["if_pred"] == -1).astype(int)
    val_results["lof_anomaly"] = (val_results["lof_pred"] == -1).astype(int)
    val_results["ocsvm_anomaly"] = (val_results["ocsvm_pred"] == -1).astype(int)
    val_results["consensus_anomaly"] = (
        val_results["if_anomaly"]
        + val_results["lof_anomaly"]
        + val_results["ocsvm_anomaly"]
    ) >= 2
    consensus_anomalies = int(val_results["consensus_anomaly"].sum())
    consensus_pct = 100 * consensus_anomalies / len(val_results)

    val_anomaly_mask = (
        val_results["if_anomaly"]
        | val_results["lof_anomaly"]
        | val_results["ocsvm_anomaly"]
    )

    validation_results_path = artifacts_path / f"{prefix}_validation_results.csv"
    val_results.to_csv(validation_results_path, index=False)
    if verbose:
        print(f"isolation_forest_path={if_model_path}")
        print(f"lof_path={lof_model_path}")
        print(f"ocsvm_path={ocsvm_model_path}")
        print(f"validation_path={validation_results_path}")
        print(f"consensus_pct={consensus_pct:.2f}")

    return {
        "models": {
            "isolation_forest": if_model_path,
            "lof": lof_model_path,
            "ocsvm": ocsvm_model_path,
        },
        "scaler_path": scaler_path,
        "validation_results_path": validation_results_path,
        "consensus_pct": consensus_pct,
        "effective_contamination": effective_contamination,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models using engineered features."
    )
    parser.add_argument(
        "--features",
        required=True,
        help="Path to engineered features CSV.",
    )
    parser.add_argument(
        "--model-tag",
        required=True,
        help="Identifier used to namespace saved model artifacts (e.g., index name).",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store trained model files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        help="Directory to store scaler and validation results (defaults to models dir).",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Estimated proportion of outliers in the data.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data to reserve for validation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    args = parser.parse_args()

    train_models(
        args.features,
        model_tag=args.model_tag,
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        contamination=args.contamination,
        test_size=args.test_size,
        random_state=args.random_state,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()