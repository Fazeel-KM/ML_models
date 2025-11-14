import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MODEL_SUFFIXES = {
    "isolation_forest": "_isolation_forest.pkl",
    "lof": "_lof.pkl",
    "ocsvm": "_ocsvm.pkl",
}
SCALER_SUFFIX = "_scaler.pkl"
REDUNDANT_FEATURES = {"body_bytes_rolling_mean"}


@dataclass
class ModelBundle:
    tag: str
    paths: Dict[str, Path]
    scaler_path: Path
    metadata: Dict[str, str]


def _extract_tag(model_path: Path, suffix: str) -> Optional[str]:
    stem = model_path.stem  # e.g. model_events-20251026_isolation_forest
    prefix = "model_"
    suffix_no_ext = suffix.replace(".pkl", "")
    if not stem.startswith(prefix):
        return None
    if not stem.endswith(suffix_no_ext):
        return None
    return stem[len(prefix) : -len(suffix_no_ext)]


def discover_model_bundles(
    models_dir: Path, artifacts_dir: Path
) -> Dict[str, ModelBundle]:
    bundles: Dict[str, Dict[str, Path]] = {}
    for variant, suffix in MODEL_SUFFIXES.items():
        for path in models_dir.glob(f"model_*{suffix}"):
            tag = _extract_tag(path, suffix)
            if not tag:
                continue
            bundles.setdefault(tag, {})[variant] = path

    model_bundles: Dict[str, ModelBundle] = {}
    for tag, paths in bundles.items():
        if len(paths) != len(MODEL_SUFFIXES):
            # Skip incomplete bundles
            continue
        scaler_path = artifacts_dir / f"model_{tag}{SCALER_SUFFIX}"
        if not scaler_path.exists():
            continue
        metadata = {
            "modified": max(path.stat().st_mtime for path in paths.values()),
        }
        model_bundles[tag] = ModelBundle(
            tag=tag,
            paths=paths,
            scaler_path=scaler_path,
            metadata=metadata,
        )
    return model_bundles


def list_feature_datasets(features_dir: Path) -> List[Path]:
    if not features_dir.exists():
        return []
    return sorted(features_dir.glob("*.csv"))


def _format_choices(options: List[str]) -> None:
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {option}")


def _prompt_choice(options: List[str], prompt: str) -> Optional[str]:
    if not options:
        return None
    while True:
        try:
            choice = input(f"{prompt} (number or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return None
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx - 1]
        except ValueError:
            pass
        print(f"Choose 1-{len(options)} or 'q' to quit.")


def _load_feature_columns(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Feature columns file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        columns = [line.strip() for line in f if line.strip()]
    if not columns:
        raise ValueError(f"No feature names found in {path}")
    return columns
    return columns


def _prepare_features(
    df: pd.DataFrame, scaler, feature_columns: Optional[List[str]]
) -> Tuple[pd.DataFrame, np.ndarray]:
    # Drop redundant feature removed during training if present
    features = df.copy()
    features = features.drop(columns=list(REDUNDANT_FEATURES), errors="ignore")

    expected: Optional[List[str]] = None
    if feature_columns:
        expected = feature_columns
    elif hasattr(scaler, "feature_names_in_"):
        expected = list(scaler.feature_names_in_)
    elif hasattr(scaler, "n_features_in_"):
        expected = None

    if expected:
        expected = [col for col in expected if col not in REDUNDANT_FEATURES]
        missing = [col for col in expected if col not in features.columns]
        for col in missing:
            features[col] = 0
        unexpected = [col for col in features.columns if col not in expected]
        if unexpected:
            features = features.drop(columns=unexpected)
        # Ensure scaler sees familiar feature names even if it recorded its own list
        if hasattr(scaler, "feature_names_in_"):
            scaler.feature_names_in_ = np.array(expected)
        features = features[expected]
    elif hasattr(scaler, "n_features_in_"):
        # Align column count by dropping extras and padding missing ones
        current_cols = list(features.columns)
        # Start from scaler expectation count
        needed = scaler.n_features_in_
        if len(current_cols) > needed:
            features = features[current_cols[:needed]]
        elif len(current_cols) < needed:
            for i in range(needed - len(current_cols)):
                placeholder = f"_missing_feature_{i}"
                features[placeholder] = 0
            current_cols = list(features.columns)
            features = features[current_cols[:needed]]

    scaled = scaler.transform(features)
    return features, scaled


def run_evaluation(
    *,
    models_dir: str | Path = "models",
    artifacts_dir: str | Path = "data/artifacts",
    features_dir: Optional[str | Path] = "data/features",
    model_tag: Optional[str] = None,
    dataset_path: Optional[str | Path] = None,
    feature_columns_path: Optional[str | Path] = None,
    metadata_path: Optional[str | Path] = None,
    cleaned_path: Optional[str | Path] = None,
    consensus_threshold: int = 2,
    interactive: bool = True,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:

    models_dir = Path(models_dir)
    artifacts_dir = Path(artifacts_dir)
    features_dir_path = Path(features_dir) if features_dir else None
    metadata_path_obj = Path(metadata_path) if metadata_path else None
    cleaned_path_obj = Path(cleaned_path) if cleaned_path else None

    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")

    bundles = discover_model_bundles(models_dir, artifacts_dir)
    if not bundles:
        raise RuntimeError(
            f"No complete model bundles found in '{models_dir}'. "
            "Ensure Isolation Forest, LOF, OCSVM, and scaler artifacts exist."
        )

    selected_tag = model_tag
    if selected_tag and selected_tag not in bundles:
        raise ValueError(
            f"Model tag '{selected_tag}' not found. "
            f"Available tags: {', '.join(sorted(bundles))}"
        )

    if (not selected_tag or interactive) and interactive:
        tags = sorted(bundles.keys(), reverse=True)
        if not tags:
            raise RuntimeError("No model bundles available for selection.")
        if not selected_tag:
            print("Model bundles:")
            _format_choices(tags)
            selected_tag = _prompt_choice(tags, "Select model bundle")
            if selected_tag is None:
                raise SystemExit(0)

    assert selected_tag is not None
    bundle = bundles[selected_tag]

    dataset = Path(dataset_path) if dataset_path else None
    available_datasets: List[Path] = []
    if not dataset and features_dir_path and features_dir_path.exists():
        available_datasets = list_feature_datasets(features_dir_path)
        if not available_datasets:
            raise RuntimeError(
                f"No feature datasets found in '{features_dir_path}'. "
                "Provide --dataset pointing to a CSV with engineered features."
            )
        if interactive:
            choices = [str(path) for path in available_datasets]
            print("Feature datasets:")
            _format_choices(choices)
            choice = _prompt_choice(choices, "Select dataset")
            if choice is None:
                raise SystemExit(0)
            dataset = Path(choice)
        else:
            dataset = available_datasets[-1]

    if dataset is None:
        raise ValueError("Dataset path must be provided when features_dir is unavailable.")

    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")

    if metadata_path_obj and not metadata_path_obj.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path_obj}")

    if cleaned_path_obj and not cleaned_path_obj.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path_obj}")

    if verbose:
        print(f"model={selected_tag}")
        print(f"dataset={dataset}")

    with bundle.paths["isolation_forest"].open("rb") as f:
        iso_model = pickle.load(f)
    with bundle.paths["lof"].open("rb") as f:
        lof_model = pickle.load(f)
    with bundle.paths["ocsvm"].open("rb") as f:
        ocsvm_model = pickle.load(f)
    with bundle.scaler_path.open("rb") as f:
        scaler = pickle.load(f)

    df = pd.read_csv(dataset)
    if verbose:
        print(f"dataset_shape={df.shape[0]}x{df.shape[1]}")

    metadata_df: Optional[pd.DataFrame] = None
    if metadata_path_obj:
        metadata_df = pd.read_csv(metadata_path_obj)
        if verbose:
            print(f"metadata_shape={metadata_df.shape[0]}x{metadata_df.shape[1]}")

    cleaned_records: Optional[List[dict]] = None
    if cleaned_path_obj:
        with cleaned_path_obj.open("r", encoding="utf-8") as f:
            cleaned_records = json.load(f)
        if not isinstance(cleaned_records, list):
            raise ValueError(
                f"Cleaned dataset at {cleaned_path_obj} must be a JSON array of records."
            )
        if verbose:
            print(f"cleaned_records={len(cleaned_records)}")

    resolved_feature_columns: Optional[List[str]] = None
    feature_columns_candidate: Optional[Path] = None
    if feature_columns_path:
        feature_columns_candidate = Path(feature_columns_path)
    else:
        if dataset.name.endswith("_features.csv"):
            candidate_name = dataset.name.replace(
                "_features.csv", "_feature_columns.txt"
            )
            candidate = dataset.parent / candidate_name
            if candidate.exists():
                feature_columns_candidate = candidate
    if feature_columns_candidate:
        resolved_feature_columns = _load_feature_columns(feature_columns_candidate)

    features, scaled = _prepare_features(df, scaler, resolved_feature_columns)

    iso_scores = iso_model.decision_function(features)
    iso_preds = iso_model.predict(features)
    lof_scores = lof_model.score_samples(scaled)
    lof_preds = lof_model.predict(scaled)
    ocsvm_scores = ocsvm_model.decision_function(scaled)
    ocsvm_preds = ocsvm_model.predict(scaled)

    summary = {}
    total = len(features)
    for name, preds in [
        ("isolation_forest", iso_preds),
        ("lof", lof_preds),
        ("ocsvm", ocsvm_preds),
    ]:
        anomalies = int((preds == -1).sum())
        summary[name] = {
            "total": total,
            "anomalies": anomalies,
            "anomaly_pct": (anomalies / total * 100) if total else 0.0,
        }

    stacked = np.column_stack([iso_preds, lof_preds, ocsvm_preds])
    anomaly_votes = (stacked == -1).sum(axis=1)
    consensus_mask = anomaly_votes >= consensus_threshold
    consensus_count = int(consensus_mask.sum())
    summary["consensus"] = {
        "threshold": consensus_threshold,
        "anomalies": consensus_count,
        "anomaly_pct": (consensus_count / total * 100) if total else 0.0,
    }

    anomaly_details: List[dict] = []
    anomaly_output_path: Optional[Path] = None
    if cleaned_records is not None or metadata_df is not None:
        source_len = (
            len(cleaned_records)
            if cleaned_records is not None
            else len(metadata_df) if metadata_df is not None else 0
        )
        if source_len and source_len != total and verbose:
            print(
                f"warning: record_count_mismatch features={total} source={source_len}"
            )
        limit = min(total, source_len) if source_len else total
        any_mask = (stacked == -1)[:limit].any(axis=1)
        anomaly_indices = np.nonzero(any_mask)[0]
        if anomaly_indices.size and source_len:
            for idx in anomaly_indices:
                if cleaned_records is not None and idx < len(cleaned_records):
                    base = cleaned_records[idx]
                    if isinstance(base, dict):
                        record = dict(base)
                    else:
                        record = {"_record": base}
                elif metadata_df is not None and idx < len(metadata_df):
                    record = metadata_df.iloc[idx].to_dict()
                else:
                    record = {"_record_missing": True}
                record["_anomaly"] = {
                    "row_index": int(idx),
                    "isolation_forest": {
                        "pred": int(iso_preds[idx]),
                        "score": float(iso_scores[idx]),
                    },
                    "lof": {
                        "pred": int(lof_preds[idx]),
                        "score": float(lof_scores[idx]),
                    },
                    "ocsvm": {
                        "pred": int(ocsvm_preds[idx]),
                        "score": float(ocsvm_scores[idx]),
                    },
                    "consensus": {
                        "anomaly": bool(consensus_mask[idx]),
                        "votes": int(anomaly_votes[idx]),
                        "threshold": int(consensus_threshold),
                    },
                }
                anomaly_details.append(record)
            anomaly_output_path = artifacts_dir / f"model_{selected_tag}_anomalies.json"
            with anomaly_output_path.open("w", encoding="utf-8") as f:
                json.dump(anomaly_details, f, indent=2)
            summary["consensus"]["anomaly_records"] = len(anomaly_details)
            summary["consensus"]["anomaly_path"] = str(anomaly_output_path)
        elif verbose and source_len:
            print("anomaly_export=skipped (no anomalies detected)")
    elif verbose:
        print("anomaly_export=skipped (cleaned dataset unavailable)")

    if verbose:
        print("evaluation:")
        consensus_stats = summary.get("consensus")
        if consensus_stats:
            print("  === CONSENSUS SUMMARY ===")
            print(
                f"  anomalies={consensus_stats['anomalies']}, "
                f"pct={consensus_stats['anomaly_pct']:.2f}, "
                f"threshold={consensus_stats['threshold']}"
            )
            print("  -------------------------")
        for name, stats in summary.items():
            if name == "consensus":
                continue
            print(
                f"{name}_anomalies={stats['anomalies']},"
                f"{name}_total={stats['total']},"
                f"{name}_pct={stats['anomaly_pct']:.2f}"
            )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained anomaly detection models on engineered feature datasets. "
            "Select model bundles and datasets interactively or via CLI arguments."
        )
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing trained model pickle files.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default="data/artifacts",
        help="Directory containing scaler artifacts.",
    )
    parser.add_argument(
        "--features-dir",
        default="data/features",
        help="Directory with engineered feature CSV datasets.",
    )
    parser.add_argument(
        "--model-tag",
        help="Specific model tag to evaluate (e.g. events-20251026).",
    )
    parser.add_argument(
        "--dataset",
        help="Path to engineered features CSV to score. If omitted, an interactive prompt is shown.",
    )
    parser.add_argument(
        "--metadata-path",
        help="Path to metadata CSV aligned with the engineered features.",
    )
    parser.add_argument(
        "--cleaned-path",
        help="Path to cleaned JSON records for exporting full anomaly details.",
    )
    parser.add_argument(
        "--feature-columns",
        help="Path to feature columns text file to enforce column ordering.",
    )
    parser.add_argument(
        "--consensus-threshold",
        type=int,
        default=2,
        help="Number of models that must agree to count a consensus anomaly.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output.",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts (requires --model-tag and --dataset or discoverable defaults).",
    )
    args = parser.parse_args()

    interactive = not args.non_interactive and sys.stdin.isatty()

    run_evaluation(
        models_dir=args.models_dir,
        artifacts_dir=args.artifacts_dir,
        features_dir=args.features_dir,
        model_tag=args.model_tag,
        dataset_path=args.dataset,
        metadata_path=args.metadata_path,
        cleaned_path=args.cleaned_path,
        feature_columns_path=args.feature_columns,
        consensus_threshold=args.consensus_threshold,
        interactive=interactive,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()

