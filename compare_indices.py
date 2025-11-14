"""
Standalone utility to download two OpenSearch indices, preprocess them, and
compare their characteristics.

Example:
    python compare_indices.py --serial-a 1 --serial-b 2 --report reports/compare.json

The script reuses the existing preprocessing and feature-engineering logic from
the pipeline so comparisons reflect the same transformations applied during
training.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from opensearchpy import OpenSearch

from download_indexWise import (
    ES_URL,
    _format_indices,
    _prompt_for_index,
    download_index,
    list_user_indices,
)
from feature_engineering import run_feature_engineering
from filter import preprocess_and_analyze_api_dataset


@dataclass
class DatasetArtifacts:
    index_name: str
    raw_path: Path
    cleaned_path: Path
    features_path: Path
    feature_columns_path: Path
    stats: Dict


def fetch_user_indices(es_url: str) -> List[dict]:
    es = OpenSearch(es_url)
    try:
        indices = list_user_indices(es)
    finally:
        es.transport.close()
    return indices


def select_index(
    *,
    indices: List[dict],
    index_name: Optional[str],
    index_serial: Optional[int],
    label: str,
    verbose: bool,
) -> str:
    if not indices:
        raise RuntimeError("No indices available in OpenSearch.")

    if index_name:
        for idx in indices:
            if idx["index"] == index_name:
                if verbose:
                    print(f"{label}_index={index_name}")
                return index_name
        available = ", ".join(sorted(entry["index"] for entry in indices))
        raise ValueError(
            f"[{label}] Index '{index_name}' not found. Available: {available}"
        )

    if index_serial is not None:
        if index_serial < 1 or index_serial > len(indices):
            raise ValueError(
                f"[{label}] Index serial {index_serial} out of range 1-{len(indices)}."
            )
        selected = indices[index_serial - 1]["index"]
        if verbose:
            print(f"{label}_index={selected}")
        return selected

    # Interactive selection
    if verbose:
        print(f"{label}_indices:")
        _format_indices(indices)
    choice = _prompt_for_index(indices)
    if choice is None:
        raise SystemExit(0)
    if verbose:
        print(f"{label}_index={choice}")
    return choice


def ensure_dataset(
    *,
    index_name: str,
    es_url: str,
    data_dir: Path,
    force_download: bool,
    verbose: bool,
) -> DatasetArtifacts:
    raw_dir = data_dir / "raw"
    clean_dir = data_dir / "clean"
    features_dir = data_dir / "features"

    raw_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    raw_path = raw_dir / f"{index_name}.json"
    if not raw_path.exists() or force_download:
        if verbose:
            print(f"{index_name}_download=true")
        _, raw_path = download_index(
            index_name=index_name,
            output_dir=raw_dir,
            es_url=es_url,
            interactive=False,
        )
    elif verbose:
        print(f"{index_name}_cached={raw_path}")

    cleaned_path = clean_dir / f"{index_name}_cleaned.json"
    if verbose:
        print(f"{index_name}_preprocess=true")
    stats = preprocess_and_analyze_api_dataset(
        raw_path,
        output_file=cleaned_path,
        verbose=verbose,
    )

    artifact_prefix = f"{index_name}_"
    if verbose:
        print(f"{index_name}_features=true")
    feature_outputs = run_feature_engineering(
        cleaned_path,
        output_dir=features_dir,
        features_filename=f"{artifact_prefix}features.csv",
        metadata_filename=f"{artifact_prefix}metadata.csv",
        feature_columns_filename=f"{artifact_prefix}feature_columns.txt",
        label_encoder_filename=f"{artifact_prefix}label_encoder_process.pkl",
        path_rank_filename=f"{artifact_prefix}path_rank_mapping.pkl",
        verbose=verbose,
    )

    features_path = Path(feature_outputs["artifacts"]["features_path"])
    feature_columns_path = Path(
        feature_outputs["artifacts"]["feature_columns_path"]
    )

    return DatasetArtifacts(
        index_name=index_name,
        raw_path=raw_path,
        cleaned_path=cleaned_path,
        features_path=features_path,
        feature_columns_path=feature_columns_path,
        stats=stats,
    )


def compute_raw_comparison(
    artifacts_a: DatasetArtifacts, artifacts_b: DatasetArtifacts
) -> Dict:
    stats_a = artifacts_a.stats
    stats_b = artifacts_b.stats

    paths_a = {path for path, _ in stats_a.get("paths", [])}
    paths_b = {path for path, _ in stats_b.get("paths", [])}
    ips_a = {ip for ip, _ in stats_a.get("source_ips", [])}
    ips_b = {ip for ip, _ in stats_b.get("source_ips", [])}

    path_overlap = paths_a & paths_b
    ip_overlap = ips_a & ips_b

    def overlap_summary(a: set, b: set, label: str) -> Dict:
        union = a | b
        return {
            "count_a": len(a),
            "count_b": len(b),
            "overlap": len(a & b),
            "jaccard": (len(a & b) / len(union)) if union else 1.0,
            "only_in_a": sorted(a - b)[:10],
            "only_in_b": sorted(b - a)[:10],
        }

    return {
        "records": {
            "count_a": stats_a["final_count"],
            "count_b": stats_b["final_count"],
            "retention_a_pct": stats_a["retention_rate"],
            "retention_b_pct": stats_b["retention_rate"],
        },
        "paths": overlap_summary(paths_a, paths_b, "paths"),
        "source_ips": overlap_summary(ips_a, ips_b, "source_ips"),
    }


def compute_feature_comparison(
    artifacts_a: DatasetArtifacts, artifacts_b: DatasetArtifacts
) -> Dict:
    df_a = pd.read_csv(artifacts_a.features_path)
    df_b = pd.read_csv(artifacts_b.features_path)

    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    common_cols = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    summary = {
        "columns": {
            "common": common_cols,
            "only_in_a": only_a,
            "only_in_b": only_b,
        },
        "shape_a": df_a.shape,
        "shape_b": df_b.shape,
    }

    if common_cols:
        stats = pd.DataFrame(
            {
                "mean_a": df_a[common_cols].mean(),
                "mean_b": df_b[common_cols].mean(),
                "std_a": df_a[common_cols].std(),
                "std_b": df_b[common_cols].std(),
                "mean_diff": (df_a[common_cols].mean() - df_b[common_cols].mean()),
                "abs_mean_diff": (df_a[common_cols].mean() - df_b[common_cols].mean())
                .abs(),
            }
        )
        stats["mean_ratio"] = stats.apply(
            lambda row: (row.mean_a / row.mean_b)
            if row.mean_b not in (0, np.nan) else np.nan,
            axis=1,
        )

        mean_vector_a = stats["mean_a"].to_numpy()
        mean_vector_b = stats["mean_b"].to_numpy()

        similarity_metrics = {}
        if len(common_cols) >= 2:
            corr = np.corrcoef(mean_vector_a, mean_vector_b)[0, 1]
            similarity_metrics["mean_vector_correlation"] = float(corr)
        else:
            similarity_metrics["mean_vector_correlation"] = None

        euclidean = float(np.linalg.norm(mean_vector_a - mean_vector_b))
        cosine = float(
            np.dot(mean_vector_a, mean_vector_b)
            / (
                np.linalg.norm(mean_vector_a) * np.linalg.norm(mean_vector_b)
                + 1e-12
            )
        )

        similarity_metrics["mean_vector_euclidean"] = euclidean
        similarity_metrics["mean_vector_cosine"] = cosine

        summary["metrics"] = similarity_metrics
        summary["column_stats"] = stats.sort_values(
            "abs_mean_diff", ascending=False
        ).head(20)

    return summary


def render_feature_stats(stats: pd.DataFrame) -> str:
    if stats.empty:
        return "No overlapping feature statistics available."
    table = stats.copy()
    table.index.name = "feature"
    return table.to_string(
        float_format=lambda x: f"{x:8.4f}",
        justify="left",
        col_space=12,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download two OpenSearch indices, preprocess them, and compare their "
            "raw and feature-level characteristics."
        ),
        epilog=(
            "You can select indices by name (--index-a/--index-b) or by serial number "
            "from the download_indexWise listing (--serial-a/--serial-b). "
            "Use --report to persist the comparison as JSON."
        ),
    )
    parser.add_argument(
        "--index-a",
        help="Exact name of the first index to compare.",
    )
    parser.add_argument(
        "--index-b",
        help="Exact name of the second index to compare.",
    )
    parser.add_argument(
        "--serial-a",
        type=int,
        help="Serial number of the first index (from download_indexWise listing).",
    )
    parser.add_argument(
        "--serial-b",
        type=int,
        help="Serial number of the second index (from download_indexWise listing).",
    )
    parser.add_argument(
        "--es-url",
        default=ES_URL,
        help="OpenSearch endpoint URL.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to store downloaded and processed data.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download raw datasets even if cached versions exist.",
    )
    parser.add_argument(
        "--report",
        help="Optional path to write JSON comparison report.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress output.",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    indices = fetch_user_indices(args.es_url)

    index_a = select_index(
        indices=indices,
        index_name=args.index_a,
        index_serial=args.serial_a,
        label="A",
        verbose=verbose,
    )
    index_b = select_index(
        indices=indices,
        index_name=args.index_b,
        index_serial=args.serial_b,
        label="B",
        verbose=verbose,
    )

    if index_a == index_b:
        print("warning=same_index", file=sys.stderr)

    data_dir = Path(args.data_dir)

    artifacts_a = ensure_dataset(
        index_name=index_a,
        es_url=args.es_url,
        data_dir=data_dir,
        force_download=args.force_download,
        verbose=verbose,
    )
    artifacts_b = ensure_dataset(
        index_name=index_b,
        es_url=args.es_url,
        data_dir=data_dir,
        force_download=args.force_download,
        verbose=verbose,
    )

    raw_comparison = compute_raw_comparison(artifacts_a, artifacts_b)
    if verbose:
        print("raw_comparison=" + json.dumps(raw_comparison))

    feature_comparison = compute_feature_comparison(artifacts_a, artifacts_b)
    if verbose:
        print(
            "feature_summary="
            + json.dumps(
                {
                    "columns": feature_comparison.get("columns"),
                    "shape_a": feature_comparison.get("shape_a"),
                    "shape_b": feature_comparison.get("shape_b"),
                    "metrics": feature_comparison.get("metrics"),
                }
            )
        )
        if "column_stats" in feature_comparison:
            print("feature_discrepancies=")
            print(render_feature_stats(feature_comparison["column_stats"]))

    report = {
        "index_a": artifacts_a.index_name,
        "index_b": artifacts_b.index_name,
        "raw": raw_comparison,
        "features": feature_comparison,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"report_path={report_path}")


if __name__ == "__main__":
    main()

