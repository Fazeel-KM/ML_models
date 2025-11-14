import argparse
import json
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def run_feature_engineering(
    cleaned_data_path: str | Path,
    *,
    output_dir: str | Path = ".",
    features_filename: str = "features_engineered_2.csv",
    metadata_filename: str = "metadata.csv",
    feature_columns_filename: str = "feature_columns.txt",
    label_encoder_filename: str = "label_encoder_process.pkl",
    path_rank_filename: str = "path_rank_mapping.pkl",
    top_paths: int = 15,
    request_rate_window: str = "1m",
    rolling_window: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    input_path = Path(cleaned_data_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features_path = output_path / features_filename
    metadata_path = output_path / metadata_filename
    feature_columns_path = output_path / feature_columns_filename
    label_encoder_path = output_path / label_encoder_filename
    path_rank_path = output_path / path_rank_filename

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)


    def safe_extract(record: Dict[str, Any], path: str) -> Any:
        keys = path.split(".")
        value: Any = record
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value

    records = []
    for record in data:
        records.append(
            {
                "timestamp": safe_extract(record, "@timestamp"),
                "http_method": safe_extract(record, "http_request.method"),
                "http_path": safe_extract(record, "http_request.path"),
                "http_direction": safe_extract(record, "http_request.direction"),
                "body_bytes": safe_extract(record, "http_response.body_bytes"),
                "src_ip": safe_extract(record, "src_endpoint.ip"),
                "actor_process": safe_extract(record, "actor.process"),
            }
        )

    df = pd.DataFrame(records)

    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_dt").reset_index(drop=True)

    time_start = df["timestamp_dt"].min()
    time_end = df["timestamp_dt"].max()
    time_span = time_end - time_start

    df["inter_arrival_time"] = df["timestamp_dt"].diff().dt.total_seconds()
    df["inter_arrival_time"].fillna(0, inplace=True)

    df["request_rate_1min"] = 0.0
    window = np.timedelta64(int(request_rate_window[:-1]), request_rate_window[-1])
    for ip in df["src_ip"].dropna().unique():
        ip_mask = df["src_ip"] == ip
        ip_times = df.loc[ip_mask, "timestamp_dt"].values
        counts = []
        for curr_time in ip_times:
            count = np.sum((ip_times >= curr_time - window) & (ip_times <= curr_time))
            counts.append(count)
        df.loc[ip_mask, "request_rate_1min"] = counts

    top_paths_list = df["http_path"].value_counts().head(top_paths).index.tolist()
    path_rank_map = {path: idx + 1 for idx, path in enumerate(top_paths_list)}

    def map_path_rank(path: Optional[str]) -> int:
        return path_rank_map.get(path, top_paths + 1)

    df["path_frequency_rank"] = df["http_path"].apply(map_path_rank)

    unique_paths_per_ip = (
        df.groupby("src_ip")["http_path"].nunique().reset_index()
    )
    unique_paths_per_ip.columns = ["src_ip", "unique_paths_per_ip"]
    df = df.merge(unique_paths_per_ip, on="src_ip", how="left")

    df["body_bytes"].fillna(0, inplace=True)

    df["body_bytes_rolling_mean"] = df.groupby("src_ip")["body_bytes"].transform(
        lambda x: x.rolling(window=rolling_window, min_periods=1).mean()
    )

    # FIXED (always same columns)
    ALL_HTTP_METHODS = ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'PATCH', 'OPTIONS']
    method_cols = []
    for method in ALL_HTTP_METHODS:
        col_name = f'method_{method}'
        df[col_name] = (df['http_method'] == method).astype(int)
        method_cols.append(col_name)


    df["direction_encoded"] = (df["http_direction"] == "outbound").astype(int)

    le_process = LabelEncoder()
    df["actor_process"].fillna("unknown", inplace=True)
    df["process_encoded"] = le_process.fit_transform(df["actor_process"])

    feature_columns = [
        "inter_arrival_time",
        "request_rate_1min",
        "path_frequency_rank",
        "unique_paths_per_ip",
        "body_bytes",
        "body_bytes_rolling_mean",
        "direction_encoded",
        "process_encoded",
    ]
    feature_columns.extend(method_cols)
    X = df[feature_columns].copy()
    missing_counts = X.isnull().sum()
    if missing_counts.sum() == 0:
        missing_message = "No missing values."
    else:
        missing_message = "Missing values handled."
    X.fillna(0, inplace=True)

    inf_counts = np.isinf(X).sum()
    if inf_counts.sum() == 0:
        inf_message = "No infinite values."
    else:
        inf_message = "Infinite values replaced."
        X.replace([np.inf, -np.inf], 0, inplace=True)

    corr_matrix = X.corr()
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.85:
                high_corr_pairs.append(
                    (
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j],
                    )
                )

    X.to_csv(features_path, index=False)

    with feature_columns_path.open("w", encoding="utf-8") as f:
        for col in X.columns:
            f.write(f"{col}\n")

    metadata = df[["timestamp_dt", "src_ip", "http_path", "http_method"]].copy()
    metadata.to_csv(metadata_path, index=False)

    with label_encoder_path.open("wb") as f:
        pickle.dump(le_process, f)

    with path_rank_path.open("wb") as f:
        pickle.dump(path_rank_map, f)

    if verbose:
        print(f"records={len(data)}")
        print(f"features_shape={X.shape[0]}x{X.shape[1]}")
        print(f"features_path={features_path}")
        print(f"metadata_path={metadata_path}")
        print(f"feature_columns_path={feature_columns_path}")
        print(f"label_encoder_path={label_encoder_path}")
        print(f"path_rank_path={path_rank_path}")

    return {
        "feature_columns": list(X.columns),
        "feature_shape": X.shape,
        "high_corr_pairs": high_corr_pairs,
        "artifacts": {
            "features_path": features_path,
            "metadata_path": metadata_path,
            "feature_columns_path": feature_columns_path,
            "label_encoder_path": label_encoder_path,
            "path_rank_path": path_rank_path,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feature engineering on cleaned API dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to cleaned JSON events file.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to store generated features and artifacts.",
    )
    parser.add_argument(
        "--features-filename",
        default="features_engineered_2.csv",
        help="Filename for the engineered features CSV.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging.",
    )
    args = parser.parse_args()

    run_feature_engineering(
        args.input,
        output_dir=args.output_dir,
        features_filename=args.features_filename,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()