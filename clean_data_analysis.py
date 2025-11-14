import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_extract(record: dict, path: str):
    keys = path.split(".")
    value = record
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    return value


def normalize_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.total_seconds()
    return value


def value_counts(series: pd.Series, total: int, *, limit: int | None = None):
    counts = series.value_counts().head(limit) if limit else series.value_counts()
    items = []
    for val, count in counts.items():
        entry = {
            "value": val,
            "count": int(count),
        }
        if total:
            entry["pct"] = round((count / total) * 100, 2)
        items.append(entry)
    return items


def main() -> None:
    dataset_path = Path("events_cleaned.json")
    if not dataset_path.exists():
        print(json.dumps({"records": 0, "error": "events_cleaned.json not found"}))
        return

    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_records = len(data)
    report: dict[str, object] = {"records": total_records, "source": str(dataset_path)}

    if total_records == 0:
        print(json.dumps(report))
        return

    fields_to_check = {
        "timestamp": "@timestamp",
        "class_name": "class_name",
        "http_method": "http_request.method",
        "http_path": "http_request.path",
        "http_url": "http_request.url",
        "http_direction": "http_request.direction",
        "user_agent": "http_request.user_agent",
        "status_code": "http_response.status_code",
        "latency": "http_response.latency",
        "body_bytes": "http_response.body_bytes",
        "duration": "duration",
        "src_ip": "src_endpoint.ip",
        "src_port": "src_endpoint.port",
        "actor_process": "actor.process",
        "actor_user": "actor.user",
        "protocol": "api.x_protocol",
    }

    field_data = {
        field_name: [safe_extract(record, field_path) for record in data]
        for field_name, field_path in fields_to_check.items()
    }
    df = pd.DataFrame(field_data)

    completeness = []
    for col in df.columns:
        series = df[col]
        non_null = int(series.notna().sum())
        null_count = int(series.isna().sum())
        null_pct = round((null_count / total_records) * 100, 2) if total_records else 0.0
        unique = int(series.nunique(dropna=True))
        completeness.append(
            {
                "field": col,
                "non_null": non_null,
                "null": null_count,
                "null_pct": null_pct,
                "unique": unique,
            }
        )
    report["field_completeness"] = completeness

    numeric_fields = ["status_code", "latency", "body_bytes", "duration", "src_port"]
    zero_values = []
    for field in numeric_fields:
        if field in df.columns:
            zero_count = int((df[field] == 0).sum())
            non_zero = int((df[field] != 0).sum())
            zero_pct = round((zero_count / total_records) * 100, 2) if total_records else 0.0
            zero_values.append(
                {
                    "field": field,
                    "zero_count": zero_count,
                    "zero_pct": zero_pct,
                    "non_zero_count": non_zero,
                }
            )
    report["zero_values"] = zero_values

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        stats = df[numeric_cols].describe().T
        stats["missing"] = df[numeric_cols].isna().sum()
        stats["zeros"] = (df[numeric_cols] == 0).sum()
        numeric_summary = {}
        for col in stats.index:
            row = stats.loc[col].to_dict()
            numeric_summary[col] = {k: normalize_value(v) for k, v in row.items()}
        report["numeric_summary"] = numeric_summary
    else:
        report["numeric_summary"] = {}

    categorical_fields = ["http_method", "http_direction", "actor_process", "protocol"]
    categorical_summary = {}
    for field in categorical_fields:
        if field in df.columns:
            categorical_summary[field] = value_counts(df[field], total_records)
    report["categorical_distribution"] = categorical_summary

    temporal_summary: dict[str, object] = {"available": False}
    if "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
        valid_timestamps = df["timestamp_dt"].dropna()
        if not valid_timestamps.empty:
            earliest = valid_timestamps.min()
            latest = valid_timestamps.max()
            span = latest - earliest
            temporal_summary.update(
                {
                    "available": True,
                    "earliest": normalize_value(earliest),
                    "latest": normalize_value(latest),
                    "span_seconds": normalize_value(span),
                }
            )

            df_sorted = df.sort_values("timestamp_dt")
            df_sorted["time_diff"] = df_sorted["timestamp_dt"].diff()
            time_diff_seconds = df_sorted["time_diff"].dt.total_seconds().dropna()
            if not time_diff_seconds.empty:
                temporal_summary["inter_arrival_seconds"] = {
                    "mean": round(float(time_diff_seconds.mean()), 2),
                    "median": round(float(time_diff_seconds.median()), 2),
                    "max": round(float(time_diff_seconds.max()), 2),
                    "min": round(float(time_diff_seconds.min()), 2),
                }
            hour_dist = df["timestamp_dt"].dt.hour.value_counts().sort_index()
            temporal_summary["hour_distribution"] = {
                str(int(hour)): int(count) for hour, count in hour_dist.items()
            }
    report["temporal"] = temporal_summary

    if "src_ip" in df.columns:
        ip_counts = df["src_ip"].value_counts()
        localhost_count = int(df["src_ip"].isin(["127.0.0.1", "::1", "localhost"]).sum())
        localhost_pct = round((localhost_count / total_records) * 100, 2) if total_records else 0.0
        report["ip_analysis"] = {
            "unique": int(df["src_ip"].nunique()),
            "top": value_counts(df["src_ip"], total_records, limit=10),
            "localhost_pct": localhost_pct,
        }

    if "http_path" in df.columns:
        report["path_analysis"] = {
            "unique": int(df["http_path"].nunique()),
            "top": value_counts(df["http_path"], total_records, limit=15),
        }

    if "user_agent" in df.columns:
        truncated = df["user_agent"].fillna("").apply(
            lambda text: text if len(text) <= 120 else text[:117] + "..."
        )
        ua_counts = truncated.value_counts().head(10)
        items = []
        for val, count in ua_counts.items():
            entry = {"value": val, "count": int(count)}
            if total_records:
                entry["pct"] = round((count / total_records) * 100, 2)
            items.append(entry)
        report["user_agent_analysis"] = {
            "unique": int(df["user_agent"].nunique()),
            "top": items,
        }

    if "body_bytes" in df.columns and df["body_bytes"].notna().any():
        body_series = df["body_bytes"].astype(float)
        variance = float(body_series.var())
        report["response_size"] = {
            "mean": float(body_series.mean()),
            "median": float(body_series.median()),
            "std": float(body_series.std()),
            "min": float(body_series.min()),
            "max": float(body_series.max()),
            "variance": variance,
            "has_variance": variance > 0,
        }

    issues = []
    recommendations = []
    if "status_code" in df.columns and (df["status_code"] == 0).all():
        issues.append("status_code_all_zero")
        recommendations.append("status_code_not_useful_for_errors")
    if "latency" in df.columns and (df["latency"] == 0).all():
        issues.append("latency_all_zero")
        recommendations.append("use_temporal_patterns_for_latency")
    if "duration" in df.columns and (df["duration"] == 0).all():
        issues.append("duration_all_zero")
        recommendations.append("derive_duration_from_timestamps")
    if "src_ip" in df.columns:
        if report.get("ip_analysis", {}).get("localhost_pct", 0) > 90:
            issues.append("ip_diversity_low")
            recommendations.append("focus_on_path_and_timing_features")
    if "body_bytes" in df.columns and report.get("response_size", {}).get("has_variance"):
        issues.append("body_bytes_has_variance")
    else:
        issues.append("body_bytes_low_variance")
    if "timestamp" in df.columns and df["timestamp"].notna().sum() == total_records:
        issues.append("timestamps_complete")

    report["issues"] = issues
    report["recommendations"] = recommendations

    feature_viability = {
        "inter_arrival_time": "viable" if "timestamp" in df.columns else "missing_timestamp",
        "request_rate_windows": "viable" if "timestamp" in df.columns else "missing_timestamp",
        "response_body_bytes": (
            "viable"
            if "body_bytes" in df.columns and report.get("response_size", {}).get("has_variance")
            else "insufficient_variance"
        ),
        "path_diversity": "viable" if "http_path" in df.columns else "missing_http_path",
        "http_method_encoding": "viable" if "http_method" in df.columns else "missing_http_method",
        "ip_cardinality": (
            "limited"
            if "src_ip" in df.columns and df["src_ip"].nunique() < 10
            else ("viable" if "src_ip" in df.columns else "missing_src_ip")
        ),
        "user_agent_diversity": "viable" if "user_agent" in df.columns else "missing_user_agent",
        "error_rate_features": (
            "not_viable"
            if "status_code" in df.columns and (df["status_code"] == 0).all()
            else ("viable" if "status_code" in df.columns else "missing_status_code")
        ),
        "latency_features": (
            "not_viable"
            if "latency" in df.columns and (df["latency"] == 0).all()
            else ("viable" if "latency" in df.columns else "missing_latency")
        ),
    }
    report["feature_viability"] = feature_viability

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
