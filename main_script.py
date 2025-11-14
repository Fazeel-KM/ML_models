import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from opensearchpy import OpenSearch

from download_indexWise import (
    ES_URL,
    _ensure_index_exists,
    _format_indices,
    _prompt_for_index,
    download_index,
    list_user_indices,
)
from feature_engineering import run_feature_engineering
from filter import preprocess_and_analyze_api_dataset
from model import train_models

MODE_CHOICES = ("train", "test")


def fetch_user_indices(es_url: str) -> List[dict]:
    es = OpenSearch(es_url)
    try:
        indices = list_user_indices(es)
    finally:
        es.transport.close()
    return indices


def _parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _prompt_for_serials(indices: List[dict], *, allow_multiple: bool) -> List[str]:
    while True:
        _format_indices(indices)
        prompt = "Enter serial number" + ("(s) (e.g. 5 or 5,6,7)" if allow_multiple else "") + " (or 'q' to quit): "
        choice = input(prompt).strip()
        if choice.lower() == "q":
            print("No indices selected. Exiting.")
            sys.exit(0)
        serials = [item.strip() for item in choice.split(",") if item.strip()]
        if not serials:
            print("Please enter at least one serial number.")
            continue
        try:
            indexes = []
            for serial_str in serials:
                serial = int(serial_str)
                if serial < 1 or serial > len(indices):
                    raise ValueError
                indexes.append(indices[serial - 1]["index"])
        except ValueError:
            print(f"Invalid input. Serial numbers must be integers between 1 and {len(indices)}.")
            continue
        if not allow_multiple and len(indexes) > 1:
            print("Multiple selections are not allowed in this mode.")
            continue
        return indexes


def select_indices(
    indices: List[dict],
    *,
    primary_index: Optional[str],
    primary_serial: Optional[int],
    index_list: Optional[str],
    serial_list: Optional[str],
    mode: str,
    combine: bool,
    verbose: bool,
) -> tuple[List[str], bool]:
    if not indices:
        raise RuntimeError("No non-system indices available in OpenSearch.")

    selected: List[str] = []
    existing = {entry["index"] for entry in indices}

    provided_names = _parse_csv(index_list)
    provided_serials = _parse_csv(serial_list)

    if primary_index:
        provided_names.append(primary_index)
    if primary_serial is not None:
        provided_serials.append(str(primary_serial))

    for name in provided_names:
        if name not in existing:
            available = ", ".join(sorted(existing))
            raise ValueError(f"Index '{name}' not found. Available: {available}")
        if name not in selected:
            selected.append(name)

    if provided_serials:
        for serial_str in provided_serials:
            try:
                serial = int(serial_str)
            except ValueError:
                raise ValueError(f"Invalid serial '{serial_str}'. Use integers only.") from None
            if serial < 1 or serial > len(indices):
                raise ValueError(
                    f"Index serial {serial} is out of range (1-{len(indices)})."
                )
            name = indices[serial - 1]["index"]
            if name not in selected:
                selected.append(name)

    if selected:
        if not combine and len(selected) > 1:
            if mode != "train":
                raise ValueError(
                    "Test mode only supports a single index at a time."
                )
            combine = True
            if verbose:
                print("  Multiple indices detected; combination enabled.")
        if verbose:
            print(f"  Selected indices: {', '.join(selected)}")
        return selected, combine

    # Fallback: interactive single selection
    allow_multiple = mode == "train"
    chosen = _prompt_for_serials(indices, allow_multiple=allow_multiple)
    if len(chosen) > 1:
        if mode != "train":
            raise ValueError("Test mode only supports a single index at a time.")
        combine = True
        if verbose:
            print(f"  Selected indices: {', '.join(chosen)} (combined)")
    else:
        if verbose:
            print(f"  Selected index: {chosen[0]}")
    return chosen, combine


def select_index(
    indices: List[dict],
    *,
    index_name: Optional[str],
    index_serial: Optional[int],
    verbose: bool,
) -> str:
    if not indices:
        raise RuntimeError("No non-system indices available in OpenSearch.")

    if index_name:
        _ensure_index_exists(index_name, indices)
        if verbose:
            _format_indices(indices)
            print(f"  Selected index: {index_name}")
        return index_name

    if index_serial is not None:
        if index_serial < 1 or index_serial > len(indices):
            raise ValueError(
                f"Index serial {index_serial} is out of range (1-{len(indices)})."
            )
        selected = indices[index_serial - 1]["index"]
        if verbose:
            _format_indices(indices)
            print(f"  Selected index: {selected}")
        return selected

    # Interactive selection
    _format_indices(indices)
    selected = _prompt_for_index(indices)
    if selected is None:
        print("No index selected. Exiting.")
        sys.exit(0)
    if verbose:
        print(f"  Selected index: {selected}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the OpenSearch anomaly pipeline in train or test mode, with optional "
            "multi-index combination for training."
        )
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        help="Pipeline mode: 'train' to retrain models, 'test' to evaluate existing ones.",
    )
    parser.add_argument(
        "--index",
        help="Exact name of the OpenSearch index to download.",
    )
    parser.add_argument(
        "--indices",
        help="Comma-separated list of index names to combine (train mode).",
    )
    parser.add_argument(
        "--index-serial",
        type=int,
        help="Serial number of the index as shown by download_indexWise.py.",
    )
    parser.add_argument(
        "--index-serials",
        help="Comma-separated list of index serial numbers to combine (train mode).",
    )
    parser.add_argument(
        "--es-url",
        default=ES_URL,
        help="OpenSearch endpoint URL.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Root directory to store raw, cleaned, and feature-engineered data.",
    )
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store trained model artifacts.",
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.01,
        help=(
            "Upper bound on the expected outlier rate. "
            "Use 0 to auto-calibrate for all-normal training data."
        ),
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Train mode only: run model evaluation after training finishes.",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="Combine multiple indices by merging cleaned datasets before feature engineering (train mode).",
    )
    parser.add_argument(
        "--eval-model-tag",
        help="Model tag to evaluate (defaults to the selected index name in train mode).",
    )
    parser.add_argument(
        "--eval-dataset",
        help="Path to engineered features CSV used for evaluation "
        "(defaults to the dataset generated in this run).",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the index even if the raw JSON already exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output from processing stages.",
    )
    args = parser.parse_args()

    verbose = not args.quiet
    indices = fetch_user_indices(args.es_url)

    mode = args.mode
    if not mode:
        if sys.stdin.isatty():
            print("\nSelect pipeline mode:")
            for i, choice in enumerate(MODE_CHOICES, 1):
                print(f"{i}. {choice}")
            while True:
                selection = input("Enter choice (number): ").strip()
                if selection.isdigit():
                    idx = int(selection)
                    if 1 <= idx <= len(MODE_CHOICES):
                        mode = MODE_CHOICES[idx - 1]
                        break
                print(f"Please choose a number between 1 and {len(MODE_CHOICES)}.")
        else:
            mode = "train"

    selected_indices, combine_enabled = select_indices(
        indices,
        primary_index=args.index,
        primary_serial=args.index_serial,
        index_list=args.indices,
        serial_list=args.index_serials,
        mode=mode,
        combine=args.combine,
        verbose=verbose,
    )

    if mode == "test" and len(selected_indices) > 1:
        raise ValueError("Test mode only supports a single index at a time.")
    if mode != "train" and (args.combine or combine_enabled):
        raise ValueError("--combine is only supported in train mode.")
    if (args.combine or combine_enabled) and len(selected_indices) < 2:
        raise ValueError("Provide two or more indices when using --combine.")

    data_root = Path(args.data_dir)
    raw_dir = data_root / "raw"
    cleaned_dir = data_root / "clean"
    features_dir = data_root / "features"
    artifacts_dir = data_root / "artifacts"

    selected_model_tag: Optional[str] = None
    if mode == "test":
        from evaluate_models import discover_model_bundles

        model_bundles = discover_model_bundles(Path(args.models_dir), artifacts_dir)
        if not model_bundles:
            raise RuntimeError(
                f"No trained model bundles found in '{args.models_dir}'. Run in train mode first."
            )
        selected_model_tag = args.eval_model_tag
        if selected_model_tag and selected_model_tag not in model_bundles:
            raise ValueError(
                f"Model tag '{selected_model_tag}' not found. Available: {', '.join(sorted(model_bundles))}"
            )
        if not selected_model_tag:
            tags = sorted(model_bundles.keys(), reverse=True)
            if sys.stdin.isatty():
                print("\nAvailable model bundles:")
                for i, tag in enumerate(tags, 1):
                    print(f"{i}. {tag}")
                while True:
                    choice = input("Select model bundle (number): ").strip()
                    if choice.isdigit():
                        idx = int(choice)
                        if 1 <= idx <= len(tags):
                            selected_model_tag = tags[idx - 1]
                            break
                    print(f"Pick a number between 1 and {len(tags)}.")
            else:
                selected_model_tag = tags[0]
                if verbose:
                    print(f"  Using latest model bundle: {selected_model_tag}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    combined_mode = (args.combine or combine_enabled) and len(selected_indices) > 1
    processed_indices = []
    raw_paths: List[Path] = []
    cleaned_paths: List[Path] = []
    combined_records = []

    if combined_mode:
        for name in selected_indices:
            if verbose:
                print(f"\nProcessing index: {name}")
            raw_path = raw_dir / f"{name}.json"
            if raw_path.exists() and not args.force_download:
                if verbose:
                    print(f"  Using cached raw data: {raw_path}")
            else:
                _, raw_path = download_index(
                    index_name=name,
                    output_dir=raw_dir,
                    es_url=args.es_url,
                    interactive=False,
                )
            raw_paths.append(raw_path)

            cleaned_path = cleaned_dir / f"{name}_cleaned.json"
            stats = preprocess_and_analyze_api_dataset(
                raw_path,
                output_file=cleaned_path,
                verbose=verbose,
            )
            cleaned_data = stats.get("cleaned_data", [])
            combined_records.extend(cleaned_data)
            stats.pop("cleaned_data", None)
            cleaned_paths.append(cleaned_path)
            processed_indices.append(name)

        combined_tag = "combined_" + "_".join(
            name.replace("/", "_").replace(" ", "_") for name in processed_indices
        )
        cleaned_path = cleaned_dir / f"{combined_tag}_cleaned.json"
        with cleaned_path.open("w", encoding="utf-8") as f:
            json.dump(combined_records, f, indent=2)
        if verbose:
            print(f"\nCombined dataset saved to: {cleaned_path}")
        cleaned_paths.append(cleaned_path)

        artifact_prefix = f"{combined_tag}_"
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
        features_path = feature_outputs["artifacts"]["features_path"]
        feature_columns_path = feature_outputs["artifacts"]["feature_columns_path"]
        metadata_path = feature_outputs["artifacts"]["metadata_path"]
        model_tag = combined_tag
        index_display = ", ".join(processed_indices)
    else:
        index_name = selected_indices[0]
        if verbose:
            print(f"\nProcessing index: {index_name}")
        raw_path = raw_dir / f"{index_name}.json"
        if raw_path.exists() and not args.force_download:
            if verbose:
                print(f"  Using cached raw data: {raw_path}")
        else:
            _, raw_path = download_index(
                index_name=index_name,
                output_dir=raw_dir,
                es_url=args.es_url,
                interactive=False,
            )
        raw_paths.append(raw_path)

        cleaned_path = cleaned_dir / f"{index_name}_cleaned.json"
        preprocess_and_analyze_api_dataset(
            raw_path,
            output_file=cleaned_path,
            verbose=verbose,
        )
        cleaned_paths.append(cleaned_path)

        artifact_prefix = f"{index_name}_"
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
        features_path = feature_outputs["artifacts"]["features_path"]
        feature_columns_path = feature_outputs["artifacts"]["feature_columns_path"]
        metadata_path = feature_outputs["artifacts"]["metadata_path"]
        model_tag = index_name
        index_display = index_name

    if mode == "train":
        train_models(
            features_path,
            model_tag=model_tag,
            models_dir=args.models_dir,
            artifacts_dir=artifacts_dir,
            contamination=args.contamination,
            verbose=verbose,
        )
        if args.evaluate:
            from evaluate_models import run_evaluation

            eval_model_tag = args.eval_model_tag or model_tag
            eval_dataset_path = args.eval_dataset or str(features_path)
            eval_interactive = (
                (args.eval_model_tag is None or args.eval_dataset is None)
                and sys.stdin.isatty()
            )

            run_evaluation(
                models_dir=args.models_dir,
                artifacts_dir=artifacts_dir,
                features_dir=features_dir,
                model_tag=eval_model_tag,
                dataset_path=eval_dataset_path,
                feature_columns_path=feature_columns_path,
                metadata_path=str(metadata_path),
                cleaned_path=str(cleaned_path),
                interactive=eval_interactive,
                verbose=verbose,
            )
            if verbose:
                print(f"\nEvaluation model: {eval_model_tag}")
    else:
        from evaluate_models import run_evaluation

        dataset_override = args.eval_dataset or str(features_path)

        run_evaluation(
            models_dir=args.models_dir,
            artifacts_dir=artifacts_dir,
            features_dir=features_dir,
            model_tag=selected_model_tag,
            dataset_path=dataset_override,
            feature_columns_path=feature_columns_path,
            metadata_path=str(metadata_path),
            cleaned_path=str(cleaned_path),
            interactive=False,
            verbose=verbose,
        )
        if verbose:
            print(f"\nEvaluation model: {selected_model_tag}")

    if verbose:
        print("\nPipeline summary")
        print("-" * 40)
        print(f"Mode             : {mode}")
        print(f"Processed indices: {index_display}")
        print("Raw data files   :")
        for path in raw_paths:
            print(f"  - {path}")
        print("Cleaned data     :")
        for path in cleaned_paths:
            print(f"  - {path}")
        print(f"Features file    : {features_path}")
        print(f"Models directory : {Path(args.models_dir).resolve()}")
    else:
        pass


if __name__ == "__main__":
    main()

