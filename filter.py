import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Optional


def preprocess_and_analyze_api_dataset(
    input_file: str | Path,
    output_file: Optional[str | Path] = None,
    *,
    verbose: bool = True,
):
    input_path = Path(input_file)
    output_path = Path(output_file) if output_file is not None else None

    with input_path.open('r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    cleaned_data = [r for r in data if r.get('http_request', {}).get('path')]
    
    final_count = len(cleaned_data)
    removed_count = original_count - final_count
    retention_rate = (final_count / original_count * 100) if original_count else 0.0

    paths = [r['http_request'].get('path') for r in cleaned_data]
    unique_paths_count = len(set(paths))
    
    path_counts = Counter(paths)
    
    source_ips = [
        r.get('src_endpoint', {}).get('ip')
        for r in cleaned_data
        if r.get('src_endpoint', {}).get('ip')
    ]
    unique_source_ips_count = len(set(source_ips))
    source_ip_counts = Counter(source_ips)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2)
        if verbose:
            print(f"cleaned_path={output_path}")
    if verbose:
        print(f"retained={final_count}")
        print(f"removed={removed_count}")
        print(f"unique_paths={unique_paths_count}")
        print(f"unique_source_ips={unique_source_ips_count}")

    return {
        'original_count': original_count,
        'final_count': final_count,
        'removed_count': removed_count,
        'retention_rate': retention_rate,
        'unique_paths': unique_paths_count,
        'unique_source_ips': unique_source_ips_count,
        'paths': sorted(path_counts.items(), key=lambda x: x[1], reverse=True),
        'source_ips': sorted(source_ip_counts.items(), key=lambda x: x[1], reverse=True),
        'cleaned_data': cleaned_data
    }

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess API dataset by removing records with empty paths."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file containing raw events.",
    )
    parser.add_argument(
        "--output",
        help="Path to save the cleaned JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed logging output.",
    )
    args = parser.parse_args()

    stats = preprocess_and_analyze_api_dataset(
        args.input,
        output_file=args.output,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print(f"retained={stats['final_count']}")
        print(f"unique_paths={stats['unique_paths']}")
        print(f"unique_source_ips={stats['unique_source_ips']}")
        print(f"retention_rate={stats['retention_rate']:.1f}")


if __name__ == "__main__":
    main()