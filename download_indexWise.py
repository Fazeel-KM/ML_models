import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from opensearchpy import OpenSearch, helpers


ES_URL = "http://3.27.173.116:9200"


def _format_indices(indices: Sequence[dict]) -> None:
    print("\nAvailable Indices")
    print("-" * 72)
    print(f"{'No.':<4} {'Index Name':<38} {'Docs':>10} {'Size':>12}")
    print("-" * 72)
    for i, idx in enumerate(indices, 1):
        print(
            f"{i:<4} {idx['index']:<38} "
            f"{idx['docs.count']:>10} {idx['store.size']:>12}"
        )
    print("-" * 72)


def _prompt_for_index(indices: Sequence[dict]) -> Optional[str]:
    while True:
        try:
            choice = input("Enter serial number (or 'q' to quit): ").strip()
            if choice.lower() == "q":
                return None
            choice_num = int(choice)
            if 1 <= choice_num <= len(indices):
                return indices[choice_num - 1]["index"]
            print(f"Enter 1-{len(indices)} or 'q'.")
        except ValueError:
            print("Enter a number or 'q'.")


def _ensure_index_exists(index_name: str, indices: Sequence[dict]) -> None:
    known_indices = {idx["index"] for idx in indices}
    if index_name not in known_indices:
        raise ValueError(
            f"Index '{index_name}' not found. Available indices: {', '.join(sorted(known_indices))}"
        )


def list_user_indices(es: OpenSearch) -> List[dict]:
    indices_response = es.cat.indices(format="json")
    user_indices = [idx for idx in indices_response if not idx["index"].startswith(".")]
    return sorted(user_indices, key=lambda idx: idx["index"])


def download_index(
    index_name: Optional[str] = None,
    output_dir: Path | str = ".",
    es_url: str = ES_URL,
    interactive: bool = True,
) -> Tuple[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    es = OpenSearch(es_url)
    user_indices = list_user_indices(es)

    if not user_indices:
        raise RuntimeError("No non-system indices found in OpenSearch.")

    if index_name is None:
        if not interactive:
            raise ValueError("Index name must be provided when running in non-interactive mode.")
        _format_indices(user_indices)
        index_name = _prompt_for_index(user_indices)
        if index_name is None:
            raise SystemExit(0)
    else:
        _ensure_index_exists(index_name, user_indices)
        if interactive:
            _format_indices(user_indices)
            print(f"selected={index_name}")

    count_resp = es.count(index=index_name)
    total = count_resp["count"]
    print(f"docs={total}")

    output_path = output_dir / f"{index_name}.json"
    docs_iter = helpers.scan(
        client=es,
        index=index_name,
        query={"query": {"match_all": {}}},
        preserve_order=True,
        size=1000,
    )

    saved = 0
    print(f"output={output_path}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for doc in docs_iter:
            if not first:
                f.write(",\n")
            obj = json.dumps(doc.get("_source", {}), ensure_ascii=False, indent=2)
            f.write("  " + obj.replace("\n", "\n  "))
            first = False
            saved += 1
        f.write("\n]\n")

    print(f"saved={saved},path={output_path}")
    return index_name, output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download an OpenSearch index to JSON.")
    parser.add_argument("--index", help="Exact name of the OpenSearch index to download.")
    parser.add_argument("--output-dir", default=".", help="Directory to store the downloaded JSON file.")
    parser.add_argument("--es-url", default=ES_URL, help="OpenSearch endpoint URL.")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompts (requires --index).",
    )

    args = parser.parse_args()
    try:
        index_name, output_path = download_index(
            index_name=args.index,
            output_dir=args.output_dir,
            es_url=args.es_url,
            interactive=not args.non_interactive,
        )
        print(f"index={index_name},path={output_path}")
    except SystemExit:
        pass
    except Exception as exc:
        print(f"error={exc}")


if __name__ == "__main__":
    main()