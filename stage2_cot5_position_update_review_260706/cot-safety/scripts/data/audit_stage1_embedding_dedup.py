#!/usr/bin/env python3
"""Cross-source prompt-neighbor audit for Stage 1 pair freeze.

The lexical freeze audit can report zero duplicate edges.  This companion audit
checks whether that zero is plausible by computing prompt-vector cosine
similarities, writing a histogram, the 0.80-0.90 near-threshold band count, and
the top cross-source nearest-neighbor pairs.

By default, top-neighbor outputs include hashes and pair/source IDs only.  Use
``--include-text`` only for internal manual review files; stdout and summary
JSON never include raw prompts or trajectories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPT_DIR))

from cot_safety.utils.io import write_json, write_jsonl

import audit_stage1_pair_freeze as freeze_audit


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:n]


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").replace("\r\n", "\n").replace("\r", "\n").split())


def load_pair_records(paths: list[Path]) -> tuple[list[freeze_audit.PairRecord], list[dict[str, Any]], dict[str, Any]]:
    pairs, rejected, input_stats = freeze_audit.load_pairs(paths, tolerate_partial_tail=False)
    kept_indices = {
        idx for idx, pair in enumerate(pairs)
        if pair.prompt and pair.source_family not in {"", "unknown", "unknown_natural"} and "+" not in pair.source_family
    }
    kept = [pair for idx, pair in enumerate(pairs) if idx in kept_indices]
    extra_rejected = []
    for idx, pair in enumerate(pairs):
        if idx in kept_indices:
            continue
        extra_rejected.append(
            {
                "pair_id": pair.pair_id,
                "source_family": pair.source_family,
                "drop_reason": "missing_prompt_unknown_or_ambiguous_source",
            }
        )
    return kept, rejected + extra_rejected, input_stats


def l2_normalize_rows(matrix: Any) -> Any:
    import numpy as np

    arr = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    arr = arr.astype("float32", copy=False)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return arr / norms


def is_sparse_matrix(matrix: Any) -> bool:
    try:
        from scipy import sparse  # type: ignore

        return bool(sparse.issparse(matrix))
    except Exception:
        return False


def char_ngrams(text: str, *, min_n: int = 3, max_n: int = 5) -> list[str]:
    text = f" {normalize_space(text).lower()} "
    grams: list[str] = []
    for n in range(min_n, max_n + 1):
        if len(text) < n:
            continue
        grams.extend(text[i : i + n] for i in range(len(text) - n + 1))
    return grams


def hashed_char_tfidf(prompts: list[str], *, n_features: int = 32768) -> Any:
    import numpy as np

    matrix = np.zeros((len(prompts), n_features), dtype="float32")
    doc_freq = np.zeros(n_features, dtype="float32")
    per_doc_indices: list[set[int]] = []
    for row_idx, prompt in enumerate(prompts):
        indices = []
        for gram in char_ngrams(prompt):
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            feature_idx = int.from_bytes(digest, "little") % n_features
            indices.append(feature_idx)
            matrix[row_idx, feature_idx] += 1.0
        unique = set(indices)
        per_doc_indices.append(unique)
        for feature_idx in unique:
            doc_freq[feature_idx] += 1.0
    if prompts:
        idf = np.log((1.0 + len(prompts)) / (1.0 + doc_freq)) + 1.0
        for row_idx, feature_indices in enumerate(per_doc_indices):
            if feature_indices:
                idx = list(feature_indices)
                matrix[row_idx, idx] *= idf[idx]
    return l2_normalize_rows(matrix)


def embed_prompts(
    prompts: list[str],
    *,
    mode: str,
    model_name: str,
    batch_size: int,
    local_files_only: bool,
    allow_tfidf_fallback: bool,
) -> tuple[Any, dict[str, Any]]:
    if mode == "sentence-transformer":
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(model_name, device="cpu", local_files_only=local_files_only)
            embeddings = model.encode(
                prompts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings, {
                "mode": "sentence-transformer",
                "model_name": model_name,
                "local_files_only": local_files_only,
                "fallback_used": False,
            }
        except Exception as exc:
            if not allow_tfidf_fallback:
                raise
            mode = "tfidf"
            fallback_reason = f"{type(exc).__name__}: {exc}"
        else:  # pragma: no cover
            fallback_reason = ""
    else:
        fallback_reason = ""

    if mode == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                ngram_range=(3, 5),
                min_df=1,
                max_features=200_000,
            )
            matrix = vectorizer.fit_transform(prompts)
            embeddings = matrix
            mode_name = "tfidf-char-wb-3-5"
            n_features = int(matrix.shape[1])
            sklearn_available = True
        except Exception as exc:
            if not allow_tfidf_fallback:
                raise
            embeddings = hashed_char_tfidf(prompts)
            mode_name = "hashed-char-tfidf-3-5"
            n_features = int(embeddings.shape[1])
            sklearn_available = False
            fallback_reason = fallback_reason or f"{type(exc).__name__}: {exc}"
        return embeddings, {
            "mode": mode_name,
            "model_name": None,
            "local_files_only": None,
            "fallback_used": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "sklearn_available": sklearn_available,
            "n_features": n_features,
        }
    raise ValueError(f"unsupported embedding mode: {mode}")


def cosine_chunks(embeddings: Any, *, chunk_size: int) -> Iterable[tuple[int, Any]]:
    import numpy as np

    if is_sparse_matrix(embeddings):
        for start in range(0, embeddings.shape[0], chunk_size):
            yield start, (embeddings[start : start + chunk_size] @ embeddings.T).toarray()
        return
    arr = embeddings.toarray() if hasattr(embeddings, "toarray") else np.asarray(embeddings)
    for start in range(0, arr.shape[0], chunk_size):
        yield start, arr[start : start + chunk_size] @ arr.T


def bin_label(score: float, *, bin_width: float) -> str:
    eps = 1e-12
    if score >= 1.0:
        low = 1.0 - bin_width
        high = 1.0
    else:
        low = math.floor((score + eps) / bin_width) * bin_width
        high = low + bin_width
    return f"{low:.2f}-{high:.2f}"


def bin_sort_key(label: str) -> float:
    try:
        return float(label.rsplit("-", 1)[0])
    except Exception:
        return 0.0


def can_enter_top(top: list[dict[str, Any]], *, cosine: float, top_k: int) -> bool:
    return len(top) < top_k or cosine > top[-1]["cosine"]


def update_top(top: list[dict[str, Any]], item: dict[str, Any], *, top_k: int) -> None:
    top.append(item)
    top.sort(key=lambda row: row["cosine"], reverse=True)
    if len(top) > top_k:
        del top[top_k:]


def pair_payload(
    *,
    a: freeze_audit.PairRecord,
    b: freeze_audit.PairRecord,
    cosine: float,
    include_text: bool,
    max_text_chars: int,
) -> dict[str, Any]:
    row = {
        "cosine": cosine,
        "a_pair_id": a.pair_id,
        "b_pair_id": b.pair_id,
        "a_prompt_instance_id": a.prompt_instance_id,
        "b_prompt_instance_id": b.prompt_instance_id,
        "a_source_family": a.source_family,
        "b_source_family": b.source_family,
        "a_prompt_hash": a.prompt_norm_hash,
        "b_prompt_hash": b.prompt_norm_hash,
    }
    if include_text:
        row.update(
            {
                "a_prompt": normalize_space(a.prompt)[:max_text_chars],
                "b_prompt": normalize_space(b.prompt)[:max_text_chars],
            }
        )
    return row


def write_histogram(path: Path, histogram: Counter[str]) -> None:
    rows = [{"bin": key, "count": histogram[key]} for key in sorted(histogram, key=bin_sort_key)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("bin\tcount\n")
        for row in rows:
            handle.write(f"{row['bin']}\t{row['count']}\n")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    input_paths = [Path(path) for path in args.input_jsonl]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs, rejected, input_stats = load_pair_records(input_paths)
    prompts = [pair.prompt for pair in pairs]
    embeddings, embedding_info = embed_prompts(
        prompts,
        mode=args.embedding_mode,
        model_name=args.embedding_model,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
        allow_tfidf_fallback=args.allow_tfidf_fallback,
    )

    cross_hist = Counter()
    all_hist = Counter()
    source_pair_counts = Counter()
    near_band_count = 0
    threshold_edges_count = 0
    top_cross: list[dict[str, Any]] = []
    top_near_band: list[dict[str, Any]] = []
    n_all_pairs = 0
    n_cross_pairs = 0

    for start, sims in cosine_chunks(embeddings, chunk_size=args.chunk_size):
        for local_i in range(sims.shape[0]):
            i = start + local_i
            for j in range(i + 1, sims.shape[1]):
                cosine = float(sims[local_i, j])
                n_all_pairs += 1
                all_hist[bin_label(cosine, bin_width=args.bin_width)] += 1
                if pairs[i].source_family == pairs[j].source_family:
                    continue
                n_cross_pairs += 1
                cross_hist[bin_label(cosine, bin_width=args.bin_width)] += 1
                source_key = "||".join(sorted((pairs[i].source_family, pairs[j].source_family)))
                source_pair_counts[source_key] += 1
                payload: dict[str, Any] | None = None
                if can_enter_top(top_cross, cosine=cosine, top_k=args.top_k):
                    payload = pair_payload(
                        a=pairs[i],
                        b=pairs[j],
                        cosine=cosine,
                        include_text=args.include_text,
                        max_text_chars=args.max_text_chars,
                    )
                    update_top(top_cross, payload, top_k=args.top_k)
                if args.near_band_low <= cosine < args.near_band_high:
                    near_band_count += 1
                    if can_enter_top(top_near_band, cosine=cosine, top_k=args.top_k):
                        if payload is None:
                            payload = pair_payload(
                                a=pairs[i],
                                b=pairs[j],
                                cosine=cosine,
                                include_text=args.include_text,
                                max_text_chars=args.max_text_chars,
                            )
                        update_top(top_near_band, payload, top_k=args.top_k)
                if cosine >= args.threshold:
                    threshold_edges_count += 1

    summary = {
        "stage": "stage1_embedding_dedup_audit",
        "input_paths": [str(path) for path in input_paths],
        "input_stats": input_stats,
        "n_pairs_loaded": len(pairs),
        "n_rejected_during_load": len(rejected),
        "pairs_by_source": dict(Counter(pair.source_family for pair in pairs)),
        "embedding": embedding_info,
        "n_all_pairwise_comparisons": n_all_pairs,
        "n_cross_source_pairwise_comparisons": n_cross_pairs,
        "cross_source_pairs_by_source_pair": dict(sorted(source_pair_counts.items())),
        "threshold": args.threshold,
        "n_cross_source_edges_at_or_above_threshold": threshold_edges_count,
        "cross_source_near_band": {"low": args.near_band_low, "high": args.near_band_high, "count": near_band_count},
        "histogram_bin_width": args.bin_width,
        "outputs": {
            "summary": str(output_dir / "embedding_dedup_summary.json"),
            "cross_source_histogram_tsv": str(output_dir / "cross_source_similarity_histogram.tsv"),
            "all_pairs_histogram_tsv": str(output_dir / "all_pairs_similarity_histogram.tsv"),
            "top_cross_source_neighbors": str(output_dir / "top_cross_source_neighbors.jsonl"),
            "top_near_band_cross_source_neighbors": str(output_dir / "top_near_band_cross_source_neighbors.jsonl"),
            "load_rejections": str(output_dir / "load_rejections.jsonl") if rejected else None,
        },
    }
    write_json(output_dir / "embedding_dedup_summary.json", summary)
    write_histogram(output_dir / "cross_source_similarity_histogram.tsv", cross_hist)
    write_histogram(output_dir / "all_pairs_similarity_histogram.tsv", all_hist)
    write_jsonl(output_dir / "top_cross_source_neighbors.jsonl", top_cross)
    write_jsonl(output_dir / "top_near_band_cross_source_neighbors.jsonl", top_near_band)
    if rejected:
        write_jsonl(output_dir / "load_rejections.jsonl", rejected)
    print(json.dumps({
        "n_pairs_loaded": summary["n_pairs_loaded"],
        "pairs_by_source": summary["pairs_by_source"],
        "embedding_mode": summary["embedding"]["mode"],
        "n_cross_source_edges_at_or_above_threshold": threshold_edges_count,
        "cross_source_near_band": summary["cross_source_near_band"],
    }, indent=2, ensure_ascii=False))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--embedding-mode", choices=["sentence-transformer", "tfidf"], default="sentence-transformer")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-tfidf-fallback", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--near-band-low", type=float, default=0.80)
    parser.add_argument("--near-band-high", type=float, default=0.90)
    parser.add_argument("--bin-width", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--include-text", action="store_true")
    parser.add_argument("--max-text-chars", type=int, default=600)
    args = parser.parse_args()
    if args.near_band_low >= args.near_band_high:
        parser.error("--near-band-low must be less than --near-band-high")
    if args.bin_width <= 0 or args.bin_width > 1:
        parser.error("--bin-width must be in (0, 1]")
    return args


def main() -> int:
    run_audit(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
