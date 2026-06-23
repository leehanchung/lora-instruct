"""Build a retrieval index from a corpus (JSONL of {id, contents}).

Separating build (`build_*`) from serve (`server.py`) keeps RL rollouts cheap and
reproducible: build a self-hosted index once, serve it for all training/eval runs
instead of hitting a rate-limited live API. (Pattern from QUEST/Search-R1.)
"""

from __future__ import annotations

import argparse


def build(corpus_path: str, out_dir: str, backend: str = "bm25") -> None:
    # TODO: load corpus.jsonl, build the chosen index, persist to out_dir.
    raise NotImplementedError


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="path to corpus.jsonl")
    p.add_argument("--out", required=True, help="output index dir")
    p.add_argument("--backend", default="bm25", choices=["bm25", "dense"])
    args = p.parse_args()
    build(args.corpus, args.out, args.backend)


if __name__ == "__main__":
    main()
