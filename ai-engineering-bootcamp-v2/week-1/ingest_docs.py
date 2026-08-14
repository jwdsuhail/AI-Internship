"""
Batch-ingest every .txt in docs/ through the running API.

Deliberately goes over HTTP rather than importing main.py: it exercises the same /ingest
endpoint a real client would, so it works unchanged against localhost or the Render URL.

    python ingest_docs.py                                  # local
    python ingest_docs.py --api-url https://x.onrender.com  # live
    python ingest_docs.py --clear                           # wipe the index first
"""

import argparse
import sys
from pathlib import Path

import httpx

DOCS_DIR = Path(__file__).resolve().parent / "docs"


def clear_index() -> None:
    """Delete every vector in the index — used to drop throwaway test data before a real run."""

    import os

    from dotenv import load_dotenv
    from pinecone import Pinecone

    load_dotenv(Path(__file__).resolve().parent / ".env")
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(os.environ["PINECONE_INDEX_NAME"])
    try:
        index.delete(delete_all=True)
        print("Cleared all vectors from the index.\n")
    except Exception as exc:
        # A brand-new or already-empty index raises rather than no-opping; that is fine.
        print(f"Nothing to clear ({exc}).\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--docs-dir", default=str(DOCS_DIR))
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--clear", action="store_true", help="wipe the index before ingesting")
    args = parser.parse_args()

    if args.clear:
        clear_index()

    docs_dir = Path(args.docs_dir)
    files = sorted(docs_dir.glob("*.txt"))
    if not files:
        print(f"No .txt files found in {docs_dir}")
        return 1

    base = args.api_url.rstrip("/")
    total_chunks = 0

    for path in files:
        document_id = path.stem  # "employee_handbook.txt" -> "employee_handbook"
        payload = {
            "text": path.read_text(encoding="utf-8"),
            "document_id": document_id,
            "source": path.name,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
        }

        try:
            response = httpx.post(f"{base}/ingest", json=payload, timeout=180.0)
        except httpx.RequestError as exc:
            print(f"  {document_id:<24} FAILED — cannot reach {base} ({exc})")
            return 1

        if response.status_code != 200:
            print(f"  {document_id:<24} FAILED — {response.status_code} {response.text}")
            continue

        chunks = response.json()["chunks_indexed"]
        total_chunks += chunks
        print(f"  {document_id:<24} {chunks:>3} chunks   ({len(payload['text']):,} chars)")

    print(f"\n  {'TOTAL':<24} {total_chunks:>3} chunks from {len(files)} documents")

    # Confirm against the index itself, not just our own running total.
    try:
        stats = httpx.get(f"{base}/debug/vectorstore", timeout=60.0).json()
        print(f"  Index '{stats['index']}' now holds {stats['total_vectors']} vectors.")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
