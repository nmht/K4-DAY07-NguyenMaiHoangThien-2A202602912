from __future__ import annotations

import csv
import re
import math
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# Ensure stdout uses UTF-8 to avoid UnicodeEncodeError on Windows
sys.stdout.reconfigure(encoding='utf-8')

from src import Document, EmbeddingStore, FixedSizeChunker
from src.chunking import ChunkingStrategyComparator


DATA_DIR = Path("data/dorms")


BENCHMARK_QUERIES = [
    {
        "question": "Ký túc xá Đại học Bách khoa Hà Nội có bao nhiêu dãy nhà, bao nhiêu phòng và có thể đón nhận khoảng bao nhiêu sinh viên?",
        "gold_answer": "Ký túc xá Bách Khoa có 10 dãy nhà, 435 phòng ở và có thể đón nhận khoảng 4.200 sinh viên.",
        "gold_doc_id": "hust-dormitory-overview",
        "metadata_filter": None,
    },
    {
        "question": "Cơ sở vật chất và lệ phí nhà X1, nhà X2 cho tân sinh viên K71 của HUCE là gì?",
        "gold_answer": "Nhà X1: 500.000 đồng x 5 tháng + 200.000 đồng tiền cọc = 2.700.000 đồng. Nhà X2: 750.000 đồng x 5 tháng + 200.000 đồng tiền cọc = 3.950.000 đồng.",
        "gold_doc_id": "dorm-registration",
        "metadata_filter": {"audience": "student"},
    },
    {
        "question": "PTIT bố trí bao nhiêu chỗ ở tại KTX B1, B2 và cơ sở Ngọc Trục cho sinh viên khóa 2025?",
        "gold_answer": "KTX B1 có 40 chỗ, KTX B2 có 460 chỗ, KTX Cơ sở Đào tạo Ngọc Trục có 340 chỗ.",
        "gold_doc_id": "dorm-slot",
        "metadata_filter": {"audience": "student"},
    },
    {
        "question": "Sinh viên Đại học Thương mại đăng ký ở Ký túc xá cơ sở Hà Nội theo quy trình nào và thời hạn đăng ký là khi nào?",
        "gold_answer": "Thời hạn đăng ký từ 25/08/2023 đến 30/08/2023. Sinh viên truy cập biểu mẫu Google Forms, xem đúng đối tượng ưu tiên và đăng ký theo hướng dẫn; sinh viên đủ điều kiện sẽ nhận tin nhắn xác nhận qua số điện thoại đã đăng ký.",
        "gold_doc_id": "tmu-dormitory-registration-hanoi",
        "metadata_filter": {"audience": "student"},
    },
    {
        "question": "Mức giá điện nước tại khu nội trú được ban hành ngày nào và file đính kèm tên gì?",
        "gold_answer": "Văn bản điều chỉnh mức giá điện, nước tại Khu nội trú sinh viên có ngày ban hành 05/08/2024 và file đính kèm là dieu-chinh-gia-dien-nuoc-kntpdf-1727328575.pdf.",
        "gold_doc_id": "tmu-dormitory-electric-water-fees",
        "metadata_filter": {"audience": "student"},
    },
]


class HeadingChunker:
    """Split Markdown by headings; long sections fall back to FixedSizeChunker."""

    def __init__(self, chunk_size: int = 900) -> None:
        self.chunk_size = chunk_size
        self.fallback = FixedSizeChunker(chunk_size=chunk_size)

    def chunk(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []

        parts = re.split(r"(?m)(?=^#{1,3}\s+)", text)
        chunks: list[str] = []
        for part in parts:
            section = part.strip()
            if not section:
                continue
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            lines = section.splitlines()
            heading = lines[0] if lines and lines[0].startswith("#") else ""
            body = "\n".join(lines[1:]).strip() if heading else section
            for piece in self.fallback.chunk(body):
                chunks.append(f"{heading}\n\n{piece}".strip() if heading else piece)
        return chunks


# Mỗi thành viên chỉ đổi một dòng này để so sánh chiến lược.
CHUNKER = HeadingChunker(chunk_size=900)


class LexicalHashEmbedder:
    """Tiny local lexical embedder for repeatable benchmark runs without APIs."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def __call__(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in re.findall(r"[\wÀ-ỹ]+", text.lower()):
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def parse_markdown(path: Path) -> tuple[dict[str, str], str]:
    raw = path.read_text(encoding="utf-8-sig")
    if raw.startswith("---"):
        _, frontmatter, content = raw.split("---", 2)
    else:
        frontmatter, content = "", raw

    metadata: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip('"')
    return metadata, content.strip()


def make_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.md")):
        metadata, content = parse_markdown(path)
        source_doc_id = path.stem
        for index, chunk in enumerate(CHUNKER.chunk(content)):
            chunk_metadata = {
                **metadata,
                "doc_id": source_doc_id,
                "source_file": str(path),
                "chunk_index": str(index),
                "chunker": CHUNKER.__class__.__name__,
            }
            documents.append(
                Document(
                    id=f"{source_doc_id}#{index}",
                    content=chunk,
                    metadata=chunk_metadata,
                )
            )
    return documents


def run_baseline(data_dir: Path = DATA_DIR) -> None:
    print("=== Baseline Analysis (frontmatter stripped) ===")
    comparator = ChunkingStrategyComparator()
    for path in sorted(data_dir.glob("*.md"))[:3]:
        _, content = parse_markdown(path)
        result = comparator.compare(content, chunk_size=300)
        print(f"\n{path.name}")
        for strategy, stats in result.items():
            print(
                f"- {strategy}: count={stats['count']}, "
                f"avg_length={stats['avg_length']:.1f}"
            )


def print_result(result: dict[str, Any], rank: int) -> None:
    metadata = result.get("metadata", {})
    preview = " ".join(result.get("content", "").split())[:180]
    print(
        f"  {rank}. score={result['score']:.3f} "
        f"doc_id={metadata.get('doc_id')} chunk={metadata.get('chunk_index')}"
    )
    print(f"     {preview}")


def main() -> int:
    run_baseline()

    docs = make_documents()
    store = EmbeddingStore(collection_name="dorm_benchmark", embedding_fn=LexicalHashEmbedder())
    store.add_documents(docs)
    print("\n=== Retrieval Benchmark ===")
    print(f"Chunker: {CHUNKER.__class__.__name__}")
    print(f"Loaded chunks: {store.get_collection_size()}")

    for index, item in enumerate(BENCHMARK_QUERIES, start=1):
        query = item["question"]
        metadata_filter = item["metadata_filter"]
        results = store.search_with_filter(query, top_k=3, metadata_filter=metadata_filter)

        print(f"\nQ{index}: {query}")
        print(f"Filter: {metadata_filter}")
        print(f"Gold doc: {item['gold_doc_id']}")
        print(f"Gold answer: {item['gold_answer']}")
        print("Top-3:")
        for rank, result in enumerate(results, start=1):
            print_result(result, rank)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
