import argparse
import time
from dataclasses import dataclass

from rag.retriever import hybrid_search, format_evidence
from rag.llm import call_local_llm
from rag import config


@dataclass
class Timing:
    name: str
    seconds: float


def now():
    return time.perf_counter()


def timed(name, fn):
    start = now()
    result = fn()
    end = now()
    return result, Timing(name, end - start)


def print_timings(timings):
    total = sum(t.seconds for t in timings)
    print("\n=== Timing Breakdown ===")
    for t in timings:
        pct = (t.seconds / total * 100) if total > 0 else 0
        print(f"{t.name:<28} {t.seconds:>8.3f} s   {pct:>6.2f}%")
    print("-" * 50)
    print(f"{'TOTAL':<28} {total:>8.3f} s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        default="What does TS 33.501 say about Service-Based Architecture security?",
    )
    parser.add_argument("--max-chunks", type=int, default=3)
    parser.add_argument("--max-chars-per-chunk", type=int, default=800)
    parser.add_argument("--k-vector", type=int, default=3)
    parser.add_argument("--k-keyword", type=int, default=3)
    parser.add_argument("--search-specs", action="store_true", default=True)
    parser.add_argument("--search-meetings", action="store_true", default=False)
    args = parser.parse_args()

    timings = []

    print("=== RAG Profiling Run ===")
    print(f"Question: {args.question}")
    print(f"CHAT_MODEL: {getattr(config, 'CHAT_MODEL', 'unknown')}")
    print(f"EMBED_MODEL: {getattr(config, 'EMBED_MODEL', 'unknown')}")
    print(f"OLLAMA_URL: {getattr(config, 'OLLAMA_URL', 'unknown')}")
    print(f"max_chunks={args.max_chunks}")
    print(f"max_chars_per_chunk={args.max_chars_per_chunk}")
    print(f"k_vector={args.k_vector}")
    print(f"k_keyword={args.k_keyword}")

    total_start = now()

    results, t = timed(
        "hybrid_search",
        lambda: hybrid_search(
            args.question,
            search_specs=args.search_specs,
            search_meetings=args.search_meetings,
            k_vector=args.k_vector,
            k_keyword=args.k_keyword,
        ),
    )
    timings.append(t)

    limited_results = results[: args.max_chunks]

    evidence, t = timed(
        "format_evidence",
        lambda: format_evidence(
            limited_results,
            max_chars_per_chunk=args.max_chars_per_chunk,
        ),
    )
    timings.append(t)

    def build_prompt():
        return f"""
You are a 3GPP SA3 research assistant.

Important acronym rule:
In 3GPP/5G context, SBA means Service-Based Architecture unless the evidence clearly says otherwise.

Answer the question using only the evidence.
If the evidence is insufficient, say so clearly.
Do not invent clause numbers, TDoc IDs, companies, or requirements.
Cite evidence using [Evidence X].

Question:
{args.question}

Evidence:
{evidence}

Answer:
"""

    prompt, t = timed("build_prompt", build_prompt)
    timings.append(t)

    answer, t = timed("llm_generation", lambda: call_local_llm(prompt))
    timings.append(t)

    total_end = now()

    print("\n=== Retrieval Stats ===")
    print(f"Retrieved results: {len(results)}")
    print(f"Used results: {len(limited_results)}")
    print(f"Evidence chars: {len(evidence)}")
    print(f"Prompt chars: {len(prompt)}")

    print_timings(timings)

    print("\n=== Total Wall Time ===")
    print(f"{total_end - total_start:.3f} s")

    print("\n=== Answer ===")
    print(answer)


if __name__ == "__main__":
    main()
