import argparse
import asyncio
import os
import time

from mellea import MelleaSession
from mellea.backends.model_options import ModelOption
from mellea.backends.openai import OpenAIBackend
from mellea.core import FancyLogger

LMSTUDIO_BASE_URL = os.environ.get("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
# MODEL_ID = "gpt-oss-20b"
MODEL_ID="granite-4.0-micro@q8_0"

SEED = 42

PROMPTS = [
    "What causes lightning?",
    "Explain the water cycle in two sentences.",
    "What is the difference between a virus and a bacterium?",
    "How does photosynthesis work?",
    "What is the speed of light?",
    "Why is the sky blue?",
    "How do black holes form?",
    "What is DNA and what does it do?",
    "Explain Newton's first law of motion.",
    "What is the greenhouse effect?",
    "Who built the Great Wall of China and why?",
    "What caused the fall of the Roman Empire?",
    "Describe the significance of the Magna Carta.",
    "What was the Industrial Revolution?",
    "Who was Cleopatra?",
    "What happened at the Battle of Waterloo?",
    "Why did the Berlin Wall fall?",
    "What was the Cold War about?",
    "Who invented the printing press?",
    "What was the significance of the moon landing?",
    "Write a two-line haiku about autumn leaves.",
    "Give a one-sentence moral for a story about a greedy merchant.",
    "Describe the setting of a haunted lighthouse in one sentence.",
    "Write a short metaphor comparing time to a river.",
    "Suggest a creative name for a bakery that specializes in sourdough.",
    "What is a good opening line for a mystery novel?",
    "Describe the color red without using color words.",
    "Write a one-sentence plot for a sci-fi story set on Mars.",
    "Give a witty slogan for a coffee shop.",
    "Describe the feeling of nostalgia in one sentence.",
    "What does the Python keyword 'yield' do?",
    "Explain the difference between a list and a tuple in Python.",
    "What is a REST API?",
    "What is the purpose of a foreign key in a database?",
    "Explain what recursion is in programming.",
    "What is the difference between TCP and UDP?",
    "What does 'async/await' do in Python?",
    "What is a hash map and when would you use one?",
    "Explain what version control is and why it matters.",
    "What is the difference between compiled and interpreted languages?",
    "What is the boiling point of water at sea level?",
    "What is the largest planet in our solar system?",
    "How many bones are in the human body?",
    "What is the chemical symbol for gold?",
    "What country has the longest coastline in the world?",
    "What is the tallest mountain on Earth?",
    "What language has the most native speakers?",
    "What is the capital of Australia?",
    "How many continents are there on Earth?",
    "What is the hardest natural substance on Earth?",
]

FancyLogger.get_logger().setLevel(FancyLogger.ERROR)

def make_backend() -> OpenAIBackend:
    return OpenAIBackend(
        model_id=MODEL_ID,
        base_url=LMSTUDIO_BASE_URL,
        api_key="lm-studio",
        model_options={ModelOption.SEED: SEED},
    )


def run_sequential() -> tuple[list[str], float]:
    session = MelleaSession(make_backend())
    results = []
    start = time.perf_counter()
    for i, prompt in enumerate(PROMPTS, 1):
        result = str(session.instruct(prompt+"\n Answer in maximum 1 sentence."))
        results.append(result)
        print(f"  [{i:02d}/50] done ({len(result)} chars)")
    elapsed = time.perf_counter() - start
    return results, elapsed


async def instruct_and_resolve(session: MelleaSession, prompt: str) -> str:
    mot = await session.ainstruct(prompt)
    return await mot.avalue()


async def run_parallel(method: str = "batched") -> tuple[list[str], float]:
    session = MelleaSession(make_backend())
    start = time.perf_counter()
    prompts = [p + "\n Answer in maximum 1 sentence." for p in PROMPTS]
    if method == "interleaved":
        results = list(await asyncio.gather(*[instruct_and_resolve(session, p) for p in prompts]))
    else:
        mots = await asyncio.gather(*[session.ainstruct(p) for p in prompts])
        results = await asyncio.gather(*[mot.avalue() for mot in mots])
    results = [str(x) for x in results]
    elapsed = time.perf_counter() - start
    for i, r in enumerate(results, 1):
        print(f"  [{i:02d}/50] done ({len(r)} chars)")
    return results, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["all", "batched", "interleaved"],
        default="all",
        help="Which parallel method to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Batching comparison: Sequential vs Parallel (50 prompts)")
    print(f"Model: {MODEL_ID}  |  Backend: LM Studio @ {LMSTUDIO_BASE_URL}")
    print("=" * 60)

    print("\n--- Warm-up (3 requests) ---")
    session = MelleaSession(make_backend())
    for i in range(3):
        str(session.instruct("Say hello."))
        print(f"  warm-up {i + 1}/3 done")

    print("\n--- Sequential run ---")
    seq_results, seq_time = run_sequential()
    print(f"Sequential total: {seq_time:.2f}s")

    par_times = {}

    if args.method in ("all", "batched"):
        print("\n--- Parallel batched (ainstruct all → avalue all) ---")
        _, t = asyncio.run(run_parallel("batched"))
        par_times["batched"] = t
        print(f"Parallel batched total: {t:.2f}s")

    if args.method in ("all", "interleaved"):
        print("\n--- Parallel interleaved (instruct+value per prompt) ---")
        _, t = asyncio.run(run_parallel("interleaved"))
        par_times["interleaved"] = t
        print(f"Parallel interleaved total: {t:.2f}s")

    print("\n" + "=" * 60)
    print(f"Sequential:          {seq_time:.2f}s")
    for name, t in par_times.items():
        speedup = seq_time / t if t > 0 else float("inf")
        print(f"Parallel {name:<12} {t:.2f}s  ({speedup:.2f}x speedup)")
    print("=" * 60)


if __name__ == "__main__":
    main()

