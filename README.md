# LMStudio Batching Comparison

Benchmarks three request strategies against a local LM Studio model using [mellea](https://github.com/IBM/mellea).

50 prompts are sent per run. A fixed seed is used for reproducibility. 3 warm-up requests run before timing starts.

## Scripts

| Script | Description |
|---|---|
| `compare_batching.py` | Compares sequential vs parallel (batched & interleaved) execution |
| `compare_formal.py` | Same comparison, with a mellea requirement: "Use a formal language." |

Both scripts compare these methods:

| Method | Description |
|---|---|
| **Sequential** | One request at a time |
| **Parallel batched** | All `ainstruct` calls first, then all `avalue` calls |
| **Parallel interleaved** | Each prompt does `ainstruct` + `avalue` concurrently |

## Requirements

- [LM Studio](https://lmstudio.ai) running locally with a model loaded
- Python 3.12+

## Setup

```bash
uv sync
```

## Usage

```bash
# Run all three methods
python compare_batching.py
python compare_formal.py

# Run a specific parallel method only
python compare_batching.py --method batched
python compare_formal.py --method interleaved
```

## Configuration

`LMSTUDIO_BASE_URL` is read from the `LM_STUDIO_BASE_URL` env variable (default: `http://127.0.0.1:1234/v1`). `MODEL_ID` and `SEED` can be edited at the top of each script.

## Some results for NO requirements

Running `granite-4.0-tiny@q8_k` on my M3 Max with parallelism `8`:

```
============================================================
Sequential:          24.59s
Parallel batched      10.51s  (2.34x speedup)
Parallel interleaved  9.19s  (2.68x speedup)
============================================================
```


Running `gpt-oss-20b` on **REMOTE** ARM64+GB200 with parallelism 8:
```
============================================================
Sequential:          16.24s
Parallel batched      9.61s  (1.69x speedup)
Parallel interleaved  10.29s  (1.58x speedup)
============================================================
```

.... with parallelism 16:
```
============================================================
Sequential:          16.50s
Parallel batched      7.97s  (2.07x speedup)
Parallel interleaved  7.76s  (2.13x speedup)
============================================================
```


## Some results for WITH requirement


Running `granite-4.0-tiny@q8_k` on my M3 Max with parallelism `8`:

```
============================================================
Sequential:          41.11s
Parallel batched      17.14s  (2.40x speedup)
Parallel interleaved  16.95s  (2.43x speedup)
============================================================
```




Running `gpt-oss-20b` on **REMOTE** ARM64+GB200 with parallelism 16:

```
============================================================
Sequential:          42.48s
Parallel batched      18.15s  (2.34x speedup)
Parallel interleaved  18.61s  (2.28x speedup)
============================================================
```



