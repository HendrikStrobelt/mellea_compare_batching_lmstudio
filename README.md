# LMStudio Batching Comparison

Benchmarks three request strategies against a local LM Studio model using [mellea](https://github.com/IBM/mellea):

| Method | Description |
|---|---|
| **Sequential** | One request at a time |
| **Parallel batched** | All `ainstruct` calls first, then all `avalue` calls |
| **Parallel interleaved** | Each prompt does `ainstruct` + `avalue` concurrently |

50 prompts are sent per run. A fixed seed is used for reproducibility. 3 warm-up requests run before timing starts.

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
python main.py

# Run a specific parallel method only
python main.py --method batched
python main.py --method interleaved
```

## Configuration

Edit the top of `main.py`:

```python
LMSTUDIO_BASE_URL = "http://127.0.0.1:1234/v1"
MODEL_ID = "granite-4.0-micro@q8_0"
SEED = 42
```