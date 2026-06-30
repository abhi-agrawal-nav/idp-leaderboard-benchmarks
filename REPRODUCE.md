# Reproducing the idp-leaderboard olmOCR-bench numbers for open VLMs

This is a fork of [`NanoNets/idp-leaderboard-benchmarks`](https://github.com/NanoNets/idp-leaderboard-benchmarks)
at `af8c743`. We ran your olmOCR-bench harness over a set of open, HF-pullable vision LLMs and
got the per-category numbers in the table below. The delta is two small commented edits to
`benchmarks/olmocr/evaluate.py` plus a thin run harness (`reproduce.sh`,
`parse_eval.py`); `git diff af8c743..HEAD` is the entire delta. We'd like your read on whether
these numbers look right.

## Changes vs stock idp

We make two small, commented edits to `benchmarks/olmocr/evaluate.py` so the scorer doesn't choke
on weaker models. Neither touches the pass/fail logic or thresholds — they only stop the scorer
from **hanging** or from **throwing away a whole model's score**.

- **Cap the equation-rendering timeout.** To grade a math test, the scorer draws the equation in a
headless browser and waits for it to finish — but those waits have *no time limit*, so a single
equation the browser can't render makes the whole run hang forever. We cap the waits at 5 seconds
(`RENDER_TIMEOUT_MS`): an equation that won't render now fails in ~5s instead of freezing scoring.
- **Don't let one bad page zero out the whole model.** If a single test errors or a page's output
is missing, the scorer already counts that test as failed — but it *also* raises a flag that makes
it discard the model's **entire** score (drops it to 0.0). We clear that flag (after logging how
many there were), so a bad page costs one failed test, not the whole model. A genuinely broken run
still shows up as a large logged count.

## Environment

You will need two venvs:

- **Bench venv** (`.venv/`) — build with the upstream installer: `./setup.sh`. Use **Python 3.12**.
  This venv runs `run.py` and `evaluate.py`.
- **Serve venv** (`.venv-serve/`) — `vllm==0.23.0`, installed from the release wheel matching your
  CUDA.

Point `BENCH_VENV` / `SERVE_VENV` at the bench / serve venv directories at the top of
`reproduce.sh` — it runs `$BENCH_VENV/bin/python` for generation and scoring, and
`$SERVE_VENV/bin/vllm` for serving. Edit those (plus `PORT`, `WORKERS`, `RENDER_TIMEOUT_MS`) to
match your machine.

## Run

```sh
bash reproduce.sh <HF_MODEL_ID>      # e.g. Qwen/Qwen3.5-4B
```

It serves the model, generates over the full benchmark (all 7 categories, 1403 PDFs), scores with
`evaluate.py`, and prints one markdown table row.

- **Bring your own serving:** set `ENDPOINT=http://host:port/v1` to skip the local `vllm serve`
  and score against an existing OpenAI-compatible endpoint.
- **Hardware:** one CUDA GPU large enough for the model + KV cache. We used one H200 (141 GB);
  7–9B models fit comfortably on ~48 GB. 

## Results

Per-category olmOCR-bench pass rates (%), `overall` = mean of the 8 per-JSONL groups (7
content + synthesized baseline — the same aggregation `evaluate.py` prints). All rows scored with the bounded render timeout (5s) and a single temp-0 run.

| model | arxiv_math | old_scans_math | headers_footers | long_tiny_text | old_scans | multi_column | tables | baseline | overall |
|---|---|---|---|---|---|---|---|---|---|
| allenai/olmOCR-2-7B-1025 | 87.7 | 84.9 | 93.8 | 85.3 | 51.5 | 86.2 | 85.5 | 99.9 | **84.4** |
| zai-org/GLM-OCR | 81.2 | 75.8 | 90.9 | 79.0 | 40.5 | 78.6 | 75.8 | 99.4 | **77.6** |
| Qwen/Qwen3.5-9B | 87.2 | 82.1 | 52.2 | 84.4 | 48.1 | 83.4 | 84.4 | 98.6 | **77.6** |
| Qwen/Qwen2.5-VL-7B-Instruct | 83.1 | 79.7 | 62.1 | 83.0 | 46.4 | 81.8 | 76.9 | 99.6 | **76.6** |
| Qwen/Qwen3.5-4B | 86.3 | 82.1 | 50.4 | 83.9 | 41.3 | 79.3 | 83.1 | 98.3 | **75.6** |
| Qwen/Qwen3-VL-8B-Instruct | 82.3 | 81.7 | 36.1 | 84.8 | 44.1 | 82.7 | 84.1 | 99.4 | **74.4** |
| Qwen/Qwen3.5-2B | 81.8 | 79.5 | 56.2 | 78.1 | 39.0 | 77.8 | 81.7 | 97.5 | **73.9** |
| Qwen/Qwen2-VL-7B-Instruct | 77.1 | 74.5 | 68.8 | 68.3 | 38.2 | 73.3 | 72.6 | 97.5 | **71.3** |
| Qwen/Qwen3.5-0.8B | 71.4 | 63.5 | 64.6 | 58.8 | 33.8 | 65.2 | 60.0 | 94.8 | **64.0** |
| google/gemma-4-E4B-it | 20.6 | 49.3 | 38.8 | 25.1 | 27.9 | 29.9 | 64.4 | 98.8 | **44.4** |
| google/gemma-4-E2B-it | 11.0 | 28.8 | 53.4 | 7.9 | 22.8 | 13.2 | 51.0 | 99.1 | **35.9** |


