#!/usr/bin/env bash
# Reproduce one model's olmOCR-bench category numbers, end to end:
#   serve (vLLM) -> wait ready -> run.py (generate) -> evaluate.py (score) -> print a markdown row.
#
#   bash reproduce.sh <HF_MODEL_ID>
#   bash reproduce.sh Qwen/Qwen3.5-4B
#
# Always runs the full benchmark: run.py generates all 7 categories (1403 PDFs) in one pass.
# Single-GPU and linear by design. Bring-your-own serving: set ENDPOINT to skip the local serve.
set -euo pipefail

# ── Overrides (edit these to your machine) ───────────────────────────────────
REPO="$(cd "$(dirname "$0")" && pwd)"
SERVE_VENV="${SERVE_VENV:-$REPO/.venv-serve}"   # serve venv (vllm==0.23.0)
BENCH_VENV="${BENCH_VENV:-$REPO/.venv}"         # bench venv (olmocr[bench]+playwright, built by setup.sh)
ENDPOINT="${ENDPOINT:-}"          # OpenAI-compatible base URL; empty => serve locally
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-32}"
RENDER_TIMEOUT_MS="${RENDER_TIMEOUT_MS:-5000}"
# ─────────────────────────────────────────────────────────────────────────────

MODEL="${1:?usage: reproduce.sh <HF_MODEL_ID>}"
NAME="${MODEL##*/}"                              # clean cache/served/row name, e.g. Qwen3.5-4B
API_BASE="${ENDPOINT:-http://localhost:$PORT/v1}"

SERVE_PID=""
cleanup() { [[ -n "$SERVE_PID" ]] && kill -TERM "$SERVE_PID" 2>/dev/null || true; }
trap cleanup EXIT

# (1) serve locally unless an ENDPOINT was provided.
if [[ -z "$ENDPOINT" ]]; then
  export PATH="$SERVE_VENV/bin:/usr/local/cuda/bin:$PATH"  # nvcc on PATH for flashinfer JIT
  EXTRA=()
  # Qwen3.5 uses GDN linear attention; "auto" picks a flashinfer kernel whose first-run JIT
  # compile is pathologically slow on Hopper. Force triton for a fast startup. (Qwen3.5-only.)
  if [[ "$MODEL" == *Qwen3.5* ]]; then
    EXTRA+=(--additional-config '{"gdn_prefill_backend": "triton"}')
  fi
  echo "### serve $MODEL on :$PORT"
  "$SERVE_VENV/bin/vllm" serve "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --served-model-name "$MODEL" \
    "${EXTRA[@]}" > "$REPO/serve_${NAME}.log" 2>&1 &
  SERVE_PID=$!
fi

# (2) wait for the endpoint to answer /v1/models (timeout ~600s).
echo "### wait for $API_BASE ready"
"$BENCH_VENV/bin/python" - "$API_BASE" <<'PY'
import sys, time, httpx
base = sys.argv[1].rstrip("/")
deadline = time.time() + 600
while time.time() < deadline:
    try:
        if httpx.get(f"{base}/models", timeout=5).status_code == 200:
            print("READY"); sys.exit(0)
    except Exception:
        pass
    time.sleep(6)
print("TIMEOUT waiting for serve", file=sys.stderr); sys.exit(1)
PY

# (3) generate predictions over all categories.
echo "### generate ($NAME)"
( cd "$REPO" && HOSTED_VLLM_API_BASE="$API_BASE" HOSTED_VLLM_API_KEY=dummy \
    "$BENCH_VENV/bin/python" benchmarks/olmocr/run.py \
      --provider litellm --model-id "hosted_vllm/$MODEL" --model "$NAME" --workers "$WORKERS" )

# (4) score with the bounded render timeout and print one markdown row.
echo "### score ($NAME)"
"$BENCH_VENV/bin/python" "$REPO/parse_eval.py" --header
( cd "$REPO" && RENDER_TIMEOUT_MS="$RENDER_TIMEOUT_MS" \
    "$BENCH_VENV/bin/python" benchmarks/olmocr/evaluate.py --model "$NAME" 2>/dev/null ) \
  | "$BENCH_VENV/bin/python" "$REPO/parse_eval.py" --name "$NAME"
