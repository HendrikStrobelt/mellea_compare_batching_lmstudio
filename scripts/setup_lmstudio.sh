#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
PORT=8011
CONTEXT_LENGTH=16000
PARALLEL=8
MODEL_GPT="gpt-oss-20b"
MODEL_GRANITE="granite-4.0-micro"
TMUX_SESSION="lmstudio"
# ──────────────────────────────────────────────────────────────────────────────

export PATH="$HOME/.lmstudio/bin:$PATH"

# ── 1. Install LM Studio CLI ───────────────────────────────────────────────────
if ! command -v lms &>/dev/null; then
  echo "==> Installing LM Studio CLI..."
  curl -fsSL https://lmstudio.ai/install.sh | bash
  # Reload PATH after install
  export PATH="$HOME/.lmstudio/bin:$PATH"
else
  echo "==> lms already installed ($(lms --version 2>/dev/null || echo 'version unknown'))"
fi

# ── 2. Start the daemon ────────────────────────────────────────────────────────
echo "==> Starting LM Studio daemon..."
lms daemon up

# ── 3. Download models (Q8 quantization) ──────────────────────────────────────
echo "==> Downloading $MODEL_GPT (q8_0)..."
lms get "$MODEL_GPT" --quantization q8_0

echo "==> Downloading $MODEL_GRANITE (q8_0)..."
lms get "$MODEL_GRANITE" --quantization q8_0

# ── 4. Start the server ────────────────────────────────────────────────────────
echo "==> Starting LM Studio server on port $PORT..."
lms server start --bind 0.0.0.0 --port "$PORT"

# ── 5. Load both models ────────────────────────────────────────────────────────
echo "==> Loading $MODEL_GPT..."
lms load "$MODEL_GPT" \
  --identifier "$MODEL_GPT" \
  --context-length "$CONTEXT_LENGTH" \
  --parallel "$PARALLEL"

echo "==> Loading $MODEL_GRANITE..."
lms load "$MODEL_GRANITE" \
  --context-length "$CONTEXT_LENGTH" \
  --parallel "$PARALLEL"

# ── 6. Open tmux monitoring session ───────────────────────────────────────────
echo "==> Setting up tmux session '$TMUX_SESSION'..."

if ! command -v tmux &>/dev/null; then
  echo "ERROR: tmux is not installed. Install it with: sudo apt install tmux" >&2
  exit 1
fi

# Kill existing session if it exists
tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

# Create new session with pane A (left)
tmux new-session -d -s "$TMUX_SESSION" -x 220 -y 50

# Split right: creates pane B1 (top-right)
tmux split-window -h -t "$TMUX_SESSION"

# Split B1 vertically to create B2 (bottom-right)
tmux split-window -v -t "$TMUX_SESSION"

# Send commands to B1 (pane index 1) and B2 (pane index 2)
tmux send-keys -t "$TMUX_SESSION:0.1" "lms log stream" Enter
tmux send-keys -t "$TMUX_SESSION:0.2" "watch -n 1 nvidia-smi" Enter

# Focus pane A (pane index 0)
tmux select-pane -t "$TMUX_SESSION:0.0"

# Attach to session
echo "==> Attaching to tmux session '$TMUX_SESSION'..."
tmux attach-session -t "$TMUX_SESSION"
