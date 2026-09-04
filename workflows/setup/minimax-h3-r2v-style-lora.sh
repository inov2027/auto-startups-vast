#!/usr/bin/env bash
# ---
# name: MiniMax H3 R2V + illustration style LoRAs
# workflow: minimax_h3_r2v_style_lora
# aliases: [minimax-h3-r2v-style-lora, h3-style-lora, h3-illustration, story-maker-v4-style]
# description: MiniMax H3 Ref2VA base stack (ref2va pruned int8 convrot UNet, Qwen3-VL 32B int8 convrot text encoder, video + audio VAEs) plus the illustration style LoRAs story-maker-v4 uses for 2D storybook, folk / flat-geometric, vintage editorial storybook, and semi-realistic painterly anime concept-art looks. Adds the official ref2v 4-step turbo LoRA and two directing LoRAs (camera motion, spatial physics). H3 nodes are comfy-core (>= v0.30.0) — the only custom packs are the optional attention/cache patches already used by the R2V graph.
# size: ~53GB
# min_vram: 24GB
# nodes: [ComfyUI-KJNodes, ComfyUI-SolAttn_triton]
# ---
set -Eeuo pipefail

# -------- Platform-aware ComfyUI discovery --------
if [[ -d /workspace/runpod-slim/ComfyUI ]]; then
  COMFYUI_DIR=/workspace/runpod-slim/ComfyUI
elif [[ -d /workspace/ComfyUI ]]; then
  COMFYUI_DIR=/workspace/ComfyUI
elif [[ -d /workspace/ComfyUI_windows_portable/ComfyUI ]]; then
  COMFYUI_DIR=/workspace/ComfyUI_windows_portable/ComfyUI
else
  COMFYUI_DIR="${COMFYUI_DIR:-$PWD}"
fi
BASE_DIR="$COMFYUI_DIR"
NODES_DIR="$COMFYUI_DIR/custom_nodes"
MODELS_DIR="$COMFYUI_DIR/models"
LORAS_DIR="$MODELS_DIR/loras"
mkdir -p "$NODES_DIR" "$MODELS_DIR" "$LORAS_DIR" "$COMFYUI_DIR/input"

detect_python() {
  local pid
  pid=$(ps -eo pid,comm,args | awk '$2 ~ /python/ && /main\.py/ && !/tcl/ {print $1; exit}') || true
  if [[ -n "${pid:-}" && -f "/proc/$pid/exe" ]]; then readlink -f "/proc/$pid/exe"; return; fi
  for p in "$COMFYUI_DIR/.venv/bin/python" "$COMFYUI_DIR/.venv-cu128/bin/python" /venv/main/bin/python3 python3; do
    command -v "$p" >/dev/null 2>&1 || [[ -x "$p" ]] && { echo "$p"; return; }
  done
  echo python3
}
COMFYUI_PYTHON=$(detect_python)

ver_ge() { [[ "$(printf '%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]; }
current_version=$($COMFYUI_PYTHON -c 'import comfyui_version; print(comfyui_version.__version__)' 2>/dev/null || echo unknown)
required_version=v0.30.0

# -------- Phase 0: H3 core-node version floor --------
echo "==> Phase 0: ComfyUI version=$current_version; required>=$required_version"
if [[ "$current_version" == unknown ]] || ! ver_ge "$current_version" "$required_version"; then
  echo "  Upgrading ComfyUI (MiniMaxH3ReferenceToVideo is comfy-core >= v0.30.0)"
  git -C "$COMFYUI_DIR" stash --quiet || true
  git -C "$COMFYUI_DIR" fetch origin --tags --quiet
  latest_tag=$(git -C "$COMFYUI_DIR" tag --sort=-version:refname | grep -E '^v0\.3[0-9]' | head -1 || true)
  if [[ -n "$latest_tag" ]]; then git -C "$COMFYUI_DIR" checkout "$latest_tag"; else git -C "$COMFYUI_DIR" checkout origin/master; fi
  [[ -f "$COMFYUI_DIR/requirements.txt" ]] && $COMFYUI_PYTHON -m pip install -r "$COMFYUI_DIR/requirements.txt" -q
fi

# -------- Phase 1: custom nodes --------
# LoraLoaderModelOnly is comfy-core. Only the optional attention/cache patch
# nodes already present in the R2V graph need installing.
install_node() {
  local dir="$1" url="$2"
  if [[ -d "$NODES_DIR/$dir/.git" ]]; then
    echo "  ✅ $dir already installed"
  else
    echo "  📥 Installing $dir"
    git clone --depth 1 "$url" "$NODES_DIR/$dir"
    if [[ -f "$NODES_DIR/$dir/requirements.txt" ]]; then
      $COMFYUI_PYTHON -m pip install -r "$NODES_DIR/$dir/requirements.txt" -q
    fi
  fi
}
install_node ComfyUI-KJNodes https://github.com/kijai/ComfyUI-KJNodes.git
install_node ComfyUI-SolAttn_triton https://github.com/kijai/ComfyUI-SolAttn_triton.git

# -------- Phase 2: models --------
$COMFYUI_PYTHON -m pip install -q huggingface_hub
export HF_HUB_DISABLE_PROGRESS_BARS=0
export HF_XET_HIGH_PERFORMANCE=1

# hf_file REPO FILENAME [LOCAL_DIR]
# local_dir defaults to the ComfyUI root so the filename's subdir prefix
# (diffusion_models/, vae/, ...) is not duplicated.
hf_file() {
  local repo=$1 file=$2 dir=${3:-$BASE_DIR}
  echo "  📥 $repo/$file"
  $COMFYUI_PYTHON - "$repo" "$file" "$dir" <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
repo, filename, local_dir = sys.argv[1:]
print(hf_hub_download(repo_id=repo, filename=filename, local_dir=local_dir))
PYEOF
}

# hf_lora REPO FILENAME LOCAL_NAME — downloads into models/loras/ under a
# ComfyUI-safe filename (community mirrors ship names with spaces/commas).
hf_lora() {
  local repo=$1 file=$2 name=$3
  if [[ -f "$LORAS_DIR/$name" ]]; then
    echo "  ✅ loras/$name already present"
    return
  fi
  echo "  📥 $repo/$file -> loras/$name"
  $COMFYUI_PYTHON - "$repo" "$file" "$LORAS_DIR" "$name" <<'PYEOF'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo, filename, loras_dir, name = sys.argv[1:]
src = hf_hub_download(repo_id=repo, filename=filename, local_dir=loras_dir)
dst = os.path.join(loras_dir, name)
if os.path.abspath(src) != os.path.abspath(dst):
    shutil.move(src, dst)
print(dst)
PYEOF
}

echo "==> Phase 2a: MiniMax H3 Ref2VA base stack (~51GB)"
H3=Comfy-Org/MiniMax-H3
hf_file "$H3" diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors
hf_file "$H3" text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
hf_file "$H3" vae/minimax_h3_video_vae_fp16.safetensors
hf_file "$H3" vae/minimax_h3_audio_vae_fp32.safetensors

echo "==> Phase 2b: acceleration LoRA (official ref2v 4-step turbo, ~1.9GB)"
hf_lora "$H3" \
  loras/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors \
  minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors

echo "==> Phase 2c: illustration style LoRAs (~1.6GB)"
# STUDIO 1939 — hand-painted golden-age animation. light(r16)=painterly
# storybook, strong(r64)=flat cel characters over painted backgrounds.
# Covers 2D storybook + folk + vintage editorial.
hf_lora lovis93/studio-1939-old-animation-lora-minimax-h3 \
  studio1939-light.safetensors studio1939-light.safetensors
hf_lora lovis93/studio-1939-old-animation-lora-minimax-h3 \
  studio1939-strong.safetensors studio1939-strong.safetensors

# Hand-drawn 2D anime sketch: rough textured outlines, flat minimalist colour.
hf_lora Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime \
  minimax_h3_looping_sketch_anime_v1.safetensors \
  minimax_h3_looping_sketch_anime_v1.safetensors

echo "==> Phase 2d: directing LoRAs (~0.3GB)"
hf_lora Jojocodex/minimax-h3-Camera-Motion-lora \
  camera_motion_h3_lora_v1_3000_pruned.safetensors \
  h3_camera_motion_v1_3000_pruned.safetensors
hf_lora Jojocodex/minimax-h3-spatial-physics-lora \
  wushu_spatial_physics_clean_3000_pruned.safetensors \
  h3_spatial_physics_clean_3000_pruned.safetensors

# -------- Phase 2e: optional community-mirror style LoRAs --------
# EllaPriest45/MinimaxH3_Styles is a Civitai backup mirror: the weights are
# real and H3-native, but provenance and licensing are the original authors'.
# Opt in with STYLE_LORA_COMMUNITY=1.
if [[ "${STYLE_LORA_COMMUNITY:-0}" == "1" ]]; then
  echo "==> Phase 2e: community-mirror style LoRAs (~0.4GB)"
  hf_lora EllaPriest45/MinimaxH3_Styles \
    "Painterly - MinimaxH3.safetensors" h3_painterly.safetensors
  hf_lora EllaPriest45/MinimaxH3_Styles \
    "Anime Flat Style - MinimaxH3.safetensors" h3_anime_flat_style.safetensors
else
  echo "==> Phase 2e: skipped (set STYLE_LORA_COMMUNITY=1 to fetch the Civitai-mirror style LoRAs)"
fi

# -------- Phase 3: restart and verify --------
COMFYUI_ARGS=$(ps aux | awk '/[p]ython.*main\.py/ {sub(/.*main\.py/, ""); print; exit}') || true
[[ -n "${COMFYUI_ARGS:-}" ]] || COMFYUI_ARGS="--listen 0.0.0.0 --port 18188 --enable-cors-header"
[[ " $COMFYUI_ARGS " == *" --disable-pinned-memory "* ]] || COMFYUI_ARGS+=" --disable-pinned-memory"
[[ " $COMFYUI_ARGS " == *" --fp16-intermediates "* ]] || COMFYUI_ARGS+=" --fp16-intermediates"
COMFYUI_PORT=$(grep -oE -- '--port [0-9]+' <<<"$COMFYUI_ARGS" | awk '{print $2}' | head -1 || true)
[[ -n "${COMFYUI_PORT:-}" ]] || COMFYUI_PORT=8188

if command -v supervisorctl >/dev/null 2>&1 && supervisorctl status comfyui >/dev/null 2>&1; then
  supervisorctl restart comfyui
else
  pid=$(ps -eo pid,comm,args | awk '$2 ~ /python/ && /main\.py/ && !/tcl/ {print $1; exit}') || true
  [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
  sleep 3
  (cd "$COMFYUI_DIR" && nohup "$COMFYUI_PYTHON" main.py $COMFYUI_ARGS > "$COMFYUI_DIR/comfyui.log" 2>&1 &)
fi
for _ in $(seq 1 45); do
  if curl -fsS --max-time 3 "http://127.0.0.1:$COMFYUI_PORT/system_stats" >/dev/null 2>&1; then
    echo "✅ ComfyUI is ready on port $COMFYUI_PORT"
    break
  fi
  sleep 2
done

for f in \
  "$MODELS_DIR/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" \
  "$MODELS_DIR/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
  "$MODELS_DIR/vae/minimax_h3_video_vae_fp16.safetensors" \
  "$MODELS_DIR/vae/minimax_h3_audio_vae_fp32.safetensors" \
  "$LORAS_DIR/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors" \
  "$LORAS_DIR/studio1939-light.safetensors" \
  "$LORAS_DIR/studio1939-strong.safetensors" \
  "$LORAS_DIR/minimax_h3_looping_sketch_anime_v1.safetensors" \
  "$LORAS_DIR/h3_camera_motion_v1_3000_pruned.safetensors" \
  "$LORAS_DIR/h3_spatial_physics_clean_3000_pruned.safetensors"; do
  [[ -f "$f" ]] && echo "✅ $(basename "$f")" || echo "❌ Missing: $f"
done
echo "🎉 MiniMax H3 R2V style-LoRA setup complete."
