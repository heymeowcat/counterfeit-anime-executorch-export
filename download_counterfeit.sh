#!/bin/bash
# Fetch the Counterfeit V3.0 anime/illustration pipeline sources.
#
# Counterfeit V3.0 is a Stable Diffusion 1.5 model, so the export path is the
# same one the epicrealism-512 build uses: base checkpoint + ByteDance Hyper-SD15
# 4-step LoRA (fused in FP32) + the full AutoencoderKL decode.
#
# The `_fix_fp16` file is the community-blessed variant: the plain fp16 release
# had a NaN/black-image bug, and this one carries the fixed UNet weights plus the
# baked VAE, so no external VAE is needed.
set -e
mkdir -p models/counterfeit models/hyper_sd models/swmansion_tokenizer

H="https://huggingface.co"
d(){
    mkdir -p "$(dirname "$1")"
    echo ">>> Downloading $1 from $2"
    aria2c -x16 -s16 -k4M --file-allocation=none -c -d "$(dirname "$1")" -o "$(basename "$1")" "$2"
    echo "OK: $1"
}

# 1. Counterfeit V3.0 pruned fp16 (fixed) safetensors (~1.98 GB, VAE included)
d "models/counterfeit/Counterfeit-V3.0_fix_fp16.safetensors" \
  "$H/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors"

# 2. Hyper-SD 4-steps LoRA (256 MB)
d "models/hyper_sd/Hyper-SD15-4steps-lora.safetensors" \
  "$H/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-4steps-lora.safetensors"

# 3. CLIP ViT-L/14 tokenizer (3.5 MB) — from Software Mansion's RN ExecuTorch SDXS export
d "models/swmansion_tokenizer/tokenizer.json" \
  "$H/software-mansion/react-native-executorch-sdxs-512-dreamshaper/resolve/main/tokenizer.json"
d "models/swmansion_tokenizer/tokenizer_config.json" \
  "$H/software-mansion/react-native-executorch-sdxs-512-dreamshaper/resolve/main/tokenizer_config.json"

echo "ALL_DOWNLOADS_COMPLETED"
