---
license: openrail
pipeline_tag: text-to-image
tags:
- text-to-image
- stable-diffusion
- anime
- illustration
- executorch
- react-native-executorch
library_name: executorch
---

# counterfeit-anime-512-executorch

This repository hosts **Counterfeit V3.0**, an anime / illustration Stable Diffusion 1.5 checkpoint, exported for ExecuTorch and the [React Native ExecuTorch](https://www.npmjs.com/package/react-native-executorch) library as an ExecuTorch `.pte` program, ready to run on device.

- **Upstream Base Model**: [Counterfeit V3.0 (gsdf, Civitai #4468)](https://civitai.com/models/4468/counterfeit-v30) — `Counterfeit-V3.0_fix_fp16.safetensors`
- **Distillation LoRA**: [ByteDance/Hyper-SD (Hyper-SD15-4steps-lora)](https://huggingface.co/ByteDance/Hyper-SD)
- **Latent Decoder**: the AutoencoderKL baked into the checkpoint (no external VAE)
- **Tokenizer**: [Software Mansion RN ExecuTorch SDXS export](https://huggingface.co/software-mansion/react-native-executorch-sdxs-512-dreamshaper) (CLIP ViT-L/14 `tokenizer.json`)
- **Maintainer**: [@heymeowcat](https://huggingface.co/heymeowcat)

> **Repository layout**: The GitHub repository contains the export pipeline, verification tooling and documentation. The multi-gigabyte `.pte` binary is hosted here on Hugging Face (GitHub does not host files >100 MB without paid LFS quota).

---

## Download

```bash
# Option A — Hugging Face CLI
hf download heymeowcat/counterfeit-anime-512-executorch \
  --include "xnnpack/*" "tokenizer.json" \
  --local-dir .

# Option B — Direct URLs
BASE="https://huggingface.co/heymeowcat/counterfeit-anime-512-executorch/resolve/main"
curl -L -o counterfeit_anime_512_xnnpack_fp32.pte "$BASE/xnnpack/counterfeit_anime_512_xnnpack_fp32.pte"
curl -L -o tokenizer.json                        "$BASE/tokenizer.json"
```

Or from Python:

```python
from huggingface_hub import hf_hub_download
pte_path = hf_hub_download("heymeowcat/counterfeit-anime-512-executorch",
                           "xnnpack/counterfeit_anime_512_xnnpack_fp32.pte")
```

---

## Variants

| Path (HF repo) | Backend | Precision | Size | Status |
| :--- | :--- | :--- | :--- | :--- |
| `xnnpack/counterfeit_anime_512_xnnpack_fp32.pte` | xnnpack | fp32 | 3,910.8 MB | **Production / Verified** |

SHA-256: `8f19cc17fbffa6659f436a1af54105987e4a2fe9877a4d9f47616fd3be960d6d`

---

## Why a distillation LoRA is fused in

Counterfeit V3.0 is a regular SD 1.5 checkpoint: it expects **20–25 steps at CFG 8–10** with a negative prompt. This export targets the fixed schedule used by React Native ExecuTorch SD 1.5 pipelines — 4 TCD steps `[999, 759, 499, 259]`, guidance 1.0, no negative branch — so the [Hyper-SD](https://arxiv.org/abs/2404.13686) 4-step distillation LoRA is fused into the UNet at export time.

Two details that matter when reproducing the build:

- **Fuse at scale 1.0.** The Hyper-SD card suggests a small fusion scale (e.g. `0.125`) for the CFG-preserving variants. At 4 steps with CFG off, anything below ~1.0 collapses into painterly mush on this checkpoint; 1.0 is required.
- **Clip skip 2 with the final layer norm.** Counterfeit is trained and used at clip skip 2. Picking the penultimate CLIP state is not enough: `hidden_states` are **pre-norm**, so `text_encoder.final_layer_norm` must be applied to the chosen state. Skipping the norm leaves activations around ±800 instead of ±30 and the UNet renders pure static. The norm is baked into `encode`, exactly as A1111 and ComfyUI do it (`layer_norm_hidden_state`).

---

## Portrait / Illustration Quality Benchmark

Exported in unquantized **FP32** with the full AutoencoderKL decoder, so there is no quantization drift and no TAESD watercolor smearing:

| Implementation | Decoder | Distillation | Visual Quality | Mean Pixel Diff vs Eager |
| :--- | :--- | :--- | :--- | :--- |
| **ExecuTorch `.pte` (Ours)** | **Full AutoencoderKL (baked)** | **Hyper-SD 4-Step** | **Clean anime line art, saturated but controlled color, detailed eyes** | **0.02 / 255 (Parity)** |
| 1-step fast tier | TAESD (5 MB) | 1-Step | Painterly smudges, mushy faces | > 40 / 255 |

### Sample Output (ExecuTorch Runtime)

Synthesized via the ExecuTorch C++ runtime on macOS CPU using the 4-step schedule `[999, 759, 499, 259]`, η = 0.3 and seed `1234`:

> *"a girl with short black hair standing on a rooftop at sunset, warm orange sky, city skyline behind her"*

| ExecuTorch Runtime | Eager PyTorch Baseline |
| :---: | :---: |
| ![ExecuTorch output](out/preview_counterfeit_xnnpack_fp32.png) | ![Eager baseline](out/counterfeit_eager_baseline.png) |
| `out/preview_counterfeit_xnnpack_fp32.png` | `out/counterfeit_eager_baseline.png` |

---

## Example Gallery

Every image below was generated **straight from `counterfeit_anime_512_xnnpack_fp32.pte`** through the ExecuTorch runtime (4-step TCD schedule `[999, 759, 499, 259]`, η = 0.3, guidance 1.0), using plain natural-language prompts in the style Counterfeit was trained on. Fully reproducible via [`generate_gallery.py`](generate_gallery.py).

| | |
| :---: | :---: |
| ![Rainy platform](out/gallery/01_rainy_platform_s1234.png) | ![Bookshop cat](out/gallery/02_bookshop_cat_s42.png) |
| *"a girl in a yellow raincoat waiting on a train platform in the rain…"* <br> `seed=1234` | *"a cozy bookshop interior with warm lamps… an orange cat asleep on a stack of books"* <br> `seed=42` |
| ![Close-up portrait](out/gallery/03_portrait_closeup_s777.png) | ![Spring walk](out/gallery/04_spring_walk_s2024.png) |
| *"close-up portrait of an anime girl with short silver hair and violet eyes…"* <br> `seed=777` | *"a student walking to school on a spring morning, cherry blossom petals…"* <br> `seed=2024` |
| ![Night market](out/gallery/05_night_market_s1997.png) | ![Rooftop stars](out/gallery/06_rooftop_stars_s123.png) |
| *"a night market street with glowing paper lanterns, food stalls and steam…"* <br> `seed=1997` | *"an astronaut sitting on a rooftop at night looking up at a sky full of stars…"* <br> `seed=123` |

---

## Inference Benchmarks

Evaluated with the ExecuTorch C++ runtime on macOS CPU (Apple Silicon):

| Pipeline Stage | Method Name | Input Shape | Latency (CPU) |
| :--- | :--- | :--- | :--- |
| Text Encoding | `encode` | `int64[1, 77]` | **0.13 s** |
| 4-Step Denoising | `denoise` | `float32[1, 4, 64, 64]` × 4 | **22.09 s** (~5.5 s / step) |
| Latent Decoding | `decode` | `float32[1, 4, 64, 64]` | **12.06 s** |
| **Total Generation** | | | **34.27 s** |

*(On phones, multi-core XNNPACK threading lands well below these desktop-CPU numbers for a 3.8 GiB FP32 graph; the decode step dominates because it is a full-resolution VAE.)*

---

## Prompting Tips

Counterfeit V3.0 was trained on **BLIP-2 natural-language captions**, not booru tag soup. That changes how you should prompt it:

- **Write a sentence.** *"a girl reading by the window, afternoon light, plants on the sill"* works; `1girl, solo, masterpiece, best quality, 8k, ultra detailed` mostly adds noise.
- **Quality keywords do nothing here.** Skip the `masterpiece / best quality / absurdres` stack — the checkpoint is already tuned for clean illustration.
- **Clip skip 2 is baked in.** Do not try to configure it at runtime; `encode` already returns the correct normed state.
- **Guidance stays at 1.0.** This is a 4-step distilled export, so CFG is off by default. If you enable client-side CFG, keep $w \le 2.5$; higher values oversaturate the line art.
- **Name the medium when you want a specific look** — `watercolor`, `flat color`, `90s cel animation`, `soft digital painting` — the model follows style words more reliably than quality words.
- **512×512 native.** Anything larger is upscaled by the client, not generated.

---

## Known Limitations

- 4-step distillation trades fine detail for speed: **hands and fingers still slip**, and complex multi-character scenes can merge.
- No negative prompt in the default path (`guidance = 1.0`), so the model cannot be steered away from an unwanted element without client-side CFG.
- No text rendering — signage and lettering come out as glyph-like shapes.
- Anime/illustration only. Photorealistic prompts drift toward painted illustration.
- Licence and lineage: Counterfeit V3.0 is distributed under CreativeML Open RAIL-M and derives from earlier anime community checkpoints; review the upstream terms before commercial use.

---

## Repository Structure

```
counterfeit-anime-executorch-export/            # GitHub repository (code + docs)
├── README.md                                   # Hugging Face Model Card / Documentation
├── LICENSE                                     # CreativeML Open RAIL-M License
├── config.json                                 # RN ExecuTorch model descriptor
├── requirements.txt                            # Python dependencies for export & verification
├── download_counterfeit.sh                     # Weights + tokenizer download script
├── export_counterfeit_pte.py                   # PyTorch export & ExecuTorch lowering pipeline
├── verify_counterfeit_pte.py                   # ExecuTorch runtime verification script
├── eager_baseline.py                           # Eager PyTorch reference for parity
├── generate_gallery.py                         # Showcase image generation from the .pte
├── xnnpack/
│   └── config.json                             # Spec-conformant API contract for the .pte
├── models/                                     # gitignored — fetched via ./download_counterfeit.sh
│   ├── counterfeit/Counterfeit-V3.0_fix_fp16.safetensors   # 1.98 GiB (VAE baked in)
│   ├── hyper_sd/Hyper-SD15-4steps-lora.safetensors         # 256 MB (distillation LoRA)
│   └── swmansion_tokenizer/tokenizer.json                  # CLIP ViT-L/14 tokenizer
└── out/
    ├── counterfeit_anime_512_xnnpack_fp32.pte  # gitignored — 3,910.8 MB, hosted on Hugging Face
    ├── preview_counterfeit_xnnpack_fp32.png    # ExecuTorch runtime output
    ├── counterfeit_eager_baseline.png          # Eager PyTorch reference
    └── gallery/                                # Multi-prompt showcase (generate_gallery.py)
```

The Hugging Face repository contains the release artifacts, following the [react-native-executorch-spec](https://huggingface.co/software-mansion/react-native-executorch-spec) layout:

```
heymeowcat/counterfeit-anime-512-executorch/
├── README.md                                   # Model card (this file)
├── LICENSE                                     # CreativeML Open RAIL-M License
├── config.json                                 # RN ExecuTorch model descriptor
├── tokenizer.json                              # CLIP ViT-L/14 tokenizer
├── tokenizer_config.json                       # CLIPTokenizer config
├── xnnpack/
│   ├── config.json                             # Spec-conformant API contract (validated)
│   └── counterfeit_anime_512_xnnpack_fp32.pte  # 3,910.8 MB production binary
└── out/
    ├── preview_counterfeit_xnnpack_fp32.png
    ├── counterfeit_eager_baseline.png
    └── gallery/                                # Showcase images (rendered in the model card)
```

---

## Compatibility

Exported for the **ExecuTorch v1.4.1+** runtime using the **XNNPACK Floating Point Partitioner**.

To load it in React Native ExecuTorch, use the `useExecutorchModule` hook from [`react-native-executorch`](https://www.npmjs.com/package/react-native-executorch).

---

## Model Details & API Contract

The single `.pte` program defines three methods. The diffusion loop is orchestrated by the client:

### 1. `encode`
Computes CLIP prompt embeddings from tokenized prompt IDs, with **clip skip 2 and its final layer norm already applied**:
- **Inputs**: `tokens: int64[1, 77]`
- **Outputs**: `prompt_embeds: float32[1, 77, 768]`
- **Backend**: XNNPACK FP32

### 2. `denoise`
Computes the noise prediction $\epsilon_\theta(z_t, t, c)$ at each timestep:
- **Inputs**:
  1. `latents: float32[1, 4, 64, 64]`
  2. `timestep: int64[1]` (one of `[999, 759, 499, 259]`)
  3. `embeds: float32[1, 77, 768]`
- **Outputs**: `noise_pred: float32[1, 4, 64, 64]`
- **Backend**: XNNPACK FP32

### 3. `decode`
Reconstructs the 512×512 RGB image using the full AutoencoderKL decoder with proper color channel projection:
- **Inputs**: `latents: float32[1, 4, 64, 64]`
- **Internal Formula**:
  $$\text{image} = \text{clamp}\left(\frac{\text{decoder}(\text{post\_quant\_conv}(\text{latents} / 0.18215))}{2} + 0.5, 0.0, 1.0\right)$$
- **Outputs**: `image: float32[1, 3, 512, 512]` in range $[0.0, 1.0]$
- **Backend**: XNNPACK FP32

### Optional: Negative Prompt via Client-Side CFG

The exported program intentionally contains **no classifier-free guidance** — Hyper-SD 4-step distillation bakes guidance in, so `guidance = 1.0` is the default and fastest mode. To steer generations away from unwanted styles, apply CFG in the client (encode a negative prompt and run `denoise` twice per step):

$$\epsilon = \epsilon_{\text{neg}} + w \cdot (\epsilon_{\text{pos}} - \epsilon_{\text{neg}})$$

Useful negatives for this checkpoint: `photo, realistic, 3d render, lowres, blurry`. Keep $w \le 2.5$.

---

## Verification

Reproduce the parity check:

```bash
python eager_baseline.py            # writes out/counterfeit_eager_baseline.png
python verify_counterfeit_pte.py    # runs the .pte, compares pixel-by-pixel
```

Expected output:

```
Inference timing: encode=0.13s, 4-step denoise=22.09s, decode=12.06s, total=34.27s
Mean pixel diff vs eager FP32 baseline: 0.02/255
VERIFY_COUNTERFEIT_FP32_PTE_OK
```

---

## Licence

Released under the [CreativeML Open RAIL-M](LICENSE), inherited from the upstream Counterfeit V3.0 checkpoint. The Hyper-SD LoRA is distributed by ByteDance under its own upstream terms.
