"""Eager PyTorch reference for the Counterfeit V3.0 export.

Runs the graph the .pte was built from, step for step: Counterfeit V3.0 + fused
Hyper-SD15 4-step LoRA, clip skip 2, the four TCD timesteps [999, 759, 499, 259]
at eta 0.3, full VAE decode.

Why not just call `pipe(...)`? Because `StableDiffusionPipeline.__call__` does not
forward `eta` to `scheduler.step` (verified: the key is absent, so TCD runs at its
default 0.0), which silently produces a different trajectory — about 46/255 apart
from the schedule the app and the .pte actually use. The manual loop below is the
same thing `verify_counterfeit_pte.py` executes, so a low diff here means the
export is faithful, not that two different schedules agree.

Saves out/counterfeit_eager_baseline.png.
"""
import gc
import os
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, TCDScheduler
from transformers import CLIPTokenizer
import transformers
from transformers import CLIPImageProcessor

setattr(transformers, "CLIPFeatureExtractor", CLIPImageProcessor)
sys.modules["transformers"].CLIPFeatureExtractor = CLIPImageProcessor

torch.set_grad_enabled(False)

BASE_CHECKPOINT = "models/counterfeit/Counterfeit-V3.0_fix_fp16.safetensors"
LORA_4STEP = "models/hyper_sd/Hyper-SD15-4steps-lora.safetensors"
OUT_PNG = "out/counterfeit_eager_baseline.png"
CLIP_SKIP = 2
SEED = 1234
PROMPT = ("a girl with short black hair standing on a rooftop at sunset, "
          "warm orange sky, city skyline behind her")

print("[1/3] Loading Counterfeit V3.0 + fusing Hyper-SD 4-Step LoRA ...", flush=True)
pipe = StableDiffusionPipeline.from_single_file(
    os.path.abspath(BASE_CHECKPOINT),
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
)
pipe.load_lora_weights(os.path.abspath(LORA_4STEP))
pipe.fuse_lora(lora_scale=1.0)
pipe.unload_lora_weights()

text_encoder = pipe.text_encoder.eval()
unet = pipe.unet.eval()
post_quant_conv = pipe.vae.post_quant_conv.eval()
decoder = pipe.vae.decoder.eval()
scale = float(pipe.vae.config.scaling_factor)
del pipe
gc.collect()

# Same tokenizer file the verify script uses.
tok = CLIPTokenizer.from_pretrained("models/swmansion_tokenizer")
tokens = tok([PROMPT], max_length=77, padding="max_length", truncation=True,
             return_tensors="pt", return_attention_mask=False).input_ids.to(torch.int64)

# Clip skip 2, in the exact form the export bakes in: the chosen hidden state is
# pre-final-layer-norm, so the norm has to be applied here too.
_hidden = text_encoder(input_ids=tokens, output_hidden_states=True).hidden_states
embeds = (getattr(text_encoder, "final_layer_norm", None) or text_encoder.text_model.final_layer_norm)(
    _hidden[-CLIP_SKIP]
)

print("[2/3] Running the 4-step TCD loop ...", flush=True)
sched = TCDScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    clip_sample=False,
)
sched.set_timesteps(4)

latents = torch.randn(1, 4, 64, 64, generator=torch.Generator().manual_seed(SEED))
g_noise = torch.Generator().manual_seed(SEED)
for i, t in enumerate(sched.timesteps):
    eps = unet(latents, torch.tensor([t.item()], dtype=torch.int64), embeds).sample
    latents = sched.step(eps, t, latents, eta=0.3, generator=g_noise).prev_sample
    print(f"    step {i + 1}/4 (t={t.item()})", flush=True)

img = decoder(post_quant_conv(latents / scale))
img = (img / 2.0 + 0.5).clamp(0.0, 1.0)
arr = (img[0].movedim(0, 2) * 255).round().to(torch.uint8).numpy()

os.makedirs("out", exist_ok=True)
Image.fromarray(arr).save(OUT_PNG)
print(f"[3/3] Saved {OUT_PNG}", flush=True)
print("EAGER_BASELINE_OK", flush=True)
