"""Export Counterfeit V3.0 + Hyper-SD 4-Step + Full VAE Decoder to ExecuTorch (.pte)
in full unquantized FP32 for 100% mathematical parity with the eager PyTorch pipeline.
Zero quantization loss, zero time-embedding distortion.

Same shape of program as the epicrealism-512 export, with one deliberate difference:
Counterfeit V3.0 is an anime/illustration model that is trained and used with
**clip skip 2**, so the penultimate CLIP hidden state is baked into `encode` here.
The app offers a single plain prompt box and cannot change skip at runtime, so the
choice has to be frozen at export time.

Methods:
  encode:  int64[1, 77]                              -> float32[1, 77, 768]
  denoise: float32[1, 4, 64, 64], int64[1], float32[1, 77, 768] -> float32[1, 4, 64, 64]
  decode:  float32[1, 4, 64, 64]                   -> float32[1, 3, 512, 512] in [0, 1]
"""
import gc
import os
import sys
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline
import transformers
from transformers import CLIPImageProcessor

setattr(transformers, "CLIPFeatureExtractor", CLIPImageProcessor)
sys.modules["transformers"].CLIPFeatureExtractor = CLIPImageProcessor

torch.set_grad_enabled(False)

BASE_CHECKPOINT = "models/counterfeit/Counterfeit-V3.0_fix_fp16.safetensors"
LORA_4STEP = "models/hyper_sd/Hyper-SD15-4steps-lora.safetensors"
OUT_PTE = "out/counterfeit_anime_512_xnnpack_fp32.pte"
FP32 = torch.float32

# 1 uses the final layer (SD 1.5 default); 2 is what the anime scene expects and
# what Counterfeit's own sample images were made with.
CLIP_SKIP = 2

# ------------------------------------------------------------ 1) Load + Fuse LoRA
print("[1/5] Loading Counterfeit V3.0 + fusing Hyper-SD 4-Step LoRA (FP32) ...", flush=True)
pipe = StableDiffusionPipeline.from_single_file(
    os.path.abspath(BASE_CHECKPOINT),
    torch_dtype=FP32,
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
scaling_factor = float(pipe.vae.config.scaling_factor)
assert abs(scaling_factor - 0.18215) < 1e-6, f"unexpected VAE scaling factor {scaling_factor}"
del pipe.vae, pipe
gc.collect()

# ------------------------------------------------------------ 2) Module Definitions
class EncodeModule(nn.Module):
    """CLIP ViT-L/14 text encoder with clip skip frozen in.

    `hidden_states` is (embedding output, …one entry per layer), so `[-2]` is the
    output of the second-to-last layer — clip skip 2.

    The final layer norm MUST be applied to that state. `hidden_states` are
    pre-norm; skipping the norm leaves unnormalised activations (|x| up to ~800
    instead of ~30) and the UNet renders pure static. A1111 and ComfyUI both
    apply the norm when stopping early (`layer_norm_hidden_state`). It still
    returns float32[1, 77, 768], so the rest of the pipeline and the app are
    unchanged.
    """

    def __init__(self, te, skip: int):
        super().__init__()
        self.net = te
        self.skip = skip
        self.norm = getattr(te, "final_layer_norm", None) or te.text_model.final_layer_norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.net(input_ids=tokens, output_hidden_states=True).hidden_states
        return self.norm(hidden[-self.skip])


class DenoiseModule(nn.Module):
    def __init__(self, u):
        super().__init__()
        self.net = u

    def forward(self, latents: torch.Tensor, timestep: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
        out = self.net(latents, timestep, embeds)
        return out.sample if hasattr(out, "sample") else out


class DecodeModule(nn.Module):
    """Full VAE Decode: post_quant_conv -> decoder -> [0, 1] RGB."""
    def __init__(self, post_conv, dec, scale):
        super().__init__()
        self.post_quant_conv = post_conv
        self.decoder = dec
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(latents / self.scale)
        img = self.decoder(z)
        return (img / 2.0 + 0.5).clamp(0.0, 1.0)


encode_m = EncodeModule(text_encoder, CLIP_SKIP).eval()
denoise_m = DenoiseModule(unet).eval()
decode_m = DecodeModule(post_quant_conv, decoder, scaling_factor).eval()

# ------------------------------------------------------------ 3) Export Each Method
print("[2/5] Tracing methods with torch.export ...", flush=True)
tokens_ex = torch.ones((1, 77), dtype=torch.int64)
lat_ex = torch.randn(1, 4, 64, 64)
t_ex = torch.tensor([999], dtype=torch.int64)
emb_ex = encode_m(tokens_ex)

# Sanity: clip skip 2 must actually change the conditioning from the default.
with torch.no_grad():
    skip1 = text_encoder(input_ids=tokens_ex).last_hidden_state
delta = (emb_ex - skip1).abs().max().item()
print(f"    clip skip {CLIP_SKIP}: embeds {tuple(emb_ex.shape)}, max |skip2 - skip1| = {delta:.4f}", flush=True)
assert emb_ex.shape == (1, 77, 768), emb_ex.shape
assert delta > 1e-3, "clip skip had no effect — the wrapper is not doing what it claims"
# Pre-norm states run to ~±800 and turn the UNet output into static.
assert emb_ex.abs().max() < 100, f"clip-skip state looks unnormalised (|x|max={emb_ex.abs().max():.1f})"

ep_encode = torch.export.export(encode_m, (tokens_ex,))
ep_denoise = torch.export.export(denoise_m, (lat_ex, t_ex, emb_ex))
ep_decode = torch.export.export(decode_m, (lat_ex,))
print("    Exported encode, denoise, decode successfully", flush=True)

del text_encoder, unet, post_quant_conv, decoder
gc.collect()

# ------------------------------------------------------------ 4) Lower to XNNPACK
print("[3/5] Lowering to ExecuTorch with XNNPACK FP32 partitioner ...", flush=True)
from executorch.exir import to_edge_transform_and_lower
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackFloatingPointPartitioner

edge = to_edge_transform_and_lower(
    {"encode": ep_encode, "denoise": ep_denoise, "decode": ep_decode},
    partitioner={
        "encode": [XnnpackFloatingPointPartitioner()],
        "denoise": [XnnpackFloatingPointPartitioner()],
        "decode": [XnnpackFloatingPointPartitioner()],
    },
    compile_config=None,
)

print("[4/5] Serializing to ExecuTorch binary ...", flush=True)
et = edge.to_executorch()
os.makedirs(os.path.dirname(OUT_PTE), exist_ok=True)
with open(OUT_PTE, "wb") as f:
    f.write(et.buffer)

size_mb = os.path.getsize(OUT_PTE) / 1024 / 1024
print(f"[5/5] Successfully wrote {OUT_PTE} ({size_mb:.1f} MB)", flush=True)
print("EXPORT_COUNTERFEIT_FP32_OK", flush=True)
