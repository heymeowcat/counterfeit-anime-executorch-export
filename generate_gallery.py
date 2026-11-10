"""Generate a showcase gallery from out/counterfeit_anime_512_xnnpack_fp32.pte
using the ExecuTorch C++ runtime (4-step TCD schedule, eta=0.3).

Prompting follows the Counterfeit V3.0 author's advice: the checkpoint was trained
with BLIP-2 captions, so **plain natural-language scene descriptions** work better
than booru tag soup. No "masterpiece / 8k / best quality" keywords.

Guidance stays at 1.0: this export is Hyper-SD 4-step distilled, so CFG must stay
low, and guidance=1.0 disables the negative branch entirely and runs fastest.

Classifier-free guidance is done client-side: when guidance > 1.0, `denoise`
is executed twice per step (positive + negative embeddings) and combined as
    eps = eps_neg + guidance * (eps_pos - eps_neg)

Clip skip 2 is baked into `encode`, so nothing to configure here.

Saves one PNG per entry to out/gallery/.
"""
import os
import time
import torch
from PIL import Image
from transformers import CLIPTokenizer

import executorch.backends.xnnpack  # registers XNNPACK delegate
from executorch.runtime import Runtime
from diffusers import TCDScheduler

PTE = "out/counterfeit_anime_512_xnnpack_fp32.pte"
OUT_DIR = "out/gallery"

PROMPTS = [
    ("01_rainy_platform_s1234", 1234, 1.0,
     "a girl in a yellow raincoat waiting on a train platform in the rain, "
     "city lights reflected in the puddles, umbrella in hand"),
    ("02_bookshop_cat_s42", 42, 1.0,
     "a cozy bookshop interior with warm lamps, tall wooden shelves, "
     "an orange cat asleep on a stack of books"),
    ("03_portrait_closeup_s777", 777, 1.0,
     "close-up portrait of an anime girl with short silver hair and violet eyes, "
     "gentle smile, soft rim light, blurred city lights behind her"),
    ("04_spring_walk_s2024", 2024, 1.0,
     "a student walking to school on a spring morning, cherry blossom petals "
     "floating in the air, quiet residential street"),
    ("05_night_market_s1997", 1997, 1.0,
     "a night market street with glowing paper lanterns, food stalls and steam "
     "rising from a ramen stand, a few people browsing"),
    ("06_rooftop_stars_s123", 123, 1.0,
     "an astronaut sitting on a rooftop at night looking up at a sky full of "
     "stars, one shooting star, quiet city below"),
]

print("--- Initializing ExecuTorch Runtime ---", flush=True)
rt = Runtime.get()
program = rt.load_program(PTE)
enc = program.load_method("encode")
den = program.load_method("denoise")
dec = program.load_method("decode")
print("Methods:", sorted(program.method_names), flush=True)

tok = CLIPTokenizer.from_pretrained("models/swmansion_tokenizer")


def tokenize(text):
    return tok([text], max_length=77, padding="max_length", truncation=True,
               return_tensors="pt", return_attention_mask=False).input_ids.to(torch.int64)


sched = TCDScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    set_alpha_to_one=False,
    clip_sample=False,
)

os.makedirs(OUT_DIR, exist_ok=True)
for name, seed, guidance, prompt in PROMPTS:
    t_start = time.time()

    # Reset TCD scheduler state (ets history + counter) for every generation
    sched.set_timesteps(4)

    embeds = enc.execute((tokenize(prompt),))[0]

    g = torch.Generator().manual_seed(seed)
    latents = torch.randn(1, 4, 64, 64, generator=g)
    g_noise = torch.Generator().manual_seed(seed)

    with torch.no_grad():
        for t in sched.timesteps:
            t_tensor = torch.tensor([t.item()], dtype=torch.int64)
            eps = den.execute((latents, t_tensor, embeds))[0]
            latents = sched.step(eps, t, latents, eta=0.3, generator=g_noise).prev_sample

        image = dec.execute((latents,))[0]

    arr = (image[0].movedim(0, 2).clamp(0, 1) * 255).round().to(torch.uint8).numpy()
    path = os.path.join(OUT_DIR, f"{name}.png")
    Image.fromarray(arr).save(path)
    print(f"[done] {path}  seed={seed} guidance={guidance}  {time.time() - t_start:.1f}s", flush=True)

print("GENERATE_GALLERY_OK", flush=True)
