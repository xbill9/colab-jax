# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Gemma 4 E2B on JAX: Google's W4A16 export against an exact repack of the QAT weights
#
# **Author:** William McLean, Google Developer Expert
#
# **Based on:** the E2B measurements in
# [`gemma4-dev/QUANTIZATION.md`](https://github.com/xbill9/gemma4-dev/blob/main/QUANTIZATION.md)
# and the repack's model card,
# [`xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct`](https://huggingface.co/xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct).
#
# ### Who this notebook is for
#
# JAX developers who load quantized checkpoints, and engineers who choose which 4-bit build of a
# model to serve. Terms are explained where they first appear. Notebook 01 covers the JAX mechanics
# used here, but this notebook does not depend on it.
#
# ### The study in one paragraph
#
# Google trained Gemma 4 E2B with quantization-aware training (QAT): during training, every weight
# was held on a 4-bit grid, so the model learned to work with exactly those values. Google publishes
# the result twice. `-qat-q4_0-unquantized` stores the grid values in bf16. `-qat-w4a16-ct` packs
# them as int4 for serving, but its recipe re-rounds every group of 32 weights onto a new grid whose
# step is the group's largest weight divided by 7.5. That step never equals the trained one, so every
# weight moves. The repack stores the trained levels and the trained step instead. Served by vLLM on
# one v5e chip, the two 4-bit builds run at the same speed, and the repack scores 2.4 points higher
# on a 3,880-record test suite (95% range +1.4 to +3.4), within 0.6 points of bf16.
#
# ### What this notebook does
#
# It takes all three checkpoints and asks, on your chip, with a pure-JAX engine:
#
# 1. **Are the weights the trained ones?** Compare each 4-bit build with the QAT values, tensor by
#    tensor.
# 2. **Does the difference reach the output?** Run all three through the same forward pass on the
#    same text and compare each 4-bit model's next-token distribution with the QAT model's.
# 3. **What does each cost?** Disk, HBM and decode speed.
#
# ### The three rules used throughout
#
# 1. **Every result is measured here.** Numbers from the source work appear beside yours as context,
#    never in place of them.
# 2. **Only the weights change.** All three checkpoints go through the same loader, the same forward
#    pass, the same text and the same prompt.
# 3. **The QAT checkpoint is the reference.** Both 4-bit builds claim to hold the QAT model, so each
#    is compared with `-qat-q4_0-unquantized`, not with the other.
#
# ### How each section is organized
#
# * **What the source measured:** the earlier result and where it was measured.
# * **What we are testing:** the claim, and the evidence that checks it.
# * A code cell to run.
# * **What you should see:** the expected result, and what a different one would mean.
#
# ### Before you start
#
# * **Runtime.** Runtime, then Change runtime type, then TPU. The target is a single v5e chip
#   (16 GB of HBM). Everything also fits on v6e.
# * **No token.** All three checkpoints are public and ungated, and the engine is a public GitHub
#   repo. Nothing asks for a Hugging Face token.
# * **Disk and time.** About 26 GB of downloads. The whole notebook takes 20 to 40 minutes, most of
#   it downloading and compiling.
# * **Safety.** Nothing is uploaded and nothing outside the Colab VM changes.

# %% [markdown]
# ## 1. Setup and version check
#
# **What we are testing.** Nothing yet. This section records the chip, the JAX version and the room
# available, because every number below depends on them.
#
# The engine is [`tpu-jax`](https://github.com/xbill9/tpu-jax), a Gemma 4 E2B decoder in pure JAX
# with no PyTorch in the path. Cell 1.2 clones it at a fixed commit, so the code you run is the code
# this notebook was written against. Its 4-bit path dequantizes each weight with plain XLA ops and
# then multiplies; no custom kernel is involved.

# %%
# Cell 1.1: chip, memory and disk
import gc
import os
import pathlib
import shutil
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

print("JAX", jax.__version__)
dev = jax.devices()[0]
print(f"{len(jax.devices())} device(s): {dev.device_kind} ({dev.platform})")
if dev.platform != "tpu":
    raise RuntimeError("This notebook needs a TPU runtime: Runtime > Change runtime type > TPU")

HBM_LIMIT = dev.memory_stats()["bytes_limit"]
host_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
disk_free = shutil.disk_usage(os.path.expanduser("~")).free
print(f"HBM {HBM_LIMIT / 1e9:.2f} GB   host RAM {host_ram / 1e9:.1f} GB   free disk {disk_free / 1e9:.0f} GB")
assert disk_free > 30e9, "need about 30 GB free disk for the three checkpoints"


def host_used_gb():
    """Host RAM in use on the whole VM, as Colab's RAM meter counts it."""
    info = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    kb = int(info["MemTotal"].split()[0]) - int(info["MemAvailable"].split()[0])
    return kb / 1e6


# %%
# Cell 1.2: the pure-JAX engine, pinned, and the readers it needs
import importlib.util

missing = [
    p for p in ("safetensors", "huggingface_hub", "tokenizers", "datasets") if importlib.util.find_spec(p) is None
]
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)

TPU_JAX_COMMIT = "4b9f8e94619f1995206617e07e3c4ac62685d47b"
ENGINE_DIR = pathlib.Path("/content/tpu-jax" if os.path.isdir("/content") else os.path.expanduser("~/tpu-jax-src"))
if not (ENGINE_DIR / ".git").exists():
    subprocess.run(["git", "clone", "-q", "https://github.com/xbill9/tpu-jax", str(ENGINE_DIR)], check=True)
subprocess.run(["git", "-C", str(ENGINE_DIR), "checkout", "-q", TPU_JAX_COMMIT], check=True)
sys.path.insert(0, str(ENGINE_DIR))

from jax_engine import JaxGemmaEngine  # noqa: E402
from ports.gemma4.jax_e_model import make_prefill_causal_mask, qat_w4a16_unpack_dequant_jax  # noqa: E402

print(
    "tpu-jax at",
    subprocess.run(
        ["git", "-C", str(ENGINE_DIR), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip(),
)

# %% [markdown]
# **What you should see.** One device whose kind contains `v5 lite` (or `v6`), about 16.9 GB of HBM
# on v5e, and `tpu-jax at 4b9f8e9`. If the device is not a TPU, the first cell stops.

# %% [markdown]
# ## 2. The three checkpoints
#
# **What the source measured.** Google's E2B W4A16 export is 8.32 GB and the repack 7.51 GB. Most of
# the difference is one tensor: Google's export stores `lm_head.weight`, a byte-for-byte copy of the
# token embedding, although the config ties the two (`tie_word_embeddings: true`). The engine never
# reads it, but it is downloaded, and on vLLM it was resident: 0.75 GiB of HBM.
#
# **What we are testing.** The size of each download, and whether Google's `lm_head` is identical to
# the embedding it duplicates. Each checkpoint is pinned to a Hub revision, so the bytes you
# download are the bytes measured here.

# %%
# Cell 2.1: download the three checkpoints (about 26 GB; 5 to 15 minutes)
# Each download runs in its own Python process. The downloader can hold several GB of host RAM
# after it finishes, and the VM only gets that back when the process exits. Downloading inside
# the notebook left too little RAM to load the checkpoints later.
DOWNLOAD = """
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], allow_patterns=["*.safetensors", "*.json"]))
"""

CHECKPOINTS = {
    "qat": ("google/gemma-4-E2B-it-qat-q4_0-unquantized", "6befbaca7398925921802abd1f277b495b78b738"),
    "stock": ("google/gemma-4-E2B-it-qat-w4a16-ct", "971342c08f607aa7779983f6b5289778b5d271a7"),
    "repack": ("xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct", "da64116ff640f56cddb2d4fe20f31e816da22ce9"),
}
paths, disk_bytes = {}, {}
for name, (repo, rev) in CHECKPOINTS.items():
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, "-c", DOWNLOAD, repo, rev], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"download of {repo} failed:\n{proc.stderr[-2000:]}")
    paths[name] = proc.stdout.strip().splitlines()[-1]
    disk_bytes[name] = sum(f.stat().st_size for f in pathlib.Path(paths[name]).glob("*.safetensors"))
    print(f"{name:7s} {repo:45s} {disk_bytes[name] / 1e9:6.2f} GB  ({time.perf_counter() - t0:.0f} s)")
print(f"\nhost RAM in use: {host_used_gb():.1f} of {host_ram / 1e9:.1f} GB")

# %%
# Cell 2.2: what each file holds, and the duplicate lm_head
from safetensors import safe_open


def tensor_index(path):
    """Map every tensor name in a checkpoint directory to the shard that holds it."""
    out = {}
    for f in sorted(pathlib.Path(path).glob("*.safetensors")):
        with safe_open(f, framework="flax") as h:
            out.update(dict.fromkeys(h.keys(), f))
    return out


def read(name, key):
    with safe_open(index[name][key], framework="flax") as h:
        return h.get_tensor(key)


index = {name: tensor_index(p) for name, p in paths.items()}
EMBED = "model.language_model.embed_tokens.weight"
for name in CHECKPOINTS:
    packed = sum(k.endswith(".weight_packed") for k in index[name])
    print(
        f"{name:7s} {len(index[name]):5d} tensors   {packed:3d} packed int4   lm_head stored: {'lm_head.weight' in index[name]}"
    )

lm_head, embed = read("stock", "lm_head.weight"), read("stock", EMBED)
dup_bytes = lm_head.size * lm_head.dtype.itemsize
print(
    f"\nstock lm_head identical to embed_tokens: {bool(jnp.array_equal(lm_head, embed))}   ({dup_bytes / 1e9:.3f} GB)"
)
print(f"stock minus repack on disk: {(disk_bytes['stock'] - disk_bytes['repack']) / 1e9:.3f} GB")
del lm_head, embed

# %% [markdown]
# **What you should see.** Both 4-bit builds hold the same number of packed tensors, because they
# quantize the same set of linear layers. Only the stock build stores `lm_head`, and it is identical
# to `embed_tokens`. That one tensor (0.805 GB) accounts for the size difference between the two
# builds. The bf16 QAT checkpoint is the largest, and it is still 4-bit data, as
# Section 3 shows.

# %% [markdown]
# ## 3. Are the weights the trained ones?
#
# **What the source measured.** Across 24 E2B tensors from the first, middle and last layers (every
# projection kind, 5.7 million groups of 32), Google's scale was max|w| / 7.5 in 100.00% of groups
# and equal to the trained step in 0.000%. Its values carried 6.65 to 6.67% relative error against
# the QAT values. The repack was checked the same way on the 12B, where none of 340,623,360 groups
# fell off the grid.
#
# **What we are testing.** The same 24 tensors, read from all three files and compared on the chip.
# For each 4-bit build, the cell dequantizes the packed weights with the engine's own function (the
# one the forward pass uses) and compares the result with the QAT values:
#
# * **relative error:** size of the difference, divided by size of the weights;
# * **identical:** share of values that are bit-for-bit the QAT value;
# * **step = max/7.5:** share of groups whose scale is the largest weight divided by 7.5, the rule
#   Google's recipe records (`observer: memoryless_minmax`);
# * **peak on a level:** share of groups whose largest weight lands on a whole level of the stored
#   step. On the trained grid every weight is step × level, with level from −8 to 7.
#
# The mechanism in one line: if a group's largest weight sits on level *m* of the trained step *d*,
# the min-max step is *m·d* / 7.5. Since *m* is a whole number from 1 to 8, that never equals *d*, so
# every weight in the group is rounded a second time onto a grid the model never trained on.

# %%
# Cell 3.1: the grid test, 24 tensors
LP = "model.language_model."
TENSORS = [LP + "per_layer_model_projection"]
for layer in (0, 17, 34):
    projs = [
        "self_attn.q_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        "per_layer_input_gate",
        "per_layer_projection",
    ]
    if layer < 15:  # layers 15-34 reuse an earlier layer's keys and values, so they have no k/v
        projs += ["self_attn.k_proj", "self_attn.v_proj"]
    TENSORS += [f"{LP}layers.{layer}.{p}" for p in projs]


@jax.jit
def grid_stats(qat, packed, scale):
    """Compare one packed W4A16 tensor with the QAT values it should hold."""
    src = qat.astype(jnp.float32)
    deq = qat_w4a16_unpack_dequant_jax(packed, scale).astype(jnp.float32)
    peak = jnp.max(jnp.abs(src.reshape(src.shape[0], -1, 32)), axis=-1)  # [out, groups]
    s = scale.astype(jnp.float32)
    live = peak > 0
    level = jnp.where(live, peak / s, 0.0)
    on_level = live & (jnp.abs(level - jnp.round(level)) < 0.05)
    return {
        "groups": jnp.sum(live),
        "values": src.size,
        "sq_err": jnp.sum((deq - src) ** 2),
        "sq_src": jnp.sum(src**2),
        "identical": jnp.sum(deq == src),
        "minmax": jnp.sum(live & (jnp.abs(s * 7.5 / jnp.where(live, peak, 1.0) - 1.0) <= 2.0**-8)),
        "on_level": jnp.sum(on_level),
        "peak_level": jnp.bincount(jnp.where(on_level, jnp.round(level), 0).astype(jnp.int32).ravel(), length=9),
    }


totals = {b: None for b in ("stock", "repack")}
print(f"{'tensor':48s} {'build':7s} {'rel err':>8s} {'identical':>10s} {'step=max/7.5':>13s} {'peak on level':>14s}")
for t in TENSORS:
    qat = read("qat", t + ".weight")
    for build in totals:
        r = jax.device_get(grid_stats(qat, read(build, t + ".weight_packed"), read(build, t + ".weight_scale")))
        totals[build] = r if totals[build] is None else {k: totals[build][k] + r[k] for k in r}
        print(
            f"{t[len(LP) :]:48s} {build:7s} {np.sqrt(r['sq_err'] / r['sq_src']):8.4f} "
            f"{r['identical'] / r['values']:10.2%} {r['minmax'] / r['groups']:13.2%} {r['on_level'] / r['groups']:14.2%}"
        )

print("\nall 24 tensors")
grid = {}
for build, r in totals.items():
    grid[build] = {
        "groups": int(r["groups"]),
        "rel_err": float(np.sqrt(r["sq_err"] / r["sq_src"])),
        "identical": float(r["identical"] / r["values"]),
        "minmax": float(r["minmax"] / r["groups"]),
        "on_level": float(r["on_level"] / r["groups"]),
    }
    g = grid[build]
    print(
        f"  {build:7s} groups {g['groups']:,}   rel err {g['rel_err']:.4f}   identical {g['identical']:.2%}   "
        f"step=max/7.5 {g['minmax']:.2%}   peak on a level {g['on_level']:.2%}"
    )
levels = totals["repack"]["peak_level"]
print("\n  repack: level the largest weight of each group sits on, share of groups")
print("   " + "   ".join(f"{m}: {levels[m] / levels.sum():.1%}" for m in range(1, 9)))

# %% [markdown]
# **What you should see.** For the stock build: relative error near 0.067 on every tensor, about one
# value in eight identical to the QAT value (the ones that happen to land on both grids), `step=max/7.5`
# at 100%, and no group whose peak lands on a whole level. That is the signature of re-rounding by
# min-max. For the repack: relative error near 0.002, about three values in four identical, and the
# peak of nearly every group on a whole level. The values that differ do so by the bf16 rounding of
# the stored step, a fraction of a percent, against 6.7% for the stock build.
#
# The last line shows why no fixed rule recovers the trained step: the peak sits on level 7 in some
# groups and level 8 in others. A rule that assumes level 8 (max/8) is right for the second kind
# only, and min-max (max/7.5) is right for neither.

# %% [markdown]
# ## 4. Does the difference reach the output?
#
# **What the source measured.** On a 3,880-record test suite, read by label probability and paired
# record for record on one v5e chip under vLLM: repack 67.8%, Google's export 65.5%, a difference of
# +2.4 points (95% range +1.4 to +3.4). The bf16 E2B release scored 68.5% on the same chip.
#
# **What we are testing.** A suite of that size is too slow for a notebook, so this section measures
# something more direct. It feeds the same WikiText-2 text through all three models and compares,
# at every position, each 4-bit model's predicted next-token distribution with the QAT model's:
#
# * **KL divergence:** how far the 4-bit distribution is from the QAT one. Zero means identical.
# * **same top token:** share of positions where both models' most likely token is the same.
# * **perplexity ratio:** the 4-bit model's perplexity on the text, divided by the QAT model's.
#   Above 1 means it predicts the real text less well.
#
# All three use the same engine, loader, forward pass and causal mask; the 4-bit builds run packed,
# as shipped. Each model is loaded, measured and released before the next, because two do not fit
# in 16 GB at once. The QAT model's distributions are kept in host memory (8 × 512 positions, 4.3 GB)
# for the comparison.

# %%
# Cell 4.1: the text, and the measurement helpers
from datasets import load_dataset
from tokenizers import Tokenizer

N_SEQ, SEQ_LEN, NEW_TOKENS = 8, 512, 128
tok = Tokenizer.from_file(os.path.join(paths["qat"], "tokenizer.json"))
BOS = tok.token_to_id("<bos>")
assert BOS is not None, "tokenizer has no <bos> token"

wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok.encode("".join(wiki["text"]), add_special_tokens=False).ids
assert len(ids) >= N_SEQ * (SEQ_LEN - 1), "not enough WikiText-2 tokens"
batch = np.array([[BOS] + ids[i * (SEQ_LEN - 1) : (i + 1) * (SEQ_LEN - 1)] for i in range(N_SEQ)], dtype=np.int32)
PROMPT = tok.encode("The history of the Roman Empire", add_special_tokens=False).ids
print(f"{len(ids):,} WikiText-2 test tokens; using {N_SEQ} sequences of {SEQ_LEN}")

ref_logprobs = []  # the QAT model's log-probabilities, one [SEQ_LEN, vocab] array per sequence
results = {}


def load(name):
    engine = JaxGemmaEngine(CHECKPOINTS[name][0], quant_mode="fp16" if name == "qat" else "w4a16")
    t0 = time.perf_counter()
    engine.load(local_dir=paths[name])
    engine.bos_token_id = BOS
    print(
        f"{name}: loaded in {time.perf_counter() - t0:.0f} s, {engine.weight_bytes / 1e9:.2f} GB of weights on the chip, "
        f"host RAM in use {host_used_gb():.1f} GB"
    )
    return engine


def logprob_fn(engine):
    valid = jnp.ones((1, SEQ_LEN), dtype=bool)
    full, sliding = (
        make_prefill_causal_mask(valid),
        make_prefill_causal_mask(valid, window=engine.config.sliding_window),
    )
    pos = jnp.arange(SEQ_LEN, dtype=jnp.int32)[None, :]

    @jax.jit
    def fn(params, x):
        logits = engine.model(
            x, params, pos, attention_mask=full, sliding_attention_mask=sliding, quant_mode=engine.quant_mode
        )
        return jax.nn.log_softmax(logits[0].astype(jnp.float32), axis=-1)

    return fn


@jax.jit
def compare(ref, lp):
    """Per-position KL(ref || lp) and whether the top tokens agree."""
    return jnp.sum(jnp.exp(ref) * (ref - lp), axis=-1), jnp.argmax(ref, -1) == jnp.argmax(lp, -1)


def measure(name):
    engine = load(name)
    fn = logprob_fn(engine)
    nll, kl, same = [], [], []
    t0 = time.perf_counter()
    for i in range(N_SEQ):
        x = jnp.asarray(batch[i : i + 1])
        lp = fn(engine.params, x)
        nll.append(np.asarray(-jnp.take_along_axis(lp[:-1], x[0, 1:, None], axis=1)[:, 0]))
        if name == "qat":
            ref_logprobs.append(np.asarray(lp))
        else:
            k, s = compare(jnp.asarray(ref_logprobs[i]), lp)
            kl.append(np.asarray(k))
            same.append(np.asarray(s))
    print(f"{name}: {N_SEQ} x {SEQ_LEN} positions scored in {time.perf_counter() - t0:.0f} s (first one compiles)")

    # Greedy decode of a fixed prompt: once to compile, then three timed runs. No stop token, so every
    # run generates exactly NEW_TOKENS and all three models do the same work.
    engine.generate(PROMPT, max_new_tokens=NEW_TOKENS)
    runs = [engine.generate(PROMPT, max_new_tokens=NEW_TOKENS) for _ in range(3)]
    results[name] = {
        "nll": np.concatenate(nll),
        "kl": np.concatenate(kl) if kl else None,
        "same": np.concatenate(same) if same else None,
        "tok_s": float(np.median([stats.decode_tok_per_s for _, stats in runs])),
        "greedy": runs[0][0],
        "weight_bytes": engine.weight_bytes,
    }
    del engine, fn
    gc.collect()
    print(
        f"{name}: released, {dev.memory_stats()['bytes_in_use'] / 1e9:.2f} GB still in use on the chip, "
        f"host RAM in use {host_used_gb():.1f} GB"
    )


# %%
# Cell 4.2: the QAT reference (bf16, 9 to 10 GB on the chip)
measure("qat")

# %%
# Cell 4.3: Google's W4A16 export
measure("stock")

# %%
# Cell 4.4: the repack
measure("repack")

# %%
# Cell 4.5: each 4-bit build against the QAT model, on the same positions
ref_nll = results["qat"]["nll"].mean()
print(
    f"{'build':7s} {'mean KL':>9s} {'same top token':>15s} {'perplexity ratio':>17s}   per-sequence mean KL, min to max"
)
for b in ("stock", "repack"):
    r = results[b]
    per_seq = r["kl"].reshape(N_SEQ, SEQ_LEN).mean(axis=1)
    r["kl_mean"], r["same_top"] = float(r["kl"].mean()), float(r["same"].mean())
    r["ppl_ratio"] = float(np.exp(r["nll"].mean() - ref_nll))
    print(
        f"{b:7s} {r['kl_mean']:9.5f} {r['same_top']:15.2%} {r['ppl_ratio']:17.4f}   {per_seq.min():.5f} to {per_seq.max():.5f}"
    )

closer = float(np.mean(results["repack"]["kl"] < results["stock"]["kl"]))
print(
    f"\npositions where the repack is closer to the QAT model than the stock build: {closer:.1%} "
    f"of {results['repack']['kl'].size:,}"
)
print(f"stock KL / repack KL: {results['stock']['kl_mean'] / results['repack']['kl_mean']:.1f}x")

# %% [markdown]
# **What you should see.** The repack's KL divergence far below the stock build's, with a higher
# share of matching top tokens and a perplexity ratio closer to 1. The repack holds the QAT values, so
# its remaining difference comes from the bf16 rounding of its scales and from the order in which the
# packed path multiplies. The stock build's difference is the second rounding Section 3 measured, carried through 35
# layers. The per-sequence range shows whether the result holds on every sequence.
#
# If the two builds came out level, check Section 3 first: a stock build that re-rounds every
# weight and a repack that stores them exactly cannot give the same distributions.
#
# The suite result in the source is the same finding measured on tasks: 2.4 points from a difference
# in weights alone.

# %% [markdown]
# ## 5. Speed and memory
#
# **What the source measured.** Under vLLM on one v5e chip, output tokens per second at 1, 4 and 16
# concurrent requests: 136.6 / 532.3 / 1,911 for Google's export and 136.5 / 532.1 / 1,910 for the
# repack. Weights resident: 7.17 GiB against 6.42 GiB, the difference being the duplicate `lm_head`.
#
# **What we are testing.** Decode speed for one request in this engine, and the bytes each build puts
# on the chip. Both 4-bit builds have the same format and the same shapes, so they should cost the
# same per token. The bf16 QAT model is included for scale.

# %%
# Cell 5.1: decode speed, memory, and what each one wrote
ref_greedy = results["qat"]["greedy"]
print(f"{'build':7s} {'disk GB':>8s} {'weights GB':>11s} {'decode tok/s':>13s} {'tokens equal to QAT greedy':>27s}")
for b in CHECKPOINTS:
    r = results[b]
    match = next((i for i, (a, c) in enumerate(zip(r["greedy"], ref_greedy)) if a != c), len(ref_greedy))
    r["greedy_match"] = match
    print(
        f"{b:7s} {disk_bytes[b] / 1e9:8.2f} {r['weight_bytes'] / 1e9:11.2f} {r['tok_s']:13.1f} {match:>20d} of {len(ref_greedy)}"
    )
print(f"\nrepack / stock decode speed: {results['repack']['tok_s'] / results['stock']['tok_s']:.3f}x")
for b in CHECKPOINTS:
    print(f"\n--- {b} ---\n{tok.decode(results[b]['greedy'][:60])}")

# %% [markdown]
# **What you should see.** The two 4-bit builds within a few percent of each other in decode speed,
# with the bf16 QAT model faster than both at one request. This engine dequantizes every packed
# weight at every step, which costs more than reading bf16 at this size. The engine can instead
# dequantize once at load (`dequant_at_load=True`), which trades the HBM saving for that speed. The
# weights on the chip are the same for both 4-bit builds, because the engine never loads `lm_head`.
# The disk column still carries it.
#
# The greedy text is one prompt, so read it as an illustration. A build that follows the QAT model
# for more tokens before diverging is closer to it, which is what Section 4 measures over 4,096
# positions.

# %% [markdown]
# ## 6. Scorecard
#
# **What we are testing.** Every source result next to the one you just measured. The source
# columns came from vLLM on a v5e chip; yours came from a pure-JAX engine on whatever chip you have.
# Each pair of columns puts the two builds side by side, so read across a pair: stock against
# repack. The claim under test is the direction of each difference, which should hold on any chip.
#
# The repack is the better build where the weights differ: fidelity to the QAT model and download
# size. Speed and memory on the chip should tie, because both builds store the same 4-bit format in
# the same shapes and do the same work per token.

# %%
# Cell 6.1: the scorecard
s, r = results["stock"], results["repack"]
rows = [
    # (row, stock: source, repack: source, stock: today, repack: today)
    (
        "relative error vs QAT values",
        "0.0665-0.0667",
        "n/a",
        f"{grid['stock']['rel_err']:.4f}",
        f"{grid['repack']['rel_err']:.4f}",
    ),
    (
        "groups with step = max/7.5",
        "100.00%",
        "n/a",
        f"{grid['stock']['minmax']:.2%}",
        f"{grid['repack']['minmax']:.2%}",
    ),
    (
        "groups with peak on a level",
        "n/a",
        "n/a",
        f"{grid['stock']['on_level']:.2%}",
        f"{grid['repack']['on_level']:.2%}",
    ),
    ("mean KL vs QAT model", "n/a", "n/a", f"{s['kl_mean']:.5f}", f"{r['kl_mean']:.5f}"),
    ("same top token as QAT model", "n/a", "n/a", f"{s['same_top']:.2%}", f"{r['same_top']:.2%}"),
    ("perplexity ratio vs QAT model", "n/a", "n/a", f"{s['ppl_ratio']:.4f}", f"{r['ppl_ratio']:.4f}"),
    ("test suite, 3,880 records", "65.5%", "67.8%", "not run", "not run"),
    ("download, GB", "8.32", "7.51", f"{disk_bytes['stock'] / 1e9:.2f}", f"{disk_bytes['repack'] / 1e9:.2f}"),
    ("decode tok/s, 1 request", "136.6", "136.5", f"{s['tok_s']:.1f}", f"{r['tok_s']:.1f}"),
    ("decode speed, repack / stock", "", f"{136.5 / 136.6:.3f}x", "", f"{r['tok_s'] / s['tok_s']:.3f}x"),
]
print(f"{'':34s} {'source (vLLM)':^29s} {'today (pure JAX)':^29s}")
print(f"{'':34s} {'stock':>14s} {'repack':>14s} {'stock':>14s} {'repack':>14s}")
for row in rows:
    print(f"{row[0]:34s} {row[1]:>14s} {row[2]:>14s} {row[3]:>14s} {row[4]:>14s}")

# %% [markdown]
# **What you should see.** The weight and output rows favor the repack by a wide margin, and the
# download row favors it by the duplicate `lm_head`. The speed ratio sits near 1.00x in both pairs:
# a tie, as the format predicts. The absolute speeds differ between the pairs because the engines
# differ: vLLM's int4 path against a reference dequantize-then-multiply in pure JAX. That gap says
# nothing about either build, so compare stock with repack inside a pair, never across pairs.
#
# **So, which one?** The repack. It holds the weights Google trained, runs at the same speed in the
# same format, and is smaller to download. Any loader that reads Google's `-qat-w4a16-ct` reads it
# unchanged.

# %% [markdown]
# ## 7. Your notes
#
# Record where your numbers differ from the ones above, with the evidence. Suggested questions:
#
# * **Chip.** Which chip and JAX version (Cell 1.1)? The ratios should carry across chips; the
#   speeds will not.
# * **Weights.** Did Section 3 reproduce the stock build's 0.067 relative error and the repack's
#   exact grid?
# * **Output.** How many times larger was the stock build's KL divergence than the repack's?
# * **Speed.** Were the two 4-bit builds level?

# %%
# Cell 7.1: your notes (edit this cell)
notes = """
Chip and JAX:
Weights (Section 3):
Output (Section 4):
Speed (Section 5):
Other findings:
"""
print(notes)
