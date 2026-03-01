# -*- coding: utf-8 -*-
import subprocess
import sys

# Install required libraries
# Run once; restart interpreter if needed after installation
subprocess.run(
    [sys.executable, "-m", "pip", "install",
     "transformers", "torch", "matplotlib", "seaborn", "--quiet"],
    check=True,
)

# ── Standard library ──────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

# ── HuggingFace Transformers ───────────────────────────────────────────────────
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ── PyTorch ───────────────────────────────────────────────────────────────────
import torch

# ── Data handling ─────────────────────────────────────────────────────────────
import pandas as pd

# ── Visualization ─────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Display settings ──────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.dpi"] = 110

print("✅ All libraries imported successfully.")
print(f"   PyTorch version : {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device in use   : {device.upper()}")

MODEL_NAME = "distilgpt2"  # Lightweight GPT-2 variant

print(f"Loading tokenizer and model: '{MODEL_NAME}' ...")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

# GPT-2 has no padding token by default; set it to the EOS token
tokenizer.pad_token = tokenizer.eos_token

# Load the language model and move it to the available device
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model = model.to(device)
model.eval()  # Set to evaluation mode (disables dropout)

print("✅ Model and tokenizer loaded.")
print(f"   Vocabulary size : {tokenizer.vocab_size:,} tokens")
print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Quick tokenizer demo ──────────────────────────────────────────────────────
demo_text = "Artificial Intelligence is transforming the world."
token_ids = tokenizer.encode(demo_text)
tokens    = tokenizer.convert_ids_to_tokens(token_ids)

print("Tokenization demo")
print("-" * 45)
print(f"Input  : {demo_text}")
print(f"Tokens : {tokens}")
print(f"IDs    : {token_ids}")
print(f"Decoded: {tokenizer.decode(token_ids)}")

def generate_text(
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.85,
    top_k: int = 50,
    top_p: float = 0.92,
    repetition_penalty: float = 1.2,
    num_return_sequences: int = 1,
    do_sample: bool = True,
) -> list[str]:
    """
    Generate text continuations for a given prompt using DistilGPT-2.

    Parameters
    ----------
    prompt               : The input text to condition the model on.
    max_new_tokens       : How many tokens to generate beyond the prompt.
    temperature          : Sampling temperature (higher = more creative).
    top_k                : Top-K sampling filter.
    top_p                : Nucleus (Top-P) sampling filter.
    repetition_penalty   : Penalise repeated n-grams.
    num_return_sequences : How many independent outputs to generate.
    do_sample            : Whether to use stochastic sampling.

    Returns
    -------
    List of generated strings (prompt excluded).
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    results = [
        tokenizer.decode(ids[prompt_len:], skip_special_tokens=True).strip()
        for ids in output_ids
    ]
    return results


def display_generation(prompt: str, outputs: list[str]) -> None:
    """Pretty-print the prompt and its generated continuations."""
    print("=" * 65)
    print(f"PROMPT : {prompt}")
    print("-" * 65)
    for i, text in enumerate(outputs, 1):
        label = f"OUTPUT {i}" if len(outputs) > 1 else "OUTPUT"
        print(f"{label} : {text}")
    print("=" * 65)
    print()


print("✅ Helper functions defined.")

# ── Basic Prompt 1 ────────────────────────────────────────────────────────────
prompt1 = "Artificial Intelligence is"
output1 = generate_text(prompt1, max_new_tokens=80)
display_generation(prompt1, output1)

# ── Basic Prompt 2 ────────────────────────────────────────────────────────────
prompt2 = "The future of technology"
output2 = generate_text(prompt2, max_new_tokens=80)
display_generation(prompt2, output2)

# Dictionary of prompt categories and prompts
advanced_prompts = {
    "Creative Writing"   : "Once upon a time in a distant galaxy, there was a robot who dreamed of",
    "Factual Question"   : "The capital of France is Paris. The Eiffel Tower was built in",
    "Incomplete Sentence": "Scientists have recently discovered that black holes can",
    "Instructional"      : "To make a cup of tea, you need to",
    "Opinion / Emotional": "The most beautiful thing about being human is",
}

# Store outputs for later analysis
experiment_results = {}

for category, prompt in advanced_prompts.items():
    print(f"\n🔖 Category: {category}")
    output = generate_text(prompt, max_new_tokens=90)
    display_generation(prompt, output)
    experiment_results[category] = {
        "prompt" : prompt,
        "output" : output[0],
    }

coherence_prompt = "Deep learning has revolutionized the field of"

# Low temperature → focused, predictable
low_temp_output  = generate_text(coherence_prompt, max_new_tokens=80, temperature=0.3)
# High temperature → creative, sometimes incoherent
high_temp_output = generate_text(coherence_prompt, max_new_tokens=80, temperature=1.4)

print("🌡️  LOW TEMPERATURE (0.3) — More deterministic")
display_generation(coherence_prompt, low_temp_output)

print("🌡️  HIGH TEMPERATURE (1.4) — More random/creative")
display_generation(coherence_prompt, high_temp_output)

rep_prompt = "The quick brown fox jumps over the lazy dog. The dog"

# Without repetition penalty
no_penalty = generate_text(rep_prompt, max_new_tokens=60, repetition_penalty=1.0)
# With repetition penalty
with_penalty = generate_text(rep_prompt, max_new_tokens=60, repetition_penalty=1.4)

print("🔁 WITHOUT Repetition Penalty (1.0)")
display_generation(rep_prompt, no_penalty)

print("✅ WITH Repetition Penalty (1.4)")
display_generation(rep_prompt, with_penalty)

context_prompt = (
    "In the year 2045, humans have colonized Mars. "
    "The first Martian city, called Nova Terra, was built near the equator. "
    "Life on Mars is"
)

context_output = generate_text(context_prompt, max_new_tokens=100)
display_generation(context_prompt, context_output)

print("📝 Observation:")
print("   The model should continue in the context of Mars colonization.")
print("   Watch whether it stays on topic or drifts to unrelated subjects.")

# Test context retention with a multi-sentence setup
rq1_prompts = [
    # Short context
    "Mars is a planet.",
    # Medium context
    "Humans have colonized Mars. The cities are domed structures. Life there is",
    # Long context
    (
        "In 2045, after decades of research, humans finally colonized Mars. "
        "The first settlers built domed cities powered by solar energy. "
        "Children born on Mars have never seen Earth. "
        "The Martian government declared independence in 2067. "
        "The most pressing challenge for Martian citizens today is"
    ),
]

for i, prompt in enumerate(rq1_prompts, 1):
    print(f"\n--- RQ1 Test {i} (context length: {len(prompt.split())} words) ---")
    out = generate_text(prompt, max_new_tokens=70)
    display_generation(prompt, out)

factual_prompts = [
    "The speed of light is approximately",
    "William Shakespeare was born in the year",
    "Water boils at 100 degrees Celsius at sea level, which means",
    "The human body has",
]

for prompt in factual_prompts:
    out = generate_text(prompt, max_new_tokens=50, temperature=0.5)
    display_generation(prompt, out)

# Vague vs specific prompts on the same topic
rq3_pairs = [
    (
        "Vague",
        "Tell me about science.",
        "Specific",
        "Quantum mechanics is a branch of physics that describes how subatomic particles behave. It explains",
    ),
    (
        "Vague",
        "Write something about a dog.",
        "Specific",
        "Max, a golden retriever, sat by the window every afternoon waiting for his owner to return from work. One day,",
    ),
]

for vague_label, vague_prompt, spec_label, spec_prompt in rq3_pairs:
    print(f"\n{'─'*65}")
    print(f"[{vague_label.upper()}]")
    out_v = generate_text(vague_prompt, max_new_tokens=70)
    display_generation(vague_prompt, out_v)

    print(f"[{spec_label.upper()}]")
    out_s = generate_text(spec_prompt, max_new_tokens=70)
    display_generation(spec_prompt, out_s)

# ── Collect statistics for all prompts ───────────────────────────────────────
all_prompts = {
    "Basic: AI is"          : "Artificial Intelligence is",
    "Basic: Future of Tech" : "The future of technology",
    "Creative Writing"      : "Once upon a time in a distant galaxy, there was a robot who dreamed of",
    "Factual Question"      : "The capital of France is Paris. The Eiffel Tower was built in",
    "Incomplete Sentence"   : "Scientists have recently discovered that black holes can",
    "Instructional"         : "To make a cup of tea, you need to",
    "Opinion / Emotional"   : "The most beautiful thing about being human is",
    "Low Temperature"       : "Deep learning has revolutionized the field of",
    "Context: Mars"         : "In 2045, humans have colonized Mars. Life on Mars is",
    "Factual: Speed Light"  : "The speed of light is approximately",
}

stats = []
print("Generating outputs for visualization ...")
for label, prompt in all_prompts.items():
    output = generate_text(prompt, max_new_tokens=80)[0]
    token_ids = tokenizer.encode(output)
    word_count = len(output.split())
    char_count = len(output)
    unique_tokens = len(set(token_ids))
    stats.append({
        "Label"         : label,
        "Token Count"   : len(token_ids),
        "Word Count"    : word_count,
        "Char Count"    : char_count,
        "Unique Tokens" : unique_tokens,
        "Output"        : output,
    })

df = pd.DataFrame(stats)
print("✅ Data collected.")
df[["Label", "Token Count", "Word Count", "Unique Tokens"]].to_string(index=False)

# ── Plot 1: Token Count per Prompt ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Bar chart — token counts
colors = sns.color_palette("muted", len(df))
bars = axes[0].barh(df["Label"], df["Token Count"], color=colors, edgecolor="white", height=0.6)
axes[0].set_xlabel("Number of Tokens Generated", fontsize=11)
axes[0].set_title("Token Count per Prompt", fontsize=13, fontweight="bold")
axes[0].invert_yaxis()
# Annotate bars
for bar, val in zip(bars, df["Token Count"]):
    axes[0].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9)

# Bar chart — word counts
bars2 = axes[1].barh(df["Label"], df["Word Count"], color=colors, edgecolor="white", height=0.6)
axes[1].set_xlabel("Number of Words Generated", fontsize=11)
axes[1].set_title("Word Count per Prompt", fontsize=13, fontweight="bold")
axes[1].invert_yaxis()
for bar, val in zip(bars2, df["Word Count"]):
    axes[1].text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9)

plt.suptitle("Generated Text Length Analysis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot_text_length.png", bbox_inches="tight", dpi=120)
plt.show()
print("📊 Plot 1 saved as 'plot_text_length.png'")

# ── Plot 2: Unique Token Ratio (Lexical Diversity) ────────────────────────────
df["Lexical Diversity"] = df["Unique Tokens"] / df["Token Count"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Scatter — Token Count vs Unique Tokens
scatter = axes[0].scatter(
    df["Token Count"], df["Unique Tokens"],
    c=range(len(df)), cmap="tab10", s=120, zorder=3, edgecolors="white", linewidths=0.8
)
for _, row in df.iterrows():
    axes[0].annotate(
        row["Label"].split(":")[-1].strip(),
        (row["Token Count"], row["Unique Tokens"]),
        textcoords="offset points", xytext=(5, 4), fontsize=7.5
    )
axes[0].set_xlabel("Total Token Count", fontsize=11)
axes[0].set_ylabel("Unique Token Count", fontsize=11)
axes[0].set_title("Total vs Unique Tokens", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.4)

# Bar — Lexical Diversity
pal = sns.color_palette("coolwarm", len(df))
axes[1].bar(range(len(df)), df["Lexical Diversity"].sort_values(), color=pal, edgecolor="white")
axes[1].set_xticks(range(len(df)))
axes[1].set_xticklabels(
    df.loc[df["Lexical Diversity"].sort_values().index, "Label"],
    rotation=45, ha="right", fontsize=8
)
axes[1].set_ylabel("Unique Tokens / Total Tokens", fontsize=11)
axes[1].set_title("Lexical Diversity Score", fontsize=13, fontweight="bold")
axes[1].axhline(0.7, color="red", linestyle="--", alpha=0.6, label="0.7 threshold")
axes[1].legend(fontsize=9)

plt.suptitle("Token Diversity Analysis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("plot_token_diversity.png", bbox_inches="tight", dpi=120)
plt.show()
print("📊 Plot 2 saved as 'plot_token_diversity.png'")

# ── Plot 3: Heatmap — correlation between stats ───────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
numeric_cols = ["Token Count", "Word Count", "Char Count", "Unique Tokens", "Lexical Diversity"]
corr = df[numeric_cols].corr()

sns.heatmap(
    corr, annot=True, fmt=".2f", cmap="YlOrRd",
    linewidths=0.5, ax=ax, annot_kws={"size": 10}
)
ax.set_title("Correlation Heatmap of Generation Statistics", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("plot_correlation_heatmap.png", bbox_inches="tight", dpi=120)
plt.show()
print("📊 Plot 3 saved as 'plot_correlation_heatmap.png'")
