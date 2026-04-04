import pandas as pd
from pathlib import Path
import random
import re
from tqdm import tqdm
from datasets import load_dataset

# Maithili specific rules (Pronouns and common verbs)
RULES = {
    "है": "अछि",
    "हैं": "छथि", 
    "था": "छल",
    "थी": "छल",
    "नहीं": "नहि",
    "तुम": "अहाँ",
    "मैं": "हम",
    "कैसे": "कोना",
    "क्या": "कि",
    "तुमको": "अहाँक",
    "कर": "कए",
    "कहा": "कहला",
    "बोला": "बजला",
    "बहुत": "बहुते"
}

class VariationEngine:
    @staticmethod
    def shuffle_words(text: str, prob: float = 0.1) -> str:
        words = text.split()
        if len(words) < 3: return text
        for i in range(len(words) - 1):
            if random.random() < prob:
                words[i], words[i+1] = words[i+1], words[i]
        return " ".join(words)

    @staticmethod
    def add_noise(text: str, prob: float = 0.05) -> str:
        chars = list(text)
        marks = [".", "!", "?", "..."]
        if random.random() < prob:
            idx = random.randint(0, len(chars))
            chars.insert(idx, random.choice(marks))
        return "".join(chars)

HINDI_DIR = Path("datasets/raw/hindi")
MAITHILI_DIR = Path("datasets/raw/maithili")

def apply_rules(text: str) -> str:
    for hi, mai in RULES.items():
        text = re.sub(r"\b" + hi + r"\b", mai, text)
    
    # Apply variations
    engine = VariationEngine()
    text = engine.shuffle_words(text)
    text = engine.add_noise(text)
    return text

def extract_from_hf():
    """Attempt to find Maithili specific rows in datasets."""
    # Samanantar doesn't have sentiment labels, but we can use it for linguistic patterns.
    # IndicSentiment might have some or we can find generic Maithili sentences.
    # For now, we will focus on the synthetic generation from Hindi base with Maithili rules.
    pass

def generate_maithili(num_samples: int = 50_000):
    MAITHILI_DIR.mkdir(parents=True, exist_ok=True)
    
    hindi_files = list(HINDI_DIR.glob("*.csv"))
    if not hindi_files:
        print("No Hindi base files found for Maithili generation. Using dummy base.")
        base_df = pd.DataFrame([
            {"text": "यह बहुत अच्छा है", "label": "positive"},
            {"text": "तुम क्या कर रहे हो", "label": "neutral"},
            {"text": "मैं नहीं जा रहा", "label": "negative"}
        ] * (num_samples // 3 + 1))
    else:
        # Load and standardize all Hindi files
        dfs = []
        for f in hindi_files:
            df_tmp = pd.read_csv(f)
            # Consolidate text columns
            if "INDIC REVIEW" in df_tmp.columns:
                df_tmp = df_tmp.rename(columns={"INDIC REVIEW": "text", "LABEL": "label"})
            elif "content" in df_tmp.columns:
                df_tmp = df_tmp.rename(columns={"content": "text", "sentiment": "label"})
            
            # Ensure we have text and label
            cols = [c for c in ["text", "label"] if c in df_tmp.columns]
            dfs.append(df_tmp[cols])
            
        base_df = pd.concat(dfs, ignore_index=True).dropna(subset=["text"])
        base_df = base_df.sample(n=min(num_samples * 2, len(base_df)), replace=len(base_df) < num_samples * 2, random_state=42)

    print(f"Generating {num_samples} Maithili samples from Hindi base...")
    
    data = []
    for _, row in tqdm(base_df.iterrows(), total=num_samples):
        if len(data) >= num_samples:
            break
            
        original = str(row.get("text", ""))
        label = row.get("label", "neutral")
        
        if not original or original == "nan": continue
        
        mai_text = apply_rules(original)
        data.append({"text": mai_text, "label": label, "language": "maithili", "source": "synthetic_generator"})

    df = pd.DataFrame(data)
    unique_count = df["text"].nunique()
    print(f"Generated: {len(df)} samples")
    print(f"Unique: {unique_count}")
    
    df.to_csv(MAITHILI_DIR / "maithili.csv", index=False)
    print(f"Maithili synthesis complete: {len(df)} samples saved to {MAITHILI_DIR / 'maithili.csv'}")

if __name__ == "__main__":
    generate_maithili(50_000)
