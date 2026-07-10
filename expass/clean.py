import pandas as pd
import json
import re
from pathlib import Path

INPUT_PATH = "/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/DL_approaches/Demo/expass_original/analysis_results/narrative_explanations_all_models_ger_initiating_problem.csv"

OUTPUT_DIR = Path("/data/ARENAS_Automatic_Extremist_Analysis/ARENAS_Automatic_Extremist_Analysis/DL_approaches/Demo/expass_original/analysis_results/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------
# JSON extraction utility
# ----------------------------
def extract_json(text):
    """Try to extract first valid JSON object from messy LLM output."""
    if not isinstance(text, str):
        return None

    # Find first {...}
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group())
    except:
        return None


# ----------------------------
# importance cleaning
# ----------------------------
def clean_importance(val):
    if val is None:
        return None
    if not isinstance(val, str):
        return None

    val = val.strip()

    # reject numeric-only strings
    if re.fullmatch(r"[0-9\.\-eE]+", val):
        return None

    # reject too short vague labels
    if val.lower() in ["high", "low", "moderate", "unknown", "none"]:
        return None

    # must be meaningful explanation
    if len(val) < 15:
        return None

    return val


# ----------------------------
# parsing function
# ----------------------------
def parse_model_column(df, col_name, model_name):
    rows = []

    for _, row in df.iterrows():
        src = row["source_text"]
        tgt = row["target_text"]
        raw = row[col_name]

        parsed = extract_json(raw)

        if parsed is None:
            continue

        importance = clean_importance(parsed.get("explanation_of_importance"))

        if importance is None:
            continue

        rows.append({
            "model": model_name,
            "source_text": src,
            "target_text": tgt,
            "explanation_of_importance": importance
        })

    return pd.DataFrame(rows)


# ----------------------------
# load data
# ----------------------------
df = pd.read_csv(INPUT_PATH)

# keep only relevant models
models = {
    "explanation_llama": "llama",
    "explanation_mistral": "mistral"
}

all_outputs = []

for col, name in models.items():
    print(f"Processing {name} ...")
    out_df = parse_model_column(df, col, name)
    out_path = OUTPUT_DIR / f"{name}_clean_importance_clean_initiating_problem_ger.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    all_outputs.append(out_df)

# optional merged file
merged = pd.concat(all_outputs, ignore_index=True)
merged.to_csv(OUTPUT_DIR / "all_models_clean_importance_initiating_problem_clean_ger.csv", index=False)

print("DONE ✔")
