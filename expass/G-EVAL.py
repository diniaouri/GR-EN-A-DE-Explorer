import os
import re
import time
import pandas as pd
import argparse
from openai import OpenAI

# =====================================================================
# ARGPARSE (ONLY ADDITION)
# =====================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="qwen2.5:32b")
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--explanation-col", type=str, default="explanation_of_importance",
                    help="Column holding the generated explanation to evaluate")
args = parser.parse_args()

# =====================================================================
# CONFIG & CLIENT SETUP (LOCAL INFERENCE ENGINE)
# =====================================================================
client = OpenAI(
    base_url="http://localhost:11434/v1/",
    api_key="ollama"
)

# Set the local judge model (NOW FROM CLI)
JUDGE_MODEL = args.model

INPUT_CSV = args.input
OUTPUT_CSV = args.output

# =====================================================================
# G-EVAL CRITERIA DEFINITION
# =====================================================================
GEVAL_PROMPT_TEMPLATE = """
You are an expert academic reviewer evaluating an Explainable AI (XAI) system for Graph Neural Networks (GNNs). 
The GNN predicted a link/edge between Node 1 and Node 2. A separate generative model provided an explanation for *why* this link exists.

Your task is to evaluate the explanation based on three strict criteria: **Faithfulness**, **Relational Validity**, and **Conciseness**.

---
[DATA TO EVALUATE]
Node 1 Text: {node1}
Node 2 Text: {node2}
Generated Explanation: {explanation}

---
[EVALUATION RUBRIC]

1. **Faithfulness (Score 1-5)**: Does the explanation rely strictly on facts, terms, or context present in Node 1 and Node 2? 
   - 5: Entirely faithful. No outside facts or unverified extremist tropes are introduced.
   - 3: Moderately faithful. Mentions concepts derived from the nodes, but infers background context not explicitly present.
   - 1: Severe hallucination. Attributes statements or facts to the nodes that do not exist in the source text.

2. **Relational Validity (Score 1-5)**: Does the explanation clearly justify a *link* or relationship between both nodes, rather than just summarizing them individually?
   - 5: Excellent. Explains exactly how Node 1 and Node 2 bridge, intersect, or interact.
   - 3: Partial. Explains the content of both nodes well, but the actual transactional relationship/link between them is vague.
   - 1: Failed link. Only discusses one node, or treats them as completely separate entities without a bridge.

3. **Conciseness (Score 1-5)**: Is the explanation precise and free of fluff, repetition, or meta-commentary?
   - 5: Highly compact, maximizing information density.
   - 3: Fluent but contains unnecessary words or repetitive framing.
   - 1: Extremely verbose, filled with empty filler text or redundant summaries.

---
[OUTPUT FORMAT INSTRUCTIONS]
You must think step-by-step. Provide your analysis first, and then conclude with the scores.
Your final output MUST contain these explicit score lines somewhere at the very end of your response text:

- Faithfulness: [Integer]
- Relational Validity: [Integer]
- Conciseness: [Integer]
"""

# =====================================================================
# PARSING HELPER
# =====================================================================
def parse_geval_output(response_text):
    """
    Extract the final 1-5 score for each criterion.

    IMPORTANT: we match an explicit "Label: <digit>" pattern and take the LAST
    such match. This deliberately ignores the rubric header the judge often
    echoes, e.g. "Faithfulness (Score 1-5)", where a naive first-digit regex
    would wrongly capture the '1' from '1-5' and floor every score at 1.
    """
    scores = {"faithfulness": None, "relational_validity": None, "conciseness": None}
    if not response_text:
        return scores

    labels = {
        "faithfulness": r"Faithfulness",
        "relational_validity": r"Relational\s*Validity",
        "conciseness": r"Conciseness",
    }
    try:
        for key, lab in labels.items():
            # label, optional markdown/space, a COLON, optional markdown/space, then the digit
            matches = re.findall(lab + r"\**\s*:\s*\**\s*([1-5])\b",
                                 response_text, re.IGNORECASE)
            if matches:
                scores[key] = int(matches[-1])   # take the FINAL stated score
    except Exception as e:
        print(f"Error parsing local output scores: {e}")
    return scores

# =====================================================================
# PIPELINE EXECUTION ENGINE
# =====================================================================
def run_geval():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file not found at {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)
    
    reasons, f_scores, rv_scores, c_scores = [], [], [], []
    total_records = len(df)
    
    print(f"Starting localized G-Eval pipeline using {JUDGE_MODEL} on {total_records} records...")
    print("Running entirely offline. Rate limits and token costs are disabled.\n")
    
    try:
        for idx, row in df.iterrows():
            node1 = str(row.get("source_text", ""))
            node2 = str(row.get("target_text", ""))
            explanation = str(row.get(args.explanation_col, "")).strip()
            if not explanation or explanation.lower() == "nan":
                # Do NOT send an empty explanation to the judge: that produces meaningless scores.
                print(f"[{idx+1}/{total_records}] SKIPPED -> empty '{args.explanation_col}'")
                reasons.append("SKIPPED: empty explanation column")
                f_scores.append(None); rv_scores.append(None); c_scores.append(None)
                continue
            
            formatted_prompt = GEVAL_PROMPT_TEMPLATE.format(
                node1=node1, node2=node2, explanation=explanation
            )
            
            try:
                response = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=[
                        {"role": "system", "content": "You are a rigid, objective, and unbiased NLP evaluation bot."},
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.1
                )
                
                output_text = response.choices[0].message.content
                scores = parse_geval_output(output_text)
                
                reasons.append(output_text)
                f_scores.append(scores["faithfulness"])
                rv_scores.append(scores["relational_validity"])
                c_scores.append(scores["conciseness"])
                
                print(f"[{idx+1}/{total_records}] Success -> F: {scores['faithfulness']} | RV: {scores['relational_validity']} | C: {scores['conciseness']}")
                
            except Exception as e:
                print(f"[{idx+1}/{total_records}] Local processing error encountered: {e}")
                reasons.append(f"Local runtime error: {e}")
                f_scores.append(None)
                rv_scores.append(None)
                c_scores.append(None)
                
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\nPipeline execution halted manually by user.")

    processed_len = len(f_scores)
    save_df = df.iloc[:processed_len].copy()
    save_df["geval_raw_output"] = reasons
    save_df["geval_faithfulness"] = f_scores
    save_df["geval_relational_validity"] = rv_scores
    save_df["geval_conciseness"] = c_scores
    
    save_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nEvaluation complete. Local data compiled cleanly into: {OUTPUT_CSV}")

if __name__ == "__main__":
    run_geval()
