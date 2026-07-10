#!/usr/bin/env python3

import argparse
import pandas as pd
import requests
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=str,
        default="analysis_results/top_100_edges_with_context_solution.csv",
        help="Input CSV file with enriched edge context",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="Solution",
        help='Label column base name, e.g. "Solution", "Topic", "Initiating Problem"',
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="analysis_results/narrative_explanations_all_models_el_solution.csv",
        help="Output CSV with explanations from all models",
    )
    parser.add_argument(
        "--ollama-url",
        type=str,
        default="http://localhost:11434/api/generate",
        help="Ollama generate endpoint",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP timeout (seconds)",
    )
    return parser.parse_args()


def generate(prompt, model_name, ollama_url, timeout=300):
    response = requests.post(
        ollama_url,
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        },
        timeout=timeout,
    )
    response.raise_for_status()
    result = response.json()
    if "response" not in result:
        print(f"ERROR: {result}")
        return ""
    return result["response"]


def main():
    args = parse_args()

    df = pd.read_csv(args.input_file)

    source_label_key = f"source_{args.label_col}"
    target_label_key = f"target_{args.label_col}"

    if source_label_key not in df.columns or target_label_key not in df.columns:
        raise ValueError(
            f"Missing expected columns: '{source_label_key}' and/or '{target_label_key}'.\n"
            f"Available columns: {list(df.columns)}"
        )

    models = {
       # "deepseek": "deepseek-r1:7b",
        "llama": "llama3",
        "mistral": "mistral"
    }

    for model_key, model_name in models.items():
        print(f"\nGenerating with {model_key}...\n")
        narratives = []

        for _, row in tqdm(df.iterrows(), total=len(df)):

            prompt = f"""
Ένα Graph Neural Network εντόπισε μια σημαντική σύνδεση μεταξύ δύο μηνυμάτων σε έναν αφηγηματικό γράφο.

Βαθμός σημαντικότητας της ακμής: {row['importance_score']}

Μήνυμα προέλευσης:
{row['source_text']}

Μήνυμα προορισμού:
{row['target_text']}

Χαρακτηριστικά του μηνύματος προέλευσης:
- Θεματική: {row[source_label_key]}
- In-group: {row['source_in_group']}
- Out-group: {row['source_out_group']}

Χαρακτηριστικά του μηνύματος προορισμού:
- Θεματική: {row[target_label_key]}
- In-group: {row['target_in_group']}
- Out-group: {row['target_out_group']}


Εργασία:
Ανάλυσε τη σχέση μεταξύ των δύο μηνυμάτων και δημιούργησε μια δομημένη εξήγηση αυτής της σύνδεσης.

Πρέπει να προσδιορίσεις:
- actors: τις κοινωνικές ή ιδεολογικές ομάδες που εμπλέκονται στην αλληλεπίδραση (π.χ. in-group, out-group ή σχετιζόμενες ομάδες ταυτότητας)
- relationship: τύπος σημασιολογικής ή λεκτικής σχέσης μεταξύ των μηνυμάτων (agreement, reinforcement, opposition, escalation, unrelated)
- narrative_type: κυρίαρχο αφηγηματικό πλαίσιο (εχθρότητα, θυματοποίηση, κατηγορία, φόβος, ενίσχυση ταυτότητας, ουδέτερο)
- importance_mechanism: σύντομη φράση που περιγράφει γιατί αυτή η ακμή είναι δομικά σημαντική στον γράφο (π.χ. αφηγηματική γέφυρα, βρόχος ενίσχυσης, μονοπάτι κλιμάκωσης, δια-ομαδικός σύνδεσμος)
- explanation_of_importance: εξήγηση σε 2-3 προτάσεις για το γιατί αυτή η σύνδεση είναι σημαντική για το μοντέλο GNN, λαμβάνοντας υπόψη τις δυναμικές in-group / out-group και τη διάδοση της αφήγησης

Οδηγίες:
- Λάβε ρητά υπόψη τις δυναμικές in-group / out-group όταν υπάρχουν
- Εστίασε στον τρόπο με τον οποίο τα μηνύματα διαδίδονται ή ενισχύονται σε μια δομή γράφου
- Αξιολόγησε αν η σύνδεση λειτουργεί ως γέφυρα, ενίσχυση ή αφηγηματική κλιμάκωση
- Χρησιμοποίησε ΜΟΝΟ τις παρεχόμενες πληροφορίες (καμία επινόηση εξωτερικού πλαισίου)

Μορφή εξόδου (ΑΥΣΤΗΡΗ):
Επίστρεψε ΜΟΝΟ ένα έγκυρο αντικείμενο JSON:

{{
  "actors": "...",
  "relationship": "...",
  "narrative_type": "...",
  "importance_mechanism": "...",
  "explanation_of_importance": "..."
}}

Περιορισμοί:
- Έξοδος αυστηρά μόνο σε έγκυρο JSON
- Κανένα κείμενο πριν ή μετά
- Κανένα markdown
- Σύντομα πεδία για όλα εκτός από το explanation_of_importance (2-3 προτάσεις το πολύ)
"""

            narrative = generate(prompt, model_name, args.ollama_url, timeout=args.timeout)
            narratives.append(narrative)

        df[f"explanation_{model_key}"] = narratives

    df.to_csv(args.output_file, index=False)
    print("Saved:", args.output_file)


if __name__ == "__main__":
    main()
