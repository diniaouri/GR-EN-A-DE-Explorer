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
        default="analysis_results/narrative_explanations_all_models_de_solution.csv",
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
Ein Graph Neural Network hat eine wichtige Verbindung zwischen zwei Nachrichten in einem narrativen Graphen identifiziert.

Wichtigkeitsscore der Kante: {row['importance_score']}

Quellnachricht:
{row['source_text']}

Zielnachricht:
{row['target_text']}

Attribute der Quellnachricht:
- Thema: {row[source_label_key]}
- In-Group: {row['source_in_group']}
- Out-Group: {row['source_out_group']}

Attribute der Zielnachricht:
- Thema: {row[target_label_key]}
- In-Group: {row['target_in_group']}
- Out-Group: {row['target_out_group']}


Aufgabe:
Analysiere die Beziehung zwischen den beiden Nachrichten und erstelle eine strukturierte Erklaerung dieser Verbindung.

Du musst Folgendes identifizieren:
- actors: die sozialen oder ideologischen Gruppen, die an der Interaktion beteiligt sind (z. B. In-Group, Out-Group oder zugehoerige Identitaetsgruppen)
- relationship: Art der semantischen oder diskursiven Beziehung zwischen den Nachrichten (agreement, reinforcement, opposition, escalation, unrelated)
- narrative_type: dominanter narrativer Rahmen (Feindseligkeit, Viktimisierung, Beschuldigung, Angst, Identitaetsverstaerkung, neutral)
- importance_mechanism: kurzer Satz, der beschreibt, warum diese Kante im Graphen strukturell wichtig ist (z. B. narrative Bruecke, Verstaerkungsschleife, Eskalationspfad, gruppenuebergreifende Verbindung)
- explanation_of_importance: Erklaerung in 2-3 Saetzen, warum diese Verbindung fuer das GNN-Modell wichtig ist, unter Beruecksichtigung der In-Group-/Out-Group-Dynamiken und der narrativen Verbreitung

Hinweise:
- Beruecksichtige ausdruecklich die In-Group-/Out-Group-Dynamiken, sofern vorhanden
- Konzentriere dich darauf, wie sich Nachrichten in einer Graphenstruktur verbreiten oder verstaerken
- Bewerte, ob die Verbindung als Bruecke, Verstaerkung oder narrative Eskalation wirkt
- Verwende AUSSCHLIESSLICH die bereitgestellten Informationen (kein Erfinden von externem Kontext)

Ausgabeformat (STRIKT):
Gib AUSSCHLIESSLICH ein gueltiges JSON-Objekt zurueck:

{{
  "actors": "...",
  "relationship": "...",
  "narrative_type": "...",
  "importance_mechanism": "...",
  "explanation_of_importance": "..."
}}

Einschraenkungen:
- Ausgabe ausschliesslich als striktes, gueltiges JSON
- Kein Text davor oder danach
- Kein Markdown
- Kurze Felder fuer alle ausser explanation_of_importance (maximal 2-3 Saetze)
"""

            narrative = generate(prompt, model_name, args.ollama_url, timeout=args.timeout)
            narratives.append(narrative)

        df[f"explanation_{model_key}"] = narratives

    df.to_csv(args.output_file, index=False)
    print("Saved:", args.output_file)


if __name__ == "__main__":
    main()
