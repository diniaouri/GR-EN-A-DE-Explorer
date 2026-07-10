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
        default="analysis_results/narrative_explanations_all_models_fr_solution.csv",
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

	Un Graph Neural Network a identifié une connexion importante entre deux messages dans un graphe narratif.

	Score d’importance de l’arête : {row['importance_score']}

	Message source :
	{row['source_text']}

	Message cible :
	{row['target_text']}

	Attributs du message source :
	- Thématique : {row[source_label_key]}
	- In-group : {row['source_in_group']}
	- Out-group : {row['source_out_group']}

	Attributs du message cible :
	- Thématique : {row[target_label_key]}
	- In-group : {row['target_in_group']}
	- Out-group : {row['target_out_group']}


	Tâche :
	Analyser la relation entre les deux messages et produire une explication structurée de cette connexion.

	Tu dois identifier :
	- actors : les groupes sociaux ou idéologiques impliqués dans l’interaction (ex : in-group, out-group, ou groupes d’identité associés)
	- relationship : type de relation sémantique ou discursive entre les messages (agreement, reinforcement, opposition, escalation, unrelated)
	- narrative_type : cadre narratif dominant (hostilité, victimisation, accusation, peur, renforcement identitaire, neutre)
	- importance_mechanism : phrase courte décrivant pourquoi cette arête est structurellement importante dans le graphe (ex : pont narratif, boucle de renforcement, chemin d’escalade, lien inter-groupes)
	- explanation_of_importance : explication en 2–3 phrases expliquant pourquoi cette connexion est importante pour le modèle GNN, en tenant compte des dynamiques in-group / out-group et de la propagation narrative

	Consignes :
	- Prendre explicitement en compte les dynamiques in-group / out-group lorsqu’elles sont présentes
	- Se concentrer sur la manière dont les messages se propagent ou se renforcent dans une structure de graphe
	- Évaluer si la connexion agit comme un pont, une amplification ou une escalade narrative
	- Utiliser UNIQUEMENT les informations fournies (aucune invention de contexte externe)

	Format de sortie (STRICT) :
	Retourner UNIQUEMENT un objet JSON valide :

	{{
	  "actors": "...",
	  "relationship": "...",
	  "narrative_type": "...",
	  "importance_mechanism": "...",
	  "explanation_of_importance": "..."
	}}

	Contraintes :
	- Sortie strictement en JSON valide uniquement
	- Aucun texte avant ou après
	- Aucun markdown
	- Champs courts pour tous sauf explanation_of_importance (2–3 phrases maximum)
	"""
	
            
            
            
            

            narrative = generate(prompt, model_name, args.ollama_url, timeout=args.timeout)
            narratives.append(narrative)

        df[f"explanation_{model_key}"] = narratives

    df.to_csv(args.output_file, index=False)
    print("Saved:", args.output_file)


if __name__ == "__main__":
    main()
