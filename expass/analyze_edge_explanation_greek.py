#!/usr/bin/env python3
"""
Analyze edge importance explanations with full context:
- Text content
- Target group labels
- In-Group and Out-group context (used to build edges in GRENADE)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def load_toxigen_data(exp_nb=11):
    possible_paths = [
        "../grenade_original/Contrastive_Learning_Approach/datasets/Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx",
        "data/Multilingual_EN_Corpus_DATA_CYPRIOT.xlsx",
    ]

    for path in possible_paths:
        if Path(path).exists():
            print(f"Loading Multilingual_EN_Corpus_DATA_FRENCH data from {path}...")

            df = pd.read_excel(path, header=4)

            df = df.rename(columns={
                "Text": "text",
                "Topic": "target_group",
                "In-Group": "In-Group",
                "Out-group": "Out-group"
            })

            print(f"✓ Loaded {len(df)} texts")
            print(f"  Columns: {list(df.columns)}")
            return df

    raise FileNotFoundError("Could not find Multilingual_EN_Corpus_DATA_FRENCH.xlsx")


def analyze_important_edges(
    edge_csv: str = "edge_importance_exp4.csv",
    top_k: int = 20,
    show_context: bool = True,
):
    """
    Analyze the most important edges with full context information.
    """
    print("=" * 100)
    print("ANALYZING EDGE IMPORTANCE WITH FULL CONTEXT DATA")
    print("=" * 100)

    # Load edge importance
    print(f"\nLoading edge importance from {edge_csv}...")
    edges_df = pd.read_csv(edge_csv)

    print(f"✓ Loaded {len(edges_df)} edges")
    print(f"  Score range: {edges_df['importance_score'].min():.4f} to {edges_df['importance_score'].max():.4f}")
    print(f"  Mean score: {edges_df['importance_score'].mean():.4f}")
    print(f"  Median score: {edges_df['importance_score'].median():.4f}")

    # Load text data
    texts_df = load_toxigen_data()

    print(f"\nAvailable columns in Toxigen data:")
    for col in texts_df.columns:
        print(f"  - {col}")

    top_edges = edges_df.head(top_k)

    print(f"\n{'=' * 100}")
    print(f"TOP {top_k} MOST IMPORTANT EDGES")
    print(f"{'=' * 100}\n")

    for idx, row in top_edges.iterrows():
        source = int(row['source_node'])
        target = int(row['target_node'])
        score = row['importance_score']
        rank = int(row['rank'])

        print(f"{'=' * 100}")
        print(f"RANK #{rank} | Importance Score: {score:.6f}")
        print(f"Edge: Node {source} → Node {target}")
        print(f"{'=' * 100}")

        # SOURCE NODE
        if source < len(texts_df):
            source_row = texts_df.iloc[source]
            print(f"\n📍 SOURCE NODE {source}:")
            print(f"  Target Group: {source_row.get('target_group', 'N/A')}")

            if show_context:
                print(f"  In-Group: {source_row.get('In-Group', 'N/A')}")
                print(f"  Out-group: {source_row.get('Out-group', 'N/A')}")

            text = source_row.get('text', 'N/A')
            print(f"  Text: {text}")

        # TARGET NODE
        if target < len(texts_df):
            target_row = texts_df.iloc[target]
            print(f"\n🎯 TARGET NODE {target}:")
            print(f"  Target Group: {target_row.get('target_group', 'N/A')}")

            if show_context:
                print(f"  In-Group: {target_row.get('In-Group', 'N/A')}")
                print(f"  Out-group: {target_row.get('Out-group', 'N/A')}")

            text = target_row.get('text', 'N/A')
            print(f"  Text: {text}")

        # CONNECTION ANALYSIS
        if source < len(texts_df) and target < len(texts_df):
            source_row = texts_df.iloc[source]
            target_row = texts_df.iloc[target]

            print(f"\n🔗 CONNECTION ANALYSIS:")

            same_target = source_row.get('target_group') == target_row.get('target_group')
            print(f"  Same target group: {'✓ YES' if same_target else '✗ NO'}")

            if 'In-Group' in texts_df.columns and 'Out-group' in texts_df.columns:
                source_in = str(source_row.get('In-Group', '')).lower()
                source_out = str(source_row.get('Out-group', '')).lower()
                target_in = str(target_row.get('In-Group', '')).lower()
                target_out = str(target_row.get('Out-group', '')).lower()

                in_in_match = source_in == target_in and source_in != 'nan'
                out_out_match = source_out == target_out and source_out != 'nan'
                cross_match = (source_in == target_out or source_out == target_in) and source_in != 'nan'

                if in_in_match:
                    print(f"  Context match: ✓ Same In-Group ({source_in})")
                if out_out_match:
                    print(f"  Context match: ✓ Same Out-group ({source_out})")
                if cross_match:
                    print(f"  Context match: ⚡ Cross-group connection")
                if not (in_in_match or out_out_match or cross_match):
                    print(f"  Context match: ✗ Different contexts")

        print()

    # PATTERN ANALYSIS (unchanged logic, cleaned indentation)
    print(f"\n{'=' * 100}")
    print(f"PATTERN ANALYSIS OF TOP {top_k} EDGES")
    print(f"{'=' * 100}\n")

    source_labels = []
    target_labels = []
    same_label_edges = []

    in_group_matches = 0
    out_group_matches = 0
    cross_group_matches = 0
    different_context = 0

    for idx, row in top_edges.iterrows():
        source = int(row['source_node'])
        target = int(row['target_node'])

        if source < len(texts_df) and target < len(texts_df):
            source_row = texts_df.iloc[source]
            target_row = texts_df.iloc[target]

            s_label = source_row.get('target_group', 'N/A')
            t_label = target_row.get('target_group', 'N/A')

            source_labels.append(s_label)
            target_labels.append(t_label)
            same_label_edges.append(s_label == t_label)

            if 'In-Group' in texts_df.columns:
                source_in = str(source_row.get('In-Group', '')).lower()
                source_out = str(source_row.get('Out-group', '')).lower()
                target_in = str(target_row.get('In-Group', '')).lower()
                target_out = str(target_row.get('Out-group', '')).lower()

                if source_in == target_in and source_in != 'nan':
                    in_group_matches += 1
                if source_out == target_out and source_out != 'nan':
                    out_group_matches += 1
                if (source_in == target_out or source_out == target_in) and source_in != 'nan':
                    cross_group_matches += 1
                if not ((source_in == target_in) or (source_out == target_out) or
                        (source_in == target_out) or (source_out == target_in)):
                    different_context += 1

    if source_labels and target_labels:
        print(f"📊 Target Group Distribution:")

        print("\n  Source nodes:")
        for label, count in pd.Series(source_labels).value_counts().items():
            print(f"    {label}: {count} ({100 * count / len(source_labels):.1f}%)")

        print("\n  Target nodes:")
        for label, count in pd.Series(target_labels).value_counts().items():
            print(f"    {label}: {count} ({100 * count / len(target_labels):.1f}%)")

        same_label_count = sum(same_label_edges)

        print(f"\n📈 Homophily:")
        print(f"  Same label: {same_label_count}/{len(same_label_edges)}")
        print(f"  Different label: {len(same_label_edges) - same_label_count}/{len(same_label_edges)}")

        print(f"\n🔗 Context-based Connections:")
        print(f"  Same In-Group: {in_group_matches}/{len(top_edges)}")
        print(f"  Same Out-group: {out_group_matches}/{len(top_edges)}")
        print(f"  Cross-group: {cross_group_matches}/{len(top_edges)}")
        print(f"  Different contexts: {different_context}/{len(top_edges)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", default="edge_importance_exp1.csv")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--no-context", action="store_true")

    args = parser.parse_args()

    analyze_important_edges(
        edge_csv=args.edges,
        top_k=args.top_k,
        show_context=not args.no_context,
    )
