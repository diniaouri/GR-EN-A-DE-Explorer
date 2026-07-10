#!/usr/bin/env python3

import argparse
import pandas as pd
import sys
from pathlib import Path
from analyze_edge_explanation_german import analyze_important_edges, load_toxigen_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label-col",
        type=str,
        default="Solution",
        help='Label column to export (e.g., "Solution", "Topic", "Initiating Problem").',
    )
    parser.add_argument(
        "--edge-csv",
        type=str,
        default="edge_importance_exp4_solution_vanilla.csv",
        help="Path to edge importance CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="analysis_results",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="solution",
        help="Tag used in output filenames (e.g., solution, topic, initiating_problem).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Optional top-k edges to keep; default is all edges.",
    )
    args = parser.parse_args()

    label_col = args.label_col
    edge_csv = args.edge_csv
    output_dir = Path(args.output_dir)
    tag = args.tag
    top_k = args.top_k

    output_dir.mkdir(exist_ok=True)

    # =========================
    # 1. SAVE FULL TEXT REPORT
    # =========================
    original_stdout = sys.stdout
    report_path = output_dir / f"edge_analysis_full_report_{tag}.txt"

    with open(report_path, "w") as f:
        sys.stdout = f
        analyze_important_edges(
            edge_csv=edge_csv,
            top_k=top_k,
            show_context=True,
        )
    sys.stdout = original_stdout
    print(f"✓ Full analysis saved: {report_path}")

    # =========================
    # 2. LOAD DATASET
    # =========================
    texts_df = load_toxigen_data()
    texts_df = texts_df.dropna(subset=["text"]).reset_index(drop=True)

    if label_col not in texts_df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in dataset. "
            f"Available columns: {list(texts_df.columns)}"
        )

    # =========================
    # 3. LOAD EDGES
    # =========================
    edges_df = pd.read_csv(edge_csv).sort_values(by="importance_score", ascending=False)
    if top_k is not None:
        edges_df = edges_df.head(top_k)

    # =========================
    # 4. BUILD ENRICHED CSV
    # =========================
    enriched_data = []

    for _, row in edges_df.iterrows():
        source = int(row["source_node"])
        target = int(row["target_node"])

        source_row = texts_df.iloc[source] if source < len(texts_df) else None
        target_row = texts_df.iloc[target] if target < len(texts_df) else None

        enriched_data.append({
            "rank": int(row["rank"]),
            "importance_score": row["importance_score"],
            "source_node": source,
            "target_node": target,

            f"source_{label_col}": source_row.get(label_col) if source_row is not None else "N/A",
            f"target_{label_col}": target_row.get(label_col) if target_row is not None else "N/A",

            "source_in_group": source_row.get("In-Group") if source_row is not None else "N/A",
            "source_out_group": source_row.get("Out-group") if source_row is not None else "N/A",

            "target_in_group": target_row.get("In-Group") if target_row is not None else "N/A",
            "target_out_group": target_row.get("Out-group") if target_row is not None else "N/A",

            "source_text": source_row.get("text", "N/A") if source_row is not None else "N/A",
            "target_text": target_row.get("text", "N/A") if target_row is not None else "N/A",
        })

    # =========================
    # 5. SAVE CSV
    # =========================
    enriched_df = pd.DataFrame(enriched_data)
    csv_path = output_dir / f"top_edges_with_context_{tag}.csv"
    enriched_df.to_csv(csv_path, index=False)

    print("\n✓ Analysis saved to:")
    print(f"  📄 {report_path}")
    print(f"  📊 {csv_path}")


if __name__ == "__main__":
    main()
