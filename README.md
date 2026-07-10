# GR-EN-A-DE Explorer: demo

Interactive Streamlit app for exploring graphs and GNN edge-importance
explanations of extremist-narrative detection.

**Live app:** https://grenade-explorer.streamlit.app/

## Quick start with the example data

The app is upload-driven. Ready-made example files live in
[`example/`](example) so you can try every tab immediately:

| File | Where to load it |
|------|------------------|
| `examples/example_adjacency.pkl` | Sidebar → **Adjacency** uploader |
| `examples/example_dataset.csv`   | Sidebar → **Dataset `.csv`** uploader, then **🚀 Load files** |
| `examples/edge_importance_sample.csv` | **Edge Importance** tab → upload → **📥 Load files** |


## Input file formats (to use your own data)

**Adjacency** — a pickled (`.pkl` / `.pkl.gz`) or `.joblib` file containing a
square `N×N` NumPy array (or a torch tensor). Non-zero entries are edges; the
matrix is treated as undirected and the diagonal is ignored.

**Dataset CSV** — one row per node. It is read with the first column as the
index, so column 1 is the node id and the rest are node features. Columns the
app recognises by name (used for LLM prompts): a text column containing any of
`text / message / content / tweet / post`, a `label` column, and `ingroup` /
`outgroup` columns. Any low-cardinality column can be used to colour nodes.

**Edge importance CSV** — one row per edge, with columns:
`source_node`, `target_node`, and a score column named `importance_score`
(`score` or `weight` also accepted). The label shown in the app is taken from the
part of the filename after the last underscore — e.g. `edge_importance_sample.csv`
appears as label **sample**.

## LLM explanations (GraphXAIN tab)

The **GraphXAIN** tab turns top-ranked important edges into natural-language
explanations using a local LLM served by [Ollama](https://ollama.com). This step
is optional — every other tab works without it.

To enable it:

1. Install Ollama and pull a model, e.g.:
   ```bash
   ollama pull deepseek-r1:7b     # or llama3, mistral, phi3, gemma3, qwen2.5:7b
   ollama serve                   # serves http://localhost:11434
   ```
2. In the GraphXAIN tab set **Ollama URL** (default `http://localhost:11434`),
   pick the **Model**, click **🔌 Check Ollama** to confirm the connection, then
   **⚡ Generate Explanations**.

> On the hosted demo there is no LLM server, so the GraphXAIN tab shows a
> "cannot connect" message — this is expected. Run the app locally (or point the
> URL field at your own reachable Ollama endpoint) to use explanations.

## Run locally

```bash
pip install -r requirements.txt
streamlit run GR-EN-A-DE_Explorer.py
```
