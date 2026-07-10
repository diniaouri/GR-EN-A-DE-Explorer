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
# License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This work is licensed under a
[**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Summary

You are free to:

- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:

- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made. You may do so in any reasonable manner, but not in any way that suggests the licensor endorses you or your use.
- **NonCommercial** — You may not use the material for commercial purposes.
- **ShareAlike** — If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.
- **No additional restrictions** — You may not apply legal terms or technological measures that legally restrict others from doing anything the license permits.

## Notices

- You do not have to comply with the license for elements of the material in the public domain or where your use is permitted by an applicable exception or limitation.
- No warranties are given. The license may not give you all of the permissions necessary for your intended use. For example, other rights such as publicity, privacy, or moral rights may limit how you use the material.

---

Full legal code: <https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode>
Human-readable summary: <https://creativecommons.org/licenses/by-nc-sa/4.0/>
