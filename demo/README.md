## Live demo

Interactive **GR-EN-A-DE Explorer** (Streamlit) for loading graphs and exploring
GNN edge-importance explanations.

- **Live app:** https://<your-app>.streamlit.app  ← paste your deployed URL
- **Run locally:**
  ```bash
  pip install -r demo/requirements.txt
  streamlit run demo/GR-EN-A-DE_Explorer.py
  ```

The demo has its own light dependencies in `demo/requirements.txt` (no PyTorch),
separate from the training environment in the root `requirements.txt`.

The GraphXAIN explanation tab calls a local LLM via Ollama (default
`http://localhost:11434`). Graph loading and visualization work without it; to
use explanations, run Ollama locally or set the in-app *Ollama URL* field.
