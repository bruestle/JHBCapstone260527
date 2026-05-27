# Healthcare Assistant (Streamlit + CrewAI)

Simple course project app to demonstrate a multi-role medical office workflow with CrewAI, Streamlit, and SQLite.

## Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app/main.py
```

## Notes

- This is demo-only software and is not for real medical use.
- Login is local and intentionally simple for class demonstration.
- If CrewAI is unavailable or not configured with an LLM, the app falls back to local canned responses so the UI still works for demos.