# PharmaQuery Pro

A Streamlit app for drug-gene interaction discovery across:

- ChEMBL
- DGIdb
- Open Targets
- IUPHAR
- PubChem

It merges evidence from those sources, scores compounds, and optionally uses AI to produce a cleaner ranked shortlist.

## What's new

- Added `Puter.js` browser-side AI support with user sign-in instead of a pasted developer key.
- Added persistent local provider settings and API key storage in `.streamlit/pharmaquery_state.json`.
- Added saved search defaults for the last gene and interaction type.
- Improved filter handling so empty filter results fail gracefully instead of breaking the UI.

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Start the app:

```bash
streamlit run streamlit_app.py
```

## AI options

### Puter.js

The default free option is `Puter.js - GPT-5 nano (browser auth)`.

- It runs in the browser through [Puter.js](https://docs.puter.com/).
- Users sign in with Puter inside the sidebar component.
- No developer API key is required for this mode.

### Server-side providers

The app also supports Groq, Ollama, OpenRouter, Gemini, and OpenAI.

- If you enter an API key for one of those providers, you can click `Save key locally`.
- Keys are stored in plain text at `.streamlit/pharmaquery_state.json`.

## Notes

- The local settings file is ignored by git.
- Puter.js requires browser access because sign-in uses a popup flow.
