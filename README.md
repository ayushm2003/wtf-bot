## Ask WTF (MVP)

Search WTF episodes (Nikhil Kamath & guests) and get brief answers with timestamped YouTube links.

Live demo: https://wtf-bot.streamlit.app/

### What it does

Semantic search over selected WTF interviews

Answers + clickable timestamps

### How it works

yt-dlp → VTT subs → cleaned JSONL

Chunks embedded (OpenAI) → FAISS index

Retrieve top chunks → GPT reply with links

### Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

or

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
streamlit run app.py
```
<br>

> Note: Still working features like speaker attribution, etc.