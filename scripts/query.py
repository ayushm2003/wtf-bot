# scripts/query.py

import re
import json
import faiss
import numpy as np
from openai import OpenAI

client = OpenAI()

# Load FAISS + chunk metadata
index = faiss.read_index("data/processed/wtf.index")
chunks = [json.loads(l) for l in open("data/processed/chunks.jsonl", "r")]

def embed(text, model="text-embedding-3-small"):
    r = client.embeddings.create(model=model, input=text)
    return np.array(r.data[0].embedding, dtype="float32").reshape(1, -1)

def extract_names(q: str):
    # crude name grabber: "Sam Altman", "Nandan", "Nikesh Arora"
    toks = re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", q)
    return [t.lower() for t in toks]

def query(question: str, top_k: int = 5):
    q_emb = embed(question)
    D, I = index.search(q_emb, top_k)
    initial = [chunks[i] for i in I[0]]

    # simple rerank: boost chunks whose EPISODE title mentions a queried name
    keywords = extract_names(question)
    def bonus(r):
        t = r.get("title", "").lower()
        return sum(0.10 for k in keywords if k in t)

    rescored = sorted(zip(D[0], initial), key=lambda x: (x[0] - bonus(x[1])))
    return [r for _, r in rescored[:3]]

def answer(question: str):
    results = query(question, top_k=5)
    context = "\n\n".join(
        f"EPISODE: {r.get('title','')}\n{r['text']}\n(Source: {r['url']})"
        for r in results
    )
    prompt = f"""Answer ONLY from the context (WTF episodes). 
If the question names a person and the episode title matches, explicitly reference that episode in your answer.
Be neutral about who said it unless the context clearly indicates the speaker. Always include the source links.

Question: {question}

Context:
{context}
"""
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'exit'): ").strip()
        if q.lower() == "exit":
            break
        print("\n--- Answer ---")
        print(answer(q))
