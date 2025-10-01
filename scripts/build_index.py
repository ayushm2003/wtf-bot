import json, os
import faiss
import numpy as np
from openai import OpenAI
import tiktoken

client = OpenAI()

def embed(texts, model="text-embedding-3-small"):
    res = client.embeddings.create(model=model, input=texts)
    return [e.embedding for e in res.data]

# Load cleaned segments
segments = []
with open("data/processed/segments.jsonl", "r") as f:
    for line in f:
        segments.append(json.loads(line))

# Merge into chunks (~500 chars)
chunks = []
buf, buf_meta = "", None
for seg in segments:
    txt = seg["text"]
    if not buf:
        buf_meta = seg
    buf += " " + txt
    if len(buf) > 500:
        chunks.append({
            "video_id": buf_meta["video_id"],
			"title": buf_meta.get("title",""),
            "start": buf_meta["start"],
            "url": buf_meta["url"],
            "text": buf.strip()
        })
        buf, buf_meta = "", None

print(f"Built {len(chunks)} chunks")

# Embed
texts = [c["text"] for c in chunks]
embs = embed(texts)
embs = np.array(embs).astype("float32")

# Save FAISS index
dim = len(embs[0])
index = faiss.IndexFlatL2(dim)
index.add(embs)

faiss.write_index(index, "data/processed/wtf.index")

# Save metadata
with open("data/processed/chunks.jsonl", "w") as f:
    for c in chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print("Index + chunks saved")
