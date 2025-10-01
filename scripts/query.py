import faiss, json, numpy as np
from openai import OpenAI

client = OpenAI()

# Load FAISS index
index = faiss.read_index("data/processed/wtf.index")

# Load metadata
chunks = []
with open("data/processed/chunks.jsonl", "r") as f:
    for line in f:
        chunks.append(json.loads(line))

def embed(text, model="text-embedding-3-small"):
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)

def query(question, top_k=3):
    q_emb = embed(question)
    D, I = index.search(q_emb, top_k)  # distances, indices
    results = [chunks[i] for i in I[0]]
    return results

def answer(question):
    # Retrieve top chunks
    results = query(question, top_k=3)

    context = "\n\n".join([f"{r['text']}\n(Source: {r['url']})" for r in results])

    prompt = f"""You are an assistant answering using only the provided context,
which comes from transcripts of Nikhil Kamath's videos.
Question: {question}

Context:
{context}

Answer in clear, conversational style and include references with the provided YouTube links.
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

if __name__ == "__main__":
    while True:
        q = input("\nAsk something (or 'exit'): ")
        if q.strip().lower() == "exit":
            break
        print("\n--- Answer ---")
        print(answer(q))
