import streamlit as st
import faiss, json, numpy as np
from openai import OpenAI

st.set_page_config(page_title="Ask WTF", page_icon="")

client = OpenAI()
index = faiss.read_index("data/processed/wtf.index")
chunks = [json.loads(line) for line in open("data/processed/chunks.jsonl")]

def embed(text, model="text-embedding-3-small"):
    res = client.embeddings.create(model=model, input=text)
    return np.array(res.data[0].embedding, dtype="float32").reshape(1, -1)

def query(question, top_k=3):
    q_emb = embed(question)
    D, I = index.search(q_emb, top_k)
    return [chunks[i] for i in I[0]]

def answer(question):
    results = query(question)
    context = "\n\n".join([f"{r['text']}\n(Source: {r['url']})" for r in results])
    prompt = f"""Answer using only the context from Nikhil Kamath's videos.
Question: {question}

Context:
{context}

Answer in conversational style with links.
"""
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}]
    )
    return res.choices[0].message.content

@st.cache_data
def get_trained_videos():
    vids = {}  # video_id -> title
    with open("data/processed/chunks.jsonl", "r") as f:
        for line in f:
            rec = json.loads(line)
            vid = rec.get("video_id","")
            title = rec.get("title","").strip()
            if vid and title and vid not in vids:
                vids[vid] = title
    # sort by title (or keep insertion order)
    return [(v_id, vids[v_id]) for v_id in sorted(vids, key=lambda k: vids[k].lower())]

def yt_link(video_id: str):
    return f"https://www.youtube.com/watch?v={video_id}"


st.title("Ask WTF")
q = st.text_input("Ask a question:")
if q:
    st.write("### Answer")
    st.write(answer(q))

with st.expander("Trained on (episodes)"):
    for vid, title in get_trained_videos():
        st.markdown(f"- [{title}]({yt_link(vid)})")