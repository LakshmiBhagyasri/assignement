import json
import os
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import uvicorn

# ----------------------------
# Basic Setup
# ----------------------------

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

vector_store = []
chat_history = {}

# ----------------------------
# Request Model
# ----------------------------

class ChatRequest(BaseModel):
    sessionId: str
    message: str


# ----------------------------
# 1. Chunking Function
# ----------------------------

def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


# ----------------------------
# 2. Get Embedding
# ----------------------------

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text
    )
    return response.data[0].embedding


# ----------------------------
# 3. Cosine Similarity
# ----------------------------

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ----------------------------
# 4. Retrieve Top 3
# ----------------------------

def retrieve(query_embedding):
    scores = []

    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])
        scores.append((score, item))

    scores.sort(reverse=True, key=lambda x: x[0])

    top_chunks = scores[:3]

    # apply small threshold
    if top_chunks[0][0] < 0.70:
        return []

    return [chunk for score, chunk in top_chunks]


# ----------------------------
# 5. Build Prompt
# ----------------------------

def build_prompt(context, history, question):

    context_text = "\n\n".join([c["text"] for c in context])

    history_text = ""
    for pair in history:
        history_text += f"User: {pair['user']}\nAssistant: {pair['assistant']}\n"

    prompt = f"""
Answer the question using only the context below.
If answer is not available, say:
"I don't have enough information."

Context:
{context_text}

Conversation:
{history_text}

Question:
{question}

Answer:
"""
    return prompt


# ----------------------------
# 6. Load Documents & Embed
# ----------------------------

def load_documents():
    with open("docs.json", "r") as f:
        docs = json.load(f)

    for doc in docs:
        chunks = chunk_text(doc["content"])

        for chunk in chunks:
            embedding = get_embedding(chunk)

            vector_store.append({
                "title": doc["title"],
                "text": chunk,
                "embedding": embedding
            })

    print("Documents embedded and stored.")


# ----------------------------
# 7. API Endpoint
# ----------------------------

@app.post("/api/chat")
def chat(request: ChatRequest):

    if request.message.strip() == "":
        return {"error": "Message is empty"}

    # Create session history if new
    if request.sessionId not in chat_history:
        chat_history[request.sessionId] = []

    # Step 1: Embed user query
    query_embedding = get_embedding(request.message)

    # Step 2: Retrieve similar chunks
    retrieved = retrieve(query_embedding)

    if not retrieved:
        return {
            "reply": "I don't have enough information.",
            "tokensUsed": 0,
            "retrievedChunks": 0
        }

    # Step 3: Take last 3 history pairs
    history = chat_history[request.sessionId][-3:]

    # Step 4: Build prompt
    prompt = build_prompt(retrieved, history, request.message)

    # Step 5: LLM call
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    reply = response.choices[0].message.content
    tokens = response.usage.total_tokens

    # Step 6: Save history
    chat_history[request.sessionId].append({
        "user": request.message,
        "assistant": reply
    })

    return {
        "reply": reply,
        "tokensUsed": tokens,
        "retrievedChunks": len(retrieved)
    }


# ----------------------------
# Run Server
# ----------------------------

if __name__ == "__main__":
    load_documents()
    uvicorn.run(app, host="127.0.0.1", port=8000)