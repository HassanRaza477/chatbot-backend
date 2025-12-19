import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

qdrant_client = QdrantClient(
    url=os.environ.get("QDRANT_URL"), 
    api_key=os.environ.get("QDRANT_API_KEY"),
    timeout=60
)

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

class ChatRequest(BaseModel):
    query: str
    selected_text: str | None = None

class ChatResponse(BaseModel):
    answer: str

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if request.selected_text:
        context = request.selected_text
    else:
        try:
            embedding = openai_client.embeddings.create(
                model="thenlper/gte-base",
                input=request.query
            ).data[0].embedding

            search_result = qdrant_client.query_points(
                collection_name="my-book-bot",
                query=embedding,
                limit=3
            )
            context = " ".join([hit.payload["text"] for hit in search_result.points])
        except Exception as e:
            return {"answer": "Error during Qdrant search"}

    try:
        completion = openai_client.chat.completions.create(
            model=os.environ.get("OPENROUTER_MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
            ]
        )
        answer = completion.choices[0].message.content
    except Exception as e:
        return {"answer": "Error during OpenRouter call"}

    return ChatResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
