import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from qdrant_client import QdrantClient

# --- DEBUGGING TEST_VAR START ---
print(f"DEBUG: TEST_VAR = {os.getenv('TEST_VAR')}")
# --- DEBUGGING TEST_VAR END ---

# Initialize FastAPI app
app = FastAPI()

# Add a health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now, can be restricted later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"), 
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=60
)

openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
)

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    selected_text: str | None = None

class ChatResponse(BaseModel):
    answer: str

# API endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Handles chat requests, performs RAG, and returns a response.
    """
    print(f"Received query: {request.query}")
    if request.selected_text:
        # Use selected_text as context
        print("Using selected text as context.")
        context = request.selected_text
    else:
        # Search Qdrant for context
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
            print(f"Error during Qdrant search: {e}")
            return {"answer": "Error during Qdrant search"}
    
    print(f"Context: {context}")

    # Call OpenRouter for answer
    print("Calling OpenRouter for answer...")
    try:
        completion = openai_client.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL_NAME"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Answer the user's question based on the provided context."},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {request.query}"}
            ]
        )
        answer = completion.choices[0].message.content
        print(f"OpenRouter response: {answer}")
    except Exception as e:
        print(f"Error during OpenRouter call: {e}")
        return {"answer": "Error during OpenRouter call"}

    return ChatResponse(answer=answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

