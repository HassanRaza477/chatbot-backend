import os
from dotenv import load_dotenv

def main():
    """
    This script will perform the following steps:
    1. Load environment variables.
    2. Find and parse all markdown files from the specified directory.
    3. Chunk the text into smaller, manageable pieces.
    4. Generate embeddings for each chunk using the Gemini API.
    5. Connect to Qdrant Cloud.
    6. Create a Qdrant collection (if it doesn't exist).
    7. Upload the text chunks and their embeddings to the collection.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dotenv_path = os.path.join(os.path.dirname(script_dir), '.env.local')
    load_dotenv(dotenv_path=dotenv_path)

    # Check for environment variables
    if not os.getenv("OPENROUTER_API_KEY") or not os.getenv("QDRANT_URL") or not os.getenv("QDRANT_API_KEY") or not os.getenv("OPENROUTER_MODEL_NAME") or not os.getenv("OPENROUTER_EMBEDDING_MODEL_NAME"):
        print("Error: Required environment variables (OPENROUTER_API_KEY, OPENROUTER_MODEL_NAME, OPENROUTER_EMBEDDING_MODEL_NAME, QDRANT_URL, QDRANT_API_KEY) are not set.")
        print("Please create or update the 'embading/.env.local' file with your credentials.")
        return

    # Step 2: Find and parse all markdown files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    docs_path = os.path.join(project_root, "my-website", "docs", "chapters")
    
    documents = []
    for root, _, files in os.walk(docs_path):
        for file in files:
            if file.endswith(".md"):
                filepath = os.path.join(root, file)
                with open(filepath, "r", encoding="utf-8") as f:
                    documents.append({"path": filepath, "content": f.read()})
    
    print(f"Found {len(documents)} markdown files.")
    
    # Step 3: Chunk the documents
    chunks = []
    for doc in documents:
        paragraphs = doc["content"].split("\n\n")
        for para in paragraphs:
            if para.strip(): # Avoid empty paragraphs
                chunks.append({"path": doc["path"], "content": para})
    
    print(f"Created {len(chunks)} text chunks.")

    # Step 4: Generate embeddings
    from openai import OpenAI
    from tqdm import tqdm
    
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_EMBEDDING_MODEL_NAME = os.getenv("OPENROUTER_EMBEDDING_MODEL_NAME")

    openrouter_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    VECTOR_SIZE_EMBEDDING = 768 # For gte-base

    print("Generating embeddings with OpenRouter...")
    for chunk in tqdm(chunks):
        response = openrouter_client.embeddings.create(
            model="thenlper/gte-base", 
            input=chunk["content"],
        )
        chunk["embedding"] = response.data[0].embedding
    
    print("Embeddings generated.")

    # Step 5: Connect to Qdrant
    from qdrant_client import QdrantClient, models

    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60
    )

    COLLECTION_NAME = "my-book-bot"
    VECTOR_SIZE = VECTOR_SIZE_EMBEDDING # Use the correct size here

    print("Connected to Qdrant.")

    # Step 6: Create and upload to Qdrant collection
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
    )
    
    print(f"Collection '{COLLECTION_NAME}' created.")

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            models.PointStruct(
                id=i,
                vector=chunk["embedding"],
                payload={"text": chunk["content"], "source": chunk["path"]},
            )
        )

    qdrant_client.upload_points(
        collection_name=COLLECTION_NAME,
        points=points,
        batch_size=100, # Upload in batches
        parallel=4 # Number of parallel threads
    )
    
    print("Finished uploading points to Qdrant.")
    
    print("Ingestion script finished.")

if __name__ == "__main__":
    main()
