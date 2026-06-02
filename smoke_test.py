import os
from rag import RAGPipeline
from persona import get_system_prompt
from google import genai
from google.genai import types

def run_smoke_test():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
    
    print("=== SMOKE TEST: Feynman Digital Twin ===\n")
    print("1. Initializing RAG Pipeline (Ingestion)...")
    rag = RAGPipeline(force_reingest=True)
    
    print("\n2. Initializing Gemini Client...")
    client = genai.Client()
    system_prompt = get_system_prompt()
    
    query = "Hello Mr. Feynman, are you ready to explain physics?"
    print(f"\nUser Query: {query}")
    
    retrieved_context = rag.retrieve(query, n_results=1)
    
    augmented_prompt = f"""
{system_prompt}

### Retrieved Context:
{retrieved_context}

User: {query}
Feynman:"""

    print("\n3. Firing API request to Gemini...")
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=augmented_prompt,
        config=types.GenerateContentConfig(temperature=0.7)
    )
    
    print(f"\nFeynman: {response.text.strip()}\n")
    print("=== Smoke Test Complete! ===")

if __name__ == "__main__":
    run_smoke_test()
