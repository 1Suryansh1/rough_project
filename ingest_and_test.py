import os
from rag import RAGPipeline
from persona import get_system_prompt
from google import genai
from google.genai import types

def test_twin():
    print("--- Starting Process ---")
    print("1. Initializing RAG Pipeline (forcing re-ingestion of raw files)")
    rag = RAGPipeline(force_reingest=True)
    
    print("\n2. Initializing Gemini API")
    if "GEMINI_API_KEY" not in os.environ:
        print("Warning: GEMINI_API_KEY not found in environment. Please set it to test generation.")
        return
        
    client = genai.Client()
    
    print("\n3. Testing Retrieval & Generation")
    test_query = "What is the path integral formulation and how does it relate to quantum electrodynamics?"
    print(f"User Query: {test_query}")
    
    retrieved_context = rag.retrieve(test_query)
    print(f"Retrieved Context length: {len(retrieved_context)} characters")
    
    system_prompt = get_system_prompt()
    augmented_prompt = f"""
{system_prompt}

### Retrieved Context (Feynman's actual words/thoughts):
{retrieved_context}

User: {test_query}
Feynman:"""

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=augmented_prompt,
        config=types.GenerateContentConfig(
            temperature=0.7,
        )
    )
    
    print(f"\nFeynman Twin Output:\n{response.text.strip()}\n")
    print("--- Process Completed Successfully ---")

if __name__ == "__main__":
    test_twin()
