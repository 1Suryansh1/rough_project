import os
import time
from rag import RAGPipeline
from persona import get_system_prompt
from memory import MemoryManager
from app import extract_fact
from google import genai
from google.genai import types

def run_demo():
    # Ensure API Key is set in the environment before pushing to GitHub!
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: Please set GEMINI_API_KEY in your environment before running.")
        return
    
    print("=== Feynman Digital Twin: 10 Query Demonstration ===\n")
    print("[1/4] Initializing RAG Pipeline (Ingesting Raw Text Files...)")
    rag = RAGPipeline(force_reingest=True)
    
    print("\n[2/4] Initializing Gemini Client...")
    client = genai.Client()
    system_prompt = get_system_prompt()
    
    print("\n[3/4] Initializing Memory System...")
    memory_manager = MemoryManager()
    
    queries = [
        "What is the core idea of the Wheeler-Feynman absorber theory?",
        "Can you explain the Path Integral formulation in simple terms?",
        "What exactly is a Parton in particle physics?",
        "How does superfluidity work? It sounds magical.",
        "Why did you invent Feynman diagrams?",
        "By the way, I should introduce myself. My name is Alice and I am studying quantum biology in grad school.",
        "What is Quantum Electrodynamics (QED) and why is it strange?",
        "Can you tell me a little bit about yourself, Richard Feynman?",
        "What do you care what other people think?",
        "Since we've been chatting a bit, do you remember my name and what I am studying?"
    ]
    
    print("\n[4/4] Running 10 Queries against the Digital Twin...\n")
    
    for i, query in enumerate(queries, 1):
        print(f"--- Query {i} ---")
        print(f"You: {query}")
        
        # 1. Retrieve RAG Context
        retrieved_context = rag.retrieve(query, n_results=3)
        
        # 2. Extract facts for Long-Term Memory
        fact = extract_fact(client, query)
        if fact:
            memory_manager.add_long_term_fact(fact)
            print(f"[Memory System] Extracted persistent fact: {fact}")
            
        # 3. Get Memories
        long_term_context = memory_manager.get_long_term_context()
        short_term_context = memory_manager.get_short_term_context()
        
        # 4. Construct Prompt
        augmented_prompt = f"""
{system_prompt}

### Long-Term Memories about the User:
{long_term_context}

### Retrieved Context (Feynman's actual words/thoughts):
{retrieved_context}

### Conversation History:
{short_term_context}

User: {query}
Feynman:"""

        # 5. Generate Response
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=augmented_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
            )
        )
        
        answer = response.text.strip()
        print(f"Feynman: {answer}\n")
        
        # 6. Update Short-Term Memory
        memory_manager.add_to_short_term("user", query)
        memory_manager.add_to_short_term("model", answer)
        
        if i < len(queries):
            print("Waiting 30 seconds for the next query...\n")
            time.sleep(30)
            
    print("\n=== Demonstration Complete ===")

if __name__ == "__main__":
    run_demo()
