import os
import sys
from google import genai
from google.genai import types
from persona import get_system_prompt
from memory import MemoryManager
from rag import RAGPipeline

def extract_fact(client, user_input: str) -> str:
    """Uses a separate Gemini call to determine if there's a permanent fact to remember."""
    prompt = f"""
    Analyze the following user input and extract any permanent facts about the user (e.g., their name, profession, hobbies, preferences).
    If there is a fact, return it as a short, single-sentence statement (e.g., "The user is a physics student.").
    If there are no facts worth remembering long-term, return "NONE".
    
    User input: {user_input}
    """
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash-lite',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )
        fact = response.text.strip()
        if fact and fact != "NONE":
            return fact
    except Exception as e:
        # Silently fail fact extraction to not interrupt the flow
        pass
    return None

def main():
    if "GEMINI_API_KEY" not in os.environ:
        print("Error: GEMINI_API_KEY environment variable not set.")
        sys.exit(1)

    print("Initializing Richard Feynman Digital Twin...")
    client = genai.Client()
    memory_manager = MemoryManager()
    rag_pipeline = RAGPipeline()
    system_prompt = get_system_prompt()

    print("\n--- Richard Feynman Digital Twin ---")
    print("Type 'exit' or 'quit' to leave.\n")
    print("Feynman: Hello there! What are we trying to figure out today?\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Feynman: Alright then, keep wondering! See you later.")
                break
            if not user_input.strip():
                continue

            # 1. Retrieve relevant context from RAG
            retrieved_context = rag_pipeline.retrieve(user_input)

            # 2. Extract facts for long-term memory
            fact = extract_fact(client, user_input)
            if fact:
                memory_manager.add_long_term_fact(fact)

            # 3. Get memory context
            long_term_context = memory_manager.get_long_term_context()
            short_term_context = memory_manager.get_short_term_context()

            # 4. Construct the prompt
            augmented_prompt = f"""
{system_prompt}

### Long-Term Memories about the User:
{long_term_context}

### Retrieved Context (Feynman's actual words/thoughts):
{retrieved_context}

### Conversation History:
{short_term_context}

User: {user_input}
Feynman:"""

            # 5. Generate response using Gemini 2.5 Flash
            response = client.models.generate_content(
                model='gemini-2.5-flash-lite',
                contents=augmented_prompt,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                )
            )
            
            answer = response.text.strip()
            print(f"\nFeynman: {answer}\n")

            # 6. Update short term memory
            memory_manager.add_to_short_term("user", user_input)
            memory_manager.add_to_short_term("model", answer)

        except KeyboardInterrupt:
            print("\nFeynman: Gotta go? See you later!")
            break
        except Exception as e:
            print(f"\n[System Error: {e}]")

if __name__ == "__main__":
    main()
