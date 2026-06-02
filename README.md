# Richard Feynman Digital Twin

This project implements a Digital Twin of Richard Feynman as a minimalist, high-ROI terminal application. It captures his famous teaching style, curiosity, and analogies.

## Core Pillars Addressed
1. **Persona**: Handled by `persona.py`, utilizing **Gemini 2.5 Flash Lite** to adopt Feynman's unique voice and intellectual honesty.
2. **RAG Pipeline**: Handled by `rag.py`. It uses ChromaDB to dynamically embed and retrieve all `.txt` documents placed in the `data/raw/` directory, grounding the Twin's responses in his actual words and thoughts.
3. **Memory System**: Handled by `memory.py` and orchestrated in `app.py`. 
    - *Short-term*: Keeps track of the rolling conversation window.
    - *Long-term*: Extracts permanent facts about the user seamlessly in the background (using a parallel LLM call) and saves them persistently to `data/memory.json`.

## Quick Start
1. Setup a virtual environment: `python -m venv venv`
2. Activate and install dependencies: `pip install -r requirements.txt`
3. Set your Gemini API key (ensure you don't commit it!): 
   - Windows: `$env:GEMINI_API_KEY="your_key"`
   - Mac/Linux: `export GEMINI_API_KEY="your_key"`
4. Run the interactive twin: `python app.py`

## Demonstrations
You can test the system's capabilities using the included scripts:
- **`smoke_test.py`**: Runs a quick test checking RAG ingestion and connection to the Gemini API.
- **`demo_10_queries.py`**: Simulates a 10-query interaction that specifically demonstrates RAG grounding across 10 varied physics topics and proves long-term persistent memory retention.

## Project Structure
- `app.py`: Main CLI interactive loop.
- `rag.py`: ChromaDB pipeline for offline embedding and retrieval of raw documents.
- `memory.py`: Short and long-term context manager.
- `persona.py`: System prompts shaping the Twin.
- `data/raw/`: Drop any `.txt` files here and they will be automatically ingested by the RAG system.
