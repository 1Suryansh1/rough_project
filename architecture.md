# Architecture Diagram

This diagram explains the core logic of the Richard Feynman Digital Twin.

```mermaid
graph TD
    User([User CLI Input]) --> App[app.py Main Loop]
    
    subgraph RAG Pipeline
        App --> |1. Query| ChromaDB[(ChromaDB)]
        ChromaDB --> |2. Retrieved Chunks| App
    end

    subgraph Memory System
        App --> |3. Extract Long Term Fact| Extractor[Gemini Extractor Call]
        Extractor --> |4. Save New Fact| JSON[(memory.json)]
        JSON --> |5. Load Context| App
        App --> |Update| ShortTerm[In-Memory Short Term]
    end
    
    App --> |6. Assemble Prompt| PromptBuilder[Combine Context + Persona + Query]
    PromptBuilder --> |7. Call Model| Gemini[Gemini 2.5 Flash]
    Gemini --> |8. Feynman Response| App
    App --> Output([CLI Output])
```

## Data Flow
1. User types a question in the CLI.
2. The `RAGPipeline` queries `ChromaDB` against the local corpus for relevant Feynman quotes or ideas.
3. The `MemoryManager` uses a background Gemini call to extract new long-term facts about the user.
4. The `MemoryManager` supplies previous conversation history (short-term) and accumulated facts (long-term).
5. All elements (system prompt, RAG chunks, memory, user query) are fed to **Gemini 2.5 Flash**.
6. Gemini responds as Richard Feynman.
