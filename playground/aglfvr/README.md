# AI Exploration Notebooks

A collection of Jupyter notebooks exploring LLMs, RAG, Agentic AI, and advanced workflows for personal learning and experimentation.

## Overview

Self-contained notebooks with:

- No hardcoded API keys (centralized key management)
- Gitignored sensitive data (API keys, downloaded files, vector databases)
- Self-contained execution (inline data downloads)
- Documented code (markdown explanations before every code block)
- Linked to theory in notes/
- Automated setup (venv creation via scripts)

## Notebooks

### 01. LLM Basics ([01-llm-basics.ipynb](01-llm-basics.ipynb))
**Topics:** OpenAI, Google Gemini, Groq, prompt engineering, data extraction
**Related Theory:** [AI Fundamentals](../../notes/03-ai/README.md), [Prompt Engineering](../../notes/03-ai/ch02_prompt_engineering/)

Learn to:
- Authenticate with multiple LLM providers
- Send prompts and receive responses
- Apply structured prompting techniques
- Compare response formats across providers

### 02. Structured Output ([02-structured-output.ipynb](02-structured-output.ipynb))
**Topics:** Pydantic schemas, type validation, nested data extraction
**Related Theory:** [AI Fundamentals](../../notes/03-ai/README.md)

Learn to:
- Define data schemas using Pydantic models
- Enforce structured JSON output from LLMs
- Extract complex nested data structures
- Validate and type-check LLM responses

### 03. RAG Basics ([03-rag-basics.ipynb](03-rag-basics.ipynb))
**Topics:** Embeddings, vector stores, semantic search, ChromaDB
**Related Theory:** [Vector Databases](../../notes/03-ai/ch05_vector_dbs/), [RAG](../../notes/03-ai/ch04_rag_and_embeddings/)

Learn to:
- Create embeddings from text documents
- Build a vector store for semantic search
- Implement basic retrieval-augmented generation
- Combine retrieval with LLM generation

### 04. Advanced RAG ([04-advanced-rag.ipynb](04-advanced-rag.ipynb))
**Topics:** Document splitting, chunk overlap, FAISS, performance optimization
**Related Theory:** [Vector Databases](../../notes/03-ai/ch05_vector_dbs/)

Learn to:
- Split long documents into manageable chunks
- Implement chunk overlap strategies
- Use FAISS for high-performance vector search
- Save and load vector indexes

### 05. Agentic AI ([05-agentic-ai.ipynb](05-agentic-ai.ipynb))
**Topics:** Tool creation, agent architecture, function calling, decision-making
**Related Theory:** [ReAct & Semantic Kernel](../../notes/03-ai/ch06_react_and_semantic_kernel/), [Advanced Agentic Patterns](../../notes/03-ai/ch11_advanced_agentic_patterns/)

Learn to:
- Create custom tools for agents
- Build agents that use multiple tools
- Handle tool errors and edge cases
- Implement autonomous decision-making

### 06. Agent Memory ([06-agent-memory.ipynb](06-agent-memory.ipynb))
**Topics:** Conversation history, SQLite persistence, thread management
**Related Theory:** [Advanced Agentic Patterns](../../notes/03-ai/ch11_advanced_agentic_patterns/)

Learn to:
- Implement in-memory and persistent memory
- Use SQLite for conversation checkpointing
- Manage conversation threads and sessions
- Query stored conversations

### 07. LangGraph ([07-langgraph.ipynb](07-langgraph.ipynb))
**Topics:** State machines, conditional routing, workflow graphs, multi-agent systems
**Related Theory:** [Advanced Agentic Patterns](../../notes/03-ai/ch11_advanced_agentic_patterns/)

Learn to:
- Build state machines with nodes and edges
- Create conditional routing logic
- Implement complex multi-agent workflows
- Visualize execution graphs

## Quick Start

### 1. Setup Environment

**Windows (PowerShell):**
```powershell
.\setup.ps1
```

**Mac/Linux (Bash):**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment (`.venv/`)
- Install all dependencies
- Create `api_keys.py` from template
- Create `.data/` directory for downloads

### 2. Configure API Keys

Edit `api_keys.py` and add your actual API keys:

```python
# Get your keys from:
OPENAI_API_KEY = "sk-..."        # https://platform.openai.com/api-keys
GOOGLE_API_KEY = "AIza..."        # https://ai.google.dev/
GROQ_API_KEY = "gsk_..."          # https://console.groq.com/keys
TAVILY_API_KEY = "tvly-..."       # https://tavily.com/
```

### 3. Activate Virtual Environment

**Windows:**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Mac/Linux:**
```bash
source .venv/bin/activate
```

### 4. Open Jupyter

```bash
jupyter notebook
```

Navigate to any notebook (`01-llm-basics.ipynb`, etc.) and run!

## Security Features

All sensitive data is gitignored:

```
.venv/              # Virtual environment
api_keys.py         # Your API keys
.env                # Environment variables
.data/              # Downloaded files
vector_db/          # Vector databases
*.db                # SQLite checkpoints
```

**Never commit:**
- API keys
- Downloaded data
- User-generated content
- Database files

## Dependencies

Core libraries (see [requirements.txt](requirements.txt)):

- **LLM Providers:** `openai`, `google-genai`, `groq`
- **LangChain:** `langchain`, `langchain-core`, `langchain-community`
- **Vector Stores:** `chromadb`, `faiss-cpu`
- **Embeddings:** `sentence-transformers`
- **Document Processing:** `pypdf`, `wikipedia`
- **Agentic AI:** `langgraph`, `langchain-tavily`
- **Data Science:** `pydantic`, `yfinance`

## Repository Structure

```
playground/aglfvr/
├── 01-llm-basics.ipynb          # LLM provider basics
├── 02-structured-output.ipynb   # Pydantic schemas
├── 03-rag-basics.ipynb          # Basic RAG
├── 04-advanced-rag.ipynb        # Advanced RAG techniques
├── 05-agentic-ai.ipynb          # Agent tools and decision-making
├── 06-agent-memory.ipynb        # Persistent memory
├── 07-langgraph.ipynb           # Complex workflows
├── api_keys_template.py         # API key template
├── api_keys.py                  # Your keys (gitignored)
├── requirements.txt             # Python dependencies
├── setup.ps1                    # Windows setup script
├── setup.sh                     # Mac/Linux setup script
├── .gitignore                   # Security rules
├── .venv/                       # Virtual environment (gitignored)
└── .data/                       # Downloaded data (gitignored)
```

## Learning Path

**Beginner:**
1. Start with [01-llm-basics.ipynb](01-llm-basics.ipynb) to understand LLM providers
2. Move to [02-structured-output.ipynb](02-structured-output.ipynb) for type-safe extraction
3. Try [03-rag-basics.ipynb](03-rag-basics.ipynb) to build your first RAG system

**Intermediate:**
4. Learn advanced retrieval in [04-advanced-rag.ipynb](04-advanced-rag.ipynb)
5. Build autonomous agents in [05-agentic-ai.ipynb](05-agentic-ai.ipynb)
6. Add memory in [06-agent-memory.ipynb](06-agent-memory.ipynb)

**Advanced:**
7. Master complex workflows in [07-langgraph.ipynb](07-langgraph.ipynb)

## Links to Theory

Each notebook references relevant chapters in the `notes/` directory:

- **AI Fundamentals:** [notes/03-ai/](../../notes/03-ai/README.md)
- **Prompt Engineering:** [notes/03-ai/ch02_prompt_engineering/](../../notes/03-ai/ch02_prompt_engineering/)
- **Vector Databases:** [notes/03-ai/ch05_vector_dbs/](../../notes/03-ai/ch05_vector_dbs/)
- **Multi-Agent AI:** [notes/04-multi_agent_ai/](../../notes/04-multi_agent_ai/README.md)

## Troubleshooting

### Virtual Environment Not Activating
**Windows:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
```

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

### Missing API Keys
If you see "API key not configured" errors:
1. Check `api_keys.py` exists
2. Ensure keys don't start with "your-"
3. Restart the Jupyter kernel

### Import Errors
If you see "ModuleNotFoundError":
```bash
# Ensure venv is activated
pip install -r requirements.txt
```

### Data Directory Issues
If notebooks can't save data:
```bash
mkdir -p .data
```
