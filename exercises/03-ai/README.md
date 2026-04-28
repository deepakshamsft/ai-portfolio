# Exercise 03: PizzaBot — Agentic AI Production System

> **Grand Challenge:** Build Mamma Rosa's PizzaBot achieving 32% conversion rate, $41.80 AOV, <3s p95 latency, <$0.08/conversation, 98% jailbreak resistance.

**Scaffolding Level:** 🔴 Minimal (demonstrate independence)

---

## Overview

PizzaBot is a production-ready RAG (Retrieval-Augmented Generation) conversational AI system for pizza ordering. Built with LangChain, ChromaDB, and Flask, it demonstrates enterprise-grade chatbot patterns with vector search, intent detection, and comprehensive monitoring.

### Key Features

- **RAG Pipeline**: ChromaDB vector database + sentence transformers for semantic search
- **Conversation Management**: Multi-turn dialogue with state tracking
- **Intent Detection**: Pattern-based classification for 7 intent types
- **Order Validation**: Business logic for pizza ordering constraints
- **Production Monitoring**: Prometheus metrics + MLflow tracking
- **REST API**: Flask endpoints with CORS, rate limiting
- **Docker Deployment**: Container orchestration with Prometheus & Grafana
- **Comprehensive Testing**: 90%+ code coverage with pytest

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      User Interface                      │
│                    (Chat Client/API)                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   Flask API Layer                        │
│  POST /chat  │  GET /menu  │  GET /session  │  /metrics │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Chatbot Engine                          │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   Intent    │  │     RAG      │  │    Order      │  │
│  │  Detector   │  │   Pipeline   │  │  Validator    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
┌──────────────────┐  ┌────────────────────┐
│   Vector DB      │  │    LLM Provider    │
│   (ChromaDB)     │  │  (OpenAI/Local)    │
│                  │  │                    │
│  - Embeddings    │  │  - GPT-3.5-turbo   │
│  - Semantic      │  │  - Fallback mode   │
│    Search        │  │                    │
└──────────────────┘  └────────────────────┘
```

### RAG Pipeline Flow

1. **Query**: User sends message → Intent detection
2. **Retrieve**: Query embeddings → ChromaDB search → Top-K relevant docs
3. **Augment**: Combine retrieved context + conversation history → Build prompt
4. **Generate**: LLM generates response → Return with metadata

---

## Concepts Covered

Ch.1-12 from [notes/03-ai/](../../notes/03-ai/):
- LLM fundamentals, tokenization, context windows
- Prompt engineering & few-shot learning
- Vector embeddings & semantic search
- Agentic patterns (ReAct, Reflection)
- Safety (jailbreak defense, content filtering)
- Evaluation, cost analysis, fine-tuning

---

## Directory Structure

```
exercises/03-ai/
├── src/
│   ├── __init__.py          # Package exports
│   ├── api.py               # Flask API endpoints
│   ├── models.py            # ChatbotEngine, IntentDetector, OrderValidator
│   ├── rag.py               # RAG pipeline (retrieve, augment, generate)
│   ├── embeddings.py        # EmbeddingManager + ChromaDB
│   ├── data.py              # Knowledge base loader
│   ├── evaluate.py          # Metrics: intent accuracy, latency, cost
│   ├── monitoring.py        # Prometheus metrics decorators
│   └── utils.py             # Logging, config, conversation tracking
├── tests/
│   ├── test_embeddings.py   # Vector DB tests
│   ├── test_rag.py          # RAG pipeline tests
│   ├── test_models.py       # Intent + validation tests
│   ├── test_api.py          # API endpoint tests
│   └── conftest.py          # Test fixtures
├── knowledge_base/
│   ├── menu.json            # Pizza menu (8 varieties)
│   ├── faqs.txt             # 15 Q&A pairs
│   └── policies.txt         # Delivery, refund, privacy policies
├── data/
│   └── chroma_db/           # Persistent vector database
├── logs/
│   ├── pizzabot.log         # Application logs
│   └── conversations/       # Conversation history JSONs
├── models/                  # Saved models (if using local LLM)
├── config.yaml              # Centralized configuration
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── docker-compose.yml       # Multi-container orchestration
├── prometheus.yml           # Prometheus scrape config
├── Makefile                 # Build automation
└── README.md                # This file
```

---

## Setup

### Prerequisites

- Python 3.11+
- (Optional) OpenAI API key for GPT-3.5-turbo (fallback mode available)
- Docker & Docker Compose (for containerized deployment)

### Installation

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1
```

**Manual Install:**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment Configuration

Create `.env` file (optional for OpenAI):
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

### Initialize Vector Database

```bash
make init-db
# Or manually:
python -c "from src.data import DataLoader; from src.embeddings import EmbeddingManager; ..."
```

---

## Usage

### 1. Run Development Server

```bash
# Using Makefile
make run

# Or directly
python -m src.api
```

Server runs on `http://localhost:5000`

### 2. Test API

**Health Check:**
```bash
curl http://localhost:5000/health
```

**Chat Endpoint:**
```bash
curl -X POST http://localhost:5000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I want to order a pepperoni pizza",
    "session_id": "user123"
  }'
```

**Get Menu:**
```bash
curl http://localhost:5000/menu
```

**Get Session:**
```bash
curl http://localhost:5000/session/user123
```

### 3. Example Conversations

**Order Flow:**
```
User: Hello!
Bot:  Hi! Welcome to Mamma Rosa's Pizza. How can I help you today?

User: What pizzas do you have?
Bot:  We have 8 delicious options: Margherita ($12.99), Pepperoni ($14.99), 
      Vegetarian ($13.99), Hawaiian ($14.99), Meat Lovers ($16.99), BBQ Chicken 
      ($15.99), Four Cheese ($14.99), and Spicy Italian ($15.99). All available 
      in small, medium, or large. What would you like?

User: I'll take a large pepperoni
Bot:  Great choice! One large pepperoni pizza is $14.99. Could you please 
      provide your delivery address?

User: 123 Main Street, Springfield
Bot:  Perfect! Your order for one large pepperoni pizza will be delivered to 
      123 Main Street, Springfield. Total: $14.99. Estimated delivery: 30-45 
      minutes. Confirm order?

User: Yes
Bot:  Order confirmed! Your pizza is being prepared. You can track your order 
      at... Thank you for choosing Mamma Rosa's!
```

### 4. Docker Deployment

**Build & Run:**
```bash
make docker-build
make docker-up
```

**View Logs:**
```bash
make docker-logs
```

**Access Services:**
- PizzaBot API: http://localhost:5000
- Prometheus: http://localhost:9091
- Grafana: http://localhost:3000 (admin/admin)

**Stop Services:**
```bash
make docker-down
```

---

## Testing

### Run Tests

```bash
# All tests with coverage
make test

# Specific test file
pytest tests/test_rag.py -v

# With coverage report
pytest --cov=src --cov-report=html
```

### Test Coverage Target

- **Overall**: 90%+
- **Core modules** (models.py, rag.py, api.py): 95%+

---

## Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:5000/metrics`

**Key Metrics:**
- `pizzabot_conversations_total` - Total conversations (by status)
- `pizzabot_messages_total` - Messages processed (by role, intent)
- `pizzabot_intent_predictions_total` - Intent classifications
- `pizzabot_retrieval_latency_seconds` - RAG retrieval latency
- `pizzabot_generation_latency_seconds` - LLM generation latency
- `pizzabot_api_latency_seconds` - API request latency
- `pizzabot_tokens_used_total` - Token consumption
- `pizzabot_errors_total` - Errors (by type, component)

### Grafana Dashboards

1. Open http://localhost:3000
2. Add Prometheus data source: http://prometheus:9090
3. Import dashboard or create panels:
   - Conversation volume over time
   - Intent distribution pie chart
   - Latency percentiles (p50, p95, p99)
   - Error rate
   - Token usage trends

### MLflow Tracking

```bash
mlflow ui --backend-store-uri file:./logs/mlruns
```

Access at `http://localhost:5001`

---

## Configuration

Edit `config.yaml` to customize:

```yaml
models:
  embedding_model: "all-MiniLM-L6-v2"  # Sentence transformer
  llm_model: "gpt-3.5-turbo"           # LLM for generation

rag:
  top_k: 3                              # Docs to retrieve
  similarity_threshold: 0.7             # Min similarity score

generation:
  max_tokens: 500
  temperature: 0.7

conversation:
  max_history: 10                       # Messages in context

intents:
  confidence_threshold: 0.6
  supported_intents:
    - order_pizza
    - check_menu
    - ask_question
    - track_order
    - cancel_order
    - complain
    - general_chat
```

---

## Success Criteria

### Performance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Intent Accuracy | >90% | Evaluate on test set |
| Response Relevance | >85% | Avg similarity of retrieved docs |
| P95 Latency | <500ms | End-to-end API response |
| Cost per Conversation | <$0.08 | Tokens × pricing |
| Uptime | >99.5% | Docker health checks |

### Business Metrics (Production)

- Conversion rate: 32%+ (orders / conversations)
- Average Order Value (AOV): $41.80+
- Customer satisfaction: 4.5+/5 (post-order survey)
- Jailbreak resistance: 98%+ (safety testing)

---

## Troubleshooting

### Issue: Vector DB Not Initialized
```bash
make init-db
# Or delete data/chroma_db/ and restart
```

### Issue: LLM Timeout
- Increase `request_timeout` in config.yaml
- Switch to faster model (gpt-3.5-turbo → local model)
- Enable response caching

### Issue: High Latency
- Reduce `top_k` (fewer docs to retrieve)
- Use smaller embedding model
- Enable embedding caching
- Scale with more Gunicorn workers

### Issue: Low Intent Accuracy
- Add more training examples to patterns
- Tune `confidence_threshold`
- Implement ML-based intent classifier (scikit-learn)

---

## Production Checklist

- [ ] Enable HTTPS (reverse proxy with Nginx)
- [ ] Set up rate limiting (Flask-Limiter)
- [ ] Configure API key authentication
- [ ] Enable request logging & audit trail
- [ ] Set up automated backups (vector DB, logs)
- [ ] Implement A/B testing framework
- [ ] Add content safety filters (PII detection)
- [ ] Configure alerting (PagerDuty, Slack)
- [ ] Load testing (Locust, k6)
- [ ] Document API with OpenAPI/Swagger

---

## Future Enhancements

1. **Advanced RAG**
   - Hybrid search (dense + sparse retrieval)
   - Re-ranking with cross-encoder
   - Query expansion with synonyms

2. **Multi-Agent Patterns**
   - ReAct agent for complex orders
   - Tool use (payment processing, order tracking API)
   - Self-critique & reflection loops

3. **Fine-Tuning**
   - Fine-tune LLM on pizza domain conversations
   - Train custom NER for ingredient extraction
   - Optimize embeddings for menu search

4. **Personalization**
   - User preference memory
   - Recommendation engine
   - Loyalty program integration

---

## Resources

- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)
- [Prometheus](https://prometheus.io/docs/introduction/overview/)

---

## License

MIT License - See course repository for details.

---

**Built with ❤️ for AI Portfolio Course**

## Resources

**Concept Review:**
- [notes/03-ai/](../../notes/03-ai/)
- [notes/03-ai/grand_solution.md](../../notes/03-ai/grand_solution.md)

---

**Status:** Phase 3 - Coming soon  
**Last Updated:** April 28, 2026
