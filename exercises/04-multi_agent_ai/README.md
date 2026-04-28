# Exercise 04: Multi-Agent AI System

> **Grand Challenge:** Build a production-ready multi-agent coordination system with task planning, execution, and monitoring capabilities

**Scaffolding Level:** 🔴 Minimal (demonstrate independence)

A sophisticated multi-agent system implementing task decomposition, coordinated execution, real-time evaluation, and production monitoring using LangChain, AutoGen, Redis, and Prometheus.

---

## Table of Contents

- [Architecture](#architecture)
- [Agent Roles](#agent-roles)
- [Communication Patterns](#communication-patterns)
- [Setup](#setup)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Success Criteria](#success-criteria)
- [Example Workflows](#example-workflows)
- [Monitoring](#monitoring)
- [Development](#development)

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Agent Coordinator                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Planner  │  │ Executor │  │  Critic  │  │Researcher│   │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘   │
│       │             │              │              │          │
│       └─────────────┴──────────────┴──────────────┘          │
│                          │                                    │
└──────────────────────────┼────────────────────────────────────┘
                           │
            ┌──────────────┴──────────────┐
            │                             │
    ┌───────▼────────┐          ┌────────▼────────┐
    │ Message Queue  │          │   Monitoring    │
    │    (Redis)     │          │  (Prometheus)   │
    └────────────────┘          └─────────────────┘
            │                             │
            └─────────────┬───────────────┘
                          │
                  ┌───────▼───────┐
                  │   Flask API   │
                  └───────────────┘
```

### Components

1. **Agent Coordinator**: Orchestrates multi-agent workflows with dependency management
2. **Message Queue**: Redis-based inter-agent communication
3. **Monitoring**: Prometheus metrics for system observability
4. **State Manager**: Shared state management across agents
5. **Evaluator**: Performance metrics and success criteria validation

---

## Agent Roles

### 1. PlannerAgent
**Role:** Task decomposition and planning

**Responsibilities:**
- Break down complex tasks into executable subtasks
- Identify task dependencies and execution order
- Detect parallel execution opportunities
- Estimate task complexity and duration
- Generate critical path analysis

**Configuration:** [agents/planner.yaml](agents/planner.yaml)

### 2. ExecutorAgent
**Role:** Execute atomic tasks

**Responsibilities:**
- Execute generic, computation, and data processing tasks
- Manage parallel task execution (max 3 concurrent)
- Handle task retries and error recovery
- Track execution metrics

**Configuration:** [agents/executor.yaml](agents/executor.yaml)

### 3. CriticAgent
**Role:** Evaluate results and provide feedback

**Responsibilities:**
- Evaluate task results against quality criteria
- Calculate scores: completeness, quality, accuracy, efficiency
- Generate constructive feedback
- Identify areas for improvement
- Suggest next steps

**Configuration:** [agents/critic.yaml](agents/critic.yaml)

### 4. ResearcherAgent
**Role:** Gather information and context

**Responsibilities:**
- Search knowledge base and external sources
- Synthesize findings with confidence scores
- Generate recommendations
- Maintain and update knowledge base

**Configuration:** [agents/researcher.yaml](agents/researcher.yaml)

---

## Communication Patterns

### Message Passing

Agents communicate via standardized messages:

```python
{
    "type": "task_request" | "task_response" | "feedback" | "status_update",
    "sender": "agent_name",
    "receiver": "agent_name",
    "content": {...},
    "timestamp": "ISO-8601",
    "id": "unique_id"
}
```

### Coordination Flow

1. **Planning Phase**: PlannerAgent decomposes task
2. **Research Phase** (optional): ResearcherAgent gathers context
3. **Execution Phase**: ExecutorAgent executes subtasks in order
4. **Evaluation Phase**: CriticAgent evaluates results
5. **Refinement Loop** (if needed): Re-execute based on feedback

### Shared State

StateManager provides:
- Shared state across agents
- State change history
- Snapshot/restore capabilities
- Thread-safe updates

---

## Setup

### Local Setup

**Unix/macOS/WSL:**
```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p logs data models
```

**Windows PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\setup.ps1
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
New-Item -ItemType Directory -Force logs, data, models
```

### Docker Setup

```bash
# Build and start all services
make docker-build
make docker-up

# Check logs
make docker-logs

# Stop services
make docker-down
```

### Configuration

Edit [config.yaml](config.yaml) to customize:
- Agent parameters (iterations, thresholds, etc.)
- Communication settings (Redis host, retry policy)
- LLM configuration (model, temperature, tokens)
- Monitoring settings (Prometheus port, metrics prefix)

---

## Usage

### Running the API Server

**Local:**
```bash
# Production mode
python src/api.py

# or using Make
make run

# Development mode with auto-reload
make run-dev
```

**Docker:**
```bash
docker-compose up
```

API available at: `http://localhost:5000`

### Executing Tasks via API

```bash
# Simple task
curl -X POST http://localhost:5000/task \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Research and analyze machine learning deployment patterns"
  }'

# Task with constraints
curl -X POST http://localhost:5000/task \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Optimize database query performance",
    "constraints": {
      "time_limit": 60,
      "quality_threshold": 0.85
    }
  }'
```

### Using Python Client

```python
import requests

# Execute task
response = requests.post(
    'http://localhost:5000/task',
    json={
        'description': 'Research and implement API rate limiting',
        'constraints': {'quality_threshold': 0.8}
    }
)

result = response.json()
print(f"Task ID: {result['task_id']}")
print(f"Status: {result['status']}")
print(f"Execution Time: {result['execution_time']}s")
print(f"Agents Used: {result['agents_used']}")
```

---

## API Endpoints

### POST /task
Execute multi-agent task

**Request:**
```json
{
  "description": "Task description",
  "constraints": {
    "time_limit": 60,
    "quality_threshold": 0.8
  },
  "workflow": "custom"
}
```

**Response:**
```json
{
  "task_id": "task_1234567890",
  "status": "success",
  "plan": {
    "subtasks": [...],
    "execution_order": [...],
    "complexity": 0.65
  },
  "results": [...],
  "evaluation": {
    "score": 0.87,
    "passes_threshold": true
  },
  "execution_time": 2.34,
  "agents_used": ["planner_agent", "executor_agent", "critic_agent"]
}
```

### GET /agents
List all agents and their status

### GET /metrics
Get system metrics (JSON or Prometheus format)

### GET /queue/status
Get message queue statistics

### GET /health
Health check endpoint

---

## Success Criteria

### Performance Targets

✅ **Task Completion Rate:** > 85%
- Percentage of tasks completed successfully
- Current: Monitored via `/metrics` endpoint

✅ **Coordination Efficiency:** > 0.7
- Measures effective agent collaboration
- Factors: parallel execution, time efficiency, success rate
- Current: Monitored via `/metrics` endpoint

### Evaluation Metrics

1. **Completion Rate**: Success ratio of executed tasks
2. **Coordination Efficiency**: Agent collaboration effectiveness
3. **Average Execution Time**: Per-task execution duration
4. **Agent Utilization**: Individual agent performance

Check current metrics:
```bash
curl http://localhost:5000/metrics | python -m json.tool
```

---

## Example Workflows

### 1. Research + Plan + Execute

**Task:** Research and implement caching strategy

**Flow:**
1. ResearcherAgent gathers caching best practices
2. PlannerAgent creates implementation plan
3. ExecutorAgent implements cache layer
4. CriticAgent evaluates implementation quality

### 2. Complex Analysis Workflow

**Task:** Analyze system performance and optimize bottlenecks

**Flow:**
1. PlannerAgent decomposes into:
   - Profile system performance
   - Identify bottlenecks
   - Generate optimization strategies
   - Implement optimizations
   - Validate improvements
2. ExecutorAgent executes each phase
3. CriticAgent provides continuous feedback
4. Refinement loop if quality < threshold

### 3. Parallel Execution

**Task:** Process multiple independent datasets

**Flow:**
1. PlannerAgent identifies parallel opportunities
2. ExecutorAgent (max 3 concurrent) processes datasets
3. CriticAgent evaluates aggregate results

---

## Monitoring

### Prometheus Metrics

Access Prometheus UI: `http://localhost:9090`

**Available Metrics:**
- `multiagent_agent_actions_total` - Agent action counts
- `multiagent_messages_sent_total` - Message counts
- `multiagent_tasks_total` - Task execution counts
- `multiagent_task_duration_seconds` - Task duration histogram
- `multiagent_coordination_efficiency` - Coordination score
- `multiagent_completion_rate` - Task completion rate

### Example Queries

```promql
# Average task duration
rate(multiagent_task_duration_seconds_sum[5m]) / 
rate(multiagent_task_duration_seconds_count[5m])

# Task success rate
rate(multiagent_tasks_total{status="success"}[5m])

# Coordination efficiency over time
multiagent_coordination_efficiency
```

### Logging

Logs available at:
- **Local:** Console output with structured logging
- **Docker:** `docker-compose logs -f multiagent-api`
- **Files:** `logs/` directory (if configured)

---

## Development

### Running Tests

```bash
# All tests with coverage
make test

# Fast test run (no coverage)
make test-fast

# Specific test file
pytest tests/test_agents.py -v
```

### Code Quality

```bash
# Format code
make format

# Check linting
make lint
```

### Project Structure

```
exercises/04-multi_agent_ai/
├── src/                          # Source code
│   ├── agents/                   # Agent implementations
│   │   ├── base.py              # Base agent class
│   │   ├── planner.py           # Planner agent
│   │   ├── executor.py          # Executor agent
│   │   ├── critic.py            # Critic agent
│   │   └── researcher.py        # Researcher agent
│   ├── coordinator.py           # Multi-agent coordinator
│   ├── messaging.py             # Message queue
│   ├── evaluate.py              # Evaluation metrics
│   ├── monitoring.py            # Prometheus monitoring
│   ├── utils.py                 # Utilities
│   └── api.py                   # Flask API
├── agents/                       # Agent configurations
│   ├── planner.yaml
│   ├── executor.yaml
│   ├── critic.yaml
│   └── researcher.yaml
├── tests/                        # Test suite
│   ├── test_agents.py
│   ├── test_coordinator.py
│   ├── test_messaging.py
│   └── test_api.py
├── config.yaml                   # System configuration
├── requirements.txt              # Dependencies
├── Dockerfile                    # Container image
├── docker-compose.yml            # Multi-container setup
├── prometheus.yml                # Prometheus config
└── Makefile                      # Build automation
```

### Adding New Agents

1. Create agent class extending `BaseAgent`:
```python
from src.agents.base import BaseAgent

class CustomAgent(BaseAgent):
    def process(self, task):
        # Implementation
        return result
```

2. Add agent configuration in `agents/custom.yaml`
3. Register in `config.yaml`
4. Add tests in `tests/test_agents.py`

---

## Resources

**Concept Review:**
- [notes/04-multi_agent_ai/](../../notes/04-multi_agent_ai/)
- [LangChain Documentation](https://python.langchain.com/)
- [Microsoft AutoGen](https://github.com/microsoft/autogen)

**Related Projects:**
- Exercise 01 (ML): Production patterns reference
- Exercise 03 (AI): LLM integration patterns

---

## Troubleshooting

### Redis Connection Issues
```bash
# Check Redis is running
docker-compose ps redis

# View Redis logs
docker-compose logs redis

# Test connection
redis-cli ping
```

### Agent Not Responding
1. Check agent is enabled in `config.yaml`
2. Review logs for error messages
3. Verify message queue is operational
4. Check agent status: `GET /agents`

### Low Coordination Efficiency
1. Review parallel execution opportunities
2. Check task dependencies (circular deps?)
3. Analyze agent processing times
4. Monitor via Prometheus dashboard

---

**Status:** ✅ Production Ready  
**Last Updated:** April 28, 2026
