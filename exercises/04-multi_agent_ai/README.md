# Exercise 04: Multi-Agent AI System (TODO Version)

> **Learning Challenge:** Build a collaborative multi-agent system with task decomposition, message passing, and coordination

**Scaffolding Level:** 🟡 Medium (guided TODOs with time estimates)

See [main.py](main.py), [src/models.py](src/models.py), and [src/features.py](src/features.py) for implementation.

## Quick Start

```bash
# Install dependencies
pip install rich

# Run demonstration (after implementing TODOs)
python main.py
```

> **Note:** Infrastructure files (Dockerfile, docker-compose.yml, Makefile) have been removed for simplicity. Use setup.ps1 (Windows) or setup.sh (Unix) for environment setup.

## What You'll Build

A multi-agent system with:
- **CoordinatorAgent**: Decomposes tasks and orchestrates workers
- **WorkerAgent**: Executes atomic tasks in parallel  
- **ResearchAgent**: Gathers information with caching
- **Message passing**: Request/response/broadcast patterns
- **Shared state**: Conflict detection and versioning
- **Metrics**: Real-time coordination analytics

## Implementation Tasks

### Phase 1: Agents (src/models.py)

1. **CoordinatorAgent.process_task()** - Task decomposition and worker assignment
2. **CoordinatorAgent.respond_to_message()** - Handle worker responses
3. **WorkerAgent.process_task()** - Execute tasks independently
4. **WorkerAgent.respond_to_message()** - Process task requests
5. **ResearchAgent.process_task()** - Research with caching
6. **ExperimentRunner.run_experiment()** - Orchestrate multi-agent collaboration
7. **ExperimentRunner.print_metrics()** - Display coordination metrics

### Phase 2: Infrastructure (src/features.py)

8. **MessageParser.parse_message()** - Validate and parse messages
9. **MessageParser.extract_task_from_message()** - Extract task details
10. **MessageParser.validate_response()** - Match requests and responses
11. **SharedStateManager.update()** - State updates with conflict detection
12. **SharedStateManager.get()** - Retrieve state values
13. **SharedStateManager.lock/unlock()** - Exclusive state access
14. **ConversationHistory.add_message()** - Track conversation history
15. **ConversationHistory.get_conversation()** - Retrieve conversation thread
16. **MessageRouter.route()** - Route messages with priority handling

**Total: 17 TODOs** (see inline code comments for details)

## Core Concepts

### Agent Autonomy
Each agent has inbox, outbox, state, and metrics. They operate independently.

### Task Decomposition
- **Low complexity**: 1 subtask
- **Medium complexity**: 3 subtasks (research → execute → validate)
- **High complexity**: 5 subtasks (research → design → execute → test → validate)

### Message Passing
```python
Message(sender, recipient, content, timestamp, message_type)
```
Types: request, response, broadcast

### Coordination Pattern
```
Coordinator receives task → Decomposes → Assigns to workers (round-robin) 
→ Workers execute in parallel → Coordinator aggregates results
```

### Shared State
- Locks prevent conflicts
- Versioning enables rollback
- Timestamps track ordering

### Emergent Behavior
Simple agent rules → complex system behavior (load balancing, caching, specialization)

## Success Criteria

- [ ] All 17 TODOs implemented
- [ ] `python main.py` runs without errors
- [ ] 3 demo tasks complete with visible output
- [ ] Coordination metrics display properly
- [ ] Message routing works between agents

## Extension Ideas

**Level 1:** CriticAgent, priority queues, retry logic, task dependencies

**Level 2:** LLM integration (OpenAI), threading, agent learning, visualization

**Level 3:** Persistent state (Redis), distributed deployment, AutoGen, Prometheus

## Reference

Full documentation with detailed explanations available in files:
- [src/models.py](src/models.py) - Agent implementations with TODOs
- [src/features.py](src/features.py) - Infrastructure with TODOs
- [main.py](main.py) - Demonstration script

Each TODO is a 1-line description. See _REFERENCE/ directory for complete implementations.

---

*For questions, see [CONTRIBUTING.md](../../CONTRIBUTING.md)*
