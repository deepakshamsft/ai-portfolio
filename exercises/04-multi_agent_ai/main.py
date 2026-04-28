"""Multi-Agent AI System - Interactive Demonstration

This script demonstrates:
1. Agent initialization and registration
2. Task decomposition and coordination
3. Message passing and state management
4. Multi-agent collaboration on complex tasks
5. Beautiful console output with metrics

Usage:
    python main.py

Expected runtime: 1-2 minutes (depends on task complexity)
Expected output: Console shows agent coordination, message flow, and metrics
"""

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.models import (
    CoordinatorAgent,
    WorkerAgent,
    ResearchAgent,
    ExperimentRunner,
    AgentConfig,
)
from src.features import (
    MessageParser,
    SharedStateManager,
    ConversationHistory,
    MessageRouter,
)

console = Console()


def main():
    """Run complete multi-agent demonstration with interactive feedback."""
    
    console.print(Panel.fit(
        "[bold cyan]Multi-Agent AI System[/bold cyan]\n"
        "Interactive Collaboration Demonstration",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 1: Initialize Infrastructure
    # ============================================
    console.print("\n[bold cyan]🔧 INITIALIZING INFRASTRUCTURE[/bold cyan]")
    
    # Message parser for validation
    parser = MessageParser()
    console.print("  ✓ Message parser initialized", style="green")
    
    # Shared state manager
    state_manager = SharedStateManager(enable_versioning=True)
    console.print("  ✓ Shared state manager initialized", style="green")
    
    # Conversation history
    history = ConversationHistory(max_history=100)
    console.print("  ✓ Conversation history initialized", style="green")
    
    # Message router
    router = MessageRouter()
    console.print("  ✓ Message router initialized", style="green")
    
    # ============================================
    # STEP 2: Initialize Agents
    # ============================================
    console.print("\n[bold cyan]🤖 INITIALIZING AGENTS[/bold cyan]")
    
    runner = ExperimentRunner()
    
    # Coordinator agent (orchestrates tasks)
    coordinator = CoordinatorAgent("coordinator")
    runner.register_agent(coordinator)
    
    # Worker agents (execute subtasks)
    worker_0 = WorkerAgent("worker_0")
    worker_1 = WorkerAgent("worker_1")
    worker_2 = WorkerAgent("worker_2")
    runner.register_agent(worker_0)
    runner.register_agent(worker_1)
    runner.register_agent(worker_2)
    
    # Research agent (gathers information)
    researcher = ResearchAgent("researcher")
    runner.register_agent(researcher)
    
    console.print(f"\n  ✓ Initialized {len(runner.agents)} agents:", style="green")
    console.print("    - 1 Coordinator (task orchestration)", style="dim")
    console.print("    - 3 Workers (parallel execution)", style="dim")
    console.print("    - 1 Researcher (information gathering)", style="dim")
    
    # ============================================
    # STEP 3: Define Tasks
    # ============================================
    console.print("\n[bold cyan]📋 DEFINING TASKS[/bold cyan]")
    
    tasks = [
        {
            "description": "Build a recommendation system for e-commerce",
            "complexity": "high",
            "requirements": ["data analysis", "ML model", "API integration"]
        },
        {
            "description": "Optimize database query performance",
            "complexity": "medium",
            "requirements": ["profiling", "indexing", "caching"]
        },
        {
            "description": "Create automated testing pipeline",
            "complexity": "medium",
            "requirements": ["test framework", "CI/CD", "coverage reports"]
        }
    ]
    
    console.print(f"  ✓ Defined {len(tasks)} tasks for demonstration:", style="green")
    for i, task in enumerate(tasks, 1):
        console.print(f"    {i}. {task['description']} ({task['complexity']} complexity)", style="dim")
    
    # ============================================
    # STEP 4: Execute Tasks
    # ============================================
    console.print("\n[bold cyan]🚀 EXECUTING TASKS[/bold cyan]")
    
    config = AgentConfig(
        max_iterations=10,
        timeout_seconds=30,
        verbose=True
    )
    
    for i, task in enumerate(tasks, 1):
        console.print(f"\n{'='*60}", style="cyan")
        console.print(f"[bold yellow]Task {i}/{len(tasks)}[/bold yellow]")
        console.print(f"{'='*60}", style="cyan")
        
        # Run multi-agent collaboration
        runner.run_experiment(task, config)
        
        # Update shared state with task result
        state_manager.update(
            agent_name="coordinator",
            key=f"task_{i}_status",
            value="completed"
        )
        
        # Log to conversation history
        for msg in runner.message_log:
            history.add_message(msg)
    
    # ============================================
    # STEP 5: Print Coordination Metrics
    # ============================================
    console.print("\n[bold cyan]📊 COORDINATION METRICS[/bold cyan]")
    
    # Agent performance
    runner.print_metrics()
    
    # Experiment leaderboard
    runner.print_leaderboard()
    
    # ============================================
    # STEP 6: Infrastructure Statistics
    # ============================================
    console.print("\n[bold cyan]🔍 INFRASTRUCTURE STATISTICS[/bold cyan]")
    
    # Message parser stats
    parser_stats = parser.get_statistics()
    console.print(f"\n📨 Message Parser:", style="yellow")
    console.print(f"  Total parsed: {parser_stats['total_parsed']}", style="dim")
    console.print(f"  Validation errors: {parser_stats['validation_errors']}", style="dim")
    console.print(f"  Success rate: {parser_stats['success_rate']:.1%}", style="dim")
    
    # State manager stats
    state_stats = state_manager.get_statistics()
    console.print(f"\n🗄️  State Manager:", style="yellow")
    console.print(f"  Total updates: {state_stats['total_updates']}", style="dim")
    console.print(f"  Conflicts: {state_stats['conflicts']}", style="dim")
    console.print(f"  State keys: {state_stats['state_keys']}", style="dim")
    console.print(f"  Versions stored: {state_stats['versions_stored']}", style="dim")
    
    # Conversation history summary
    history.print_summary()
    
    # Message router stats
    router_stats = router.get_statistics()
    console.print(f"\n🔀 Message Router:", style="yellow")
    console.print(f"  Delivered: {router_stats['delivered']}", style="dim")
    console.print(f"  Dropped: {router_stats['dropped']}", style="dim")
    console.print(f"  Delivery rate: {router_stats['delivery_rate']:.1%}", style="dim")
    
    # ============================================
    # STEP 7: Message Flow Analysis
    # ============================================
    console.print("\n[bold cyan]🔄 MESSAGE FLOW ANALYSIS[/bold cyan]")
    
    message_flow = runner.get_message_flow()
    console.print("\nTop message paths:", style="yellow")
    for path, count in sorted(message_flow.items(), key=lambda x: -x[1])[:5]:
        console.print(f"  {path}: {count} messages", style="dim")
    
    # ============================================
    # STEP 8: Emergent Behavior Analysis
    # ============================================
    console.print("\n[bold cyan]🌟 EMERGENT BEHAVIOR INSIGHTS[/bold cyan]")
    
    total_subtasks = sum(r["subtasks"] for r in runner.results)
    total_messages = sum(r["messages_exchanged"] for r in runner.results)
    avg_workers = sum(r["workers_used"] for r in runner.results) / len(runner.results)
    
    console.print("\n📈 System-level patterns:", style="yellow")
    console.print(f"  Average workers per task: {avg_workers:.1f}", style="dim")
    console.print(f"  Total subtasks generated: {total_subtasks}", style="dim")
    console.print(f"  Total messages exchanged: {total_messages}", style="dim")
    console.print(f"  Messages per subtask: {total_messages / max(total_subtasks, 1):.1f}", style="dim")
    
    # Coordination efficiency
    efficiency = (total_subtasks * 100) / max(total_messages, 1)
    console.print(f"\n🎯 Coordination efficiency: {efficiency:.1f}%", style="cyan")
    
    if efficiency > 80:
        console.print("  ✨ Excellent! Low communication overhead", style="green")
    elif efficiency > 60:
        console.print("  ✓ Good coordination efficiency", style="yellow")
    else:
        console.print("  ⚠️  High communication overhead - consider optimization", style="yellow")
    
    # ============================================
    # STEP 9: Success Summary
    # ============================================
    console.print("\n[bold green]✨ Multi-Agent Demonstration Complete![/bold green]")
    
    completed_tasks = len([r for r in runner.results if r["status"] == "distributed"])
    console.print(f"\n📊 Summary:", style="cyan")
    console.print(f"  ✓ Tasks completed: {completed_tasks}/{len(tasks)}", style="green")
    console.print(f"  ✓ Agents coordinated: {len(runner.agents)}", style="green")
    console.print(f"  ✓ Messages exchanged: {total_messages}", style="green")
    console.print(f"  ✓ State updates: {state_stats['total_updates']}", style="green")
    
    # ============================================
    # NEXT STEPS (Optional)
    # ============================================
    console.print("\n[dim]Next steps:")
    console.print("  1. Implement the TODO functions in src/models.py")
    console.print("  2. Implement the TODO functions in src/features.py")
    console.print("  3. Run this script again to see agent coordination in action")
    console.print("  4. Experiment with different task complexities and agent configurations")
    console.print("  5. Add custom agent types (CriticAgent, PlannerAgent, etc.)")
    console.print("  6. Integrate with LangChain or AutoGen for LLM-powered agents[/dim]")
    
    console.print("\n[bold cyan]Happy Hacking! 🚀[/bold cyan]\n")


if __name__ == "__main__":
    main()
