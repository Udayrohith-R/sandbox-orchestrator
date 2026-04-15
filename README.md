# Ephemeral Sandbox Orchestrator

> Dynamically provisions, manages, and recovers thousands of concurrent coding agent rollouts in isolated micro-containers. Built for large-scale RL training of autonomous software engineering agents.

## The Problem

Training autonomous coding agents (like Devin) via RL requires running hundreds of thousands of concurrent rollouts, each in its own isolated sandbox. Each agent needs a full development environment — filesystem, git, terminal, package manager — that is completely isolated from others. When a spot instance gets preempted mid-rollout, all progress is lost unless the agent's state can be migrated instantly.

## The Solution

```
                    ┌──────────────────────────────────┐
                    │      RL Training Loop             │
                    │   (submits rollout batches)       │
                    └───────────────┬──────────────────┘
                                    │
                    ┌───────────────▼──────────────────┐
                    │     SANDBOX ORCHESTRATOR          │
                    │                                   │
                    │  ┌─────────────────────────────┐ │
                    │  │   Task Queue (priority)      │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  Sandbox Pool Manager        │ │
                    │  │  • Pre-warmed containers     │ │
                    │  │  • Recycling (reuse > create)│ │
                    │  │  • Warm hit rate tracking    │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  State Streaming Engine      │ │
                    │  │  • Checkpoint / Resume       │ │
                    │  │  • Cross-node migration      │ │
                    │  │  • State forking             │ │
                    │  └──────────┬──────────────────┘ │
                    │             │                     │
                    │  ┌──────────▼──────────────────┐ │
                    │  │  Spot Instance Optimizer     │ │
                    │  │  • Cost-aware placement      │ │
                    │  │  • Preemption prediction     │ │
                    │  │  • Auto-migration            │ │
                    │  └─────────────────────────────┘ │
                    └──────────────────────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  On-Demand   │    │    Spot      │    │    Spot      │
    │  Nodes       │    │   Nodes      │    │   Nodes      │
    │ [sbx][sbx]   │    │ [sbx][sbx]   │    │ [sbx][sbx]   │
    │ [sbx][sbx]   │    │ [sbx][sbx]   │    │ [sbx][sbx]   │
    └──────────────┘    └──────────────┘    └──────────────┘
      (reliable)          (cheap, preemptible)
```

## Key Features

| Feature | Description |
|---|---|
| **Pre-warmed Pool** | Keeps N sandboxes ready for instant allocation (no cold-start) |
| **Container Recycling** | Cleans and reuses sandboxes (~50ms) vs fresh provision (~125ms) |
| **State Streaming** | Captures full agent state (filesystem, git, terminal, env) for migration |
| **Spot Optimization** | Routes non-critical rollouts to spot (70% cheaper), auto-migrates on preemption |
| **Periodic Checkpoints** | Snapshots every N steps, limits max state loss on failure |
| **Fault Recovery** | Failed rollouts capture failure state for debugging, graceful degradation |

## Usage

```python
from sandbox_orchestrator import SandboxOrchestrator, RolloutTask

orchestrator = SandboxOrchestrator(
    num_on_demand_nodes=3,
    num_spot_nodes=7,
    warm_pool_target=100,
    checkpoint_interval_steps=50,
    max_concurrent_rollouts=500,
)

tasks = [
    RolloutTask(
        rollout_id=f"rollout-{i}",
        agent_id="devin-v3",
        task_description="Fix authentication bug in user service",
        max_steps=1000,
        priority=5,
    )
    for i in range(1000)
]

results = orchestrator.run(tasks)
```

## Running the Demo

```bash
python sandbox_orchestrator.py
```

Runs 50 concurrent agent rollouts across 7 compute nodes (2 on-demand + 5 spot) with pre-warming, checkpointing, and simulated spot preemption events.

## Production Roadmap

- [ ] Firecracker microVM integration (replace simulation)
- [ ] CRIU-based process checkpoint/restore
- [ ] Kubernetes operator for node management
- [ ] GPU fractional allocation for agent model inference
- [ ] Apache Arrow / Plasma zero-copy state transfers (replace JSON serialization)
- [ ] cgroup v2 resource accounting with PSI-based oversubscription detection
- [ ] Prometheus + Grafana dashboard
- [ ] Integration with RL training loop (reward from rollout results)

### Frontier Extensions

- [ ] **eBPF-based Sandbox Observability:** Replace standard health checks with eBPF probes monitoring system calls within each sandbox. Detects if an agent is stuck in an infinite loop, deadlocked on I/O, or exhibiting anomalous syscall patterns — without instrumenting the agent's code or adding latency to the rollout.

- [ ] **Hierarchical State Forking:** Fork a sandbox at any checkpoint step, spawning N independent branches from the exact same environmental state. Enables the RL training loop to evaluate multiple policy variations in parallel from identical starting conditions — critical for reducing variance in reward estimation and accelerating policy search.

## Author

Uday — ML Infrastructure Engineer | Ex-Google DeepMind (Gemini)
