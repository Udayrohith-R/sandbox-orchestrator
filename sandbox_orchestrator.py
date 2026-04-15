# Ephemeral Sandbox Orchestrator for Distributed Agent Rollouts
# ==============================================================
# Dynamically provisions, manages, and recovers thousands of concurrent
# coding agent rollouts in isolated micro-containers. Designed for
# large-scale RL training of autonomous software engineering agents.
#
# Architecture:
# 1. Sandbox Pool Manager — pre-warms and recycles micro-containers
# 2. State Streaming — snapshots agent progress for cross-node migration
# 3. Spot Instance Optimizer — migrates sandboxes to cheapest compute
# 4. Fault Tolerance — detects sandbox failures, auto-recovers with state
# 5. Telemetry — real-time metrics on rollout throughput and utilization
#
# Author: Uday
# Target: Cognition (Devin) — Research Engineer, Infrastructure

import os
import time
import json
import uuid
import hashlib
import threading
import logging
import random
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import deque, defaultdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import queue

# ============================================================
# PRODUCTION NOTES ON GIL & SERIALIZATION
# ============================================================
# 1. GIL CONTENTION: This prototype uses ThreadPoolExecutor because
#    rollout simulation is I/O-bound (time.sleep). In production,
#    agent rollouts involve heavy data serialization, log processing,
#    and state capture that are CPU-bound. MUST use ProcessPoolExecutor
#    or distributed actors (Ray, Celery) to avoid GIL starvation.
#
# 2. STATE TRANSFER: AgentSnapshot currently serializes to JSON/bytes.
#    At scale (thousands of snapshots/min), this becomes a CPU bottleneck.
#    Production should use Apache Arrow / Plasma for zero-copy state
#    transfers, or multiprocessing.shared_memory for cross-process
#    snapshot access without serialization overhead.
#
# 3. RESOURCE ACCOUNTING: Simple integer subtraction for CPU/memory.
#    Production should use cgroup v2 monitoring for real-time resource
#    tracking, handling bursty agent workloads and oversubscription.
# ============================================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger("sandbox_orchestrator")


# ============================================================
# PART 1: Core Data Structures
# ============================================================

class SandboxState(Enum):
    """Lifecycle states for a sandbox micro-container."""
    COLD = "cold"               # Not yet provisioned
    WARMING = "warming"         # Container being created
    READY = "ready"             # Warm, waiting for assignment
    RUNNING = "running"         # Agent rollout in progress
    SNAPSHOTTING = "snapshotting"  # State being captured for migration
    MIGRATING = "migrating"     # Moving to different node
    COMPLETED = "completed"     # Rollout finished successfully
    FAILED = "failed"           # Rollout failed
    RECYCLING = "recycling"     # Being cleaned for reuse


class NodeType(Enum):
    """Compute node types with different cost/reliability profiles."""
    ON_DEMAND = "on_demand"     # Reliable, expensive
    SPOT = "spot"               # Cheap, can be preempted
    RESERVED = "reserved"       # Pre-purchased, cheapest for steady load


@dataclass
class AgentSnapshot:
    """
    Serializable checkpoint of an agent's execution state.
    Enables pause/resume and cross-node migration.
    
    Production optimization: For high-throughput snapshot transfer (thousands/min),
    replace JSON serialization with Apache Arrow / Plasma zero-copy IPC, or use
    multiprocessing.shared_memory to avoid serialization overhead entirely.
    Filesystem state should use btrfs/ZFS snapshots or overlayfs copy-on-write
    rather than full serialization.
    """
    snapshot_id: str
    sandbox_id: str
    agent_id: str
    step: int
    filesystem_hash: str       # Hash of sandbox filesystem state
    memory_size_mb: float      # Size of serialized state
    env_variables: Dict[str, str]
    open_files: List[str]      # Files the agent had open
    git_state: Dict            # Branch, commit, staged changes
    terminal_history: List[str]  # Recent commands executed
    timestamp: float
    
    def serialize(self) -> bytes:
        """
        Serialize snapshot for storage/transfer.
        
        TODO (Production): Replace JSON serialization with zero-copy approach:
        - Same-node: multiprocessing.shared_memory (no serialization)
        - Cross-node: Apache Arrow Flight RPC (columnar, zero-copy deserialize)
        - Filesystem: btrfs send/receive for incremental snapshot transfer
        """
        return json.dumps({
            "snapshot_id": self.snapshot_id,
            "sandbox_id": self.sandbox_id,
            "agent_id": self.agent_id,
            "step": self.step,
            "filesystem_hash": self.filesystem_hash,
            "memory_size_mb": self.memory_size_mb,
            "env_variables": self.env_variables,
            "open_files": self.open_files,
            "git_state": self.git_state,
            "terminal_history": self.terminal_history,
            "timestamp": self.timestamp,
        }).encode()
    
    @classmethod
    def deserialize(cls, data: bytes) -> "AgentSnapshot":
        d = json.loads(data.decode())
        return cls(**d)


@dataclass
class Sandbox:
    """
    Represents a single isolated micro-container for agent execution.
    In production: Firecracker microVM or gVisor container.
    """
    sandbox_id: str
    node_id: str
    state: SandboxState = SandboxState.COLD
    agent_id: Optional[str] = None
    rollout_id: Optional[str] = None
    
    # Resource allocation
    cpu_cores: int = 2
    memory_mb: int = 4096
    disk_mb: int = 10240
    gpu_fraction: float = 0.0   # Optional GPU slice for model inference
    
    # Lifecycle tracking
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    total_steps: int = 0
    
    # State management
    last_snapshot: Optional[AgentSnapshot] = None
    snapshot_count: int = 0
    
    # Health
    health_checks_passed: int = 0
    health_checks_failed: int = 0
    last_health_check: float = 0.0


@dataclass
class ComputeNode:
    """
    A compute node that hosts multiple sandboxes.
    
    Production resource accounting: Simple integer subtraction below is a
    prototype approximation. Real agent workloads are bursty — an agent
    compiling code spikes CPU briefly, then idles during LLM inference.
    Production should use cgroup v2 controllers for:
    - cpu.max: Hard CPU limits per sandbox
    - memory.max: OOM kill protection
    - io.max: Disk I/O throttling
    - cpu.pressure / memory.pressure: PSI-based oversubscription detection
    This enables safe 2-3x oversubscription of CPU while protecting against
    noisy-neighbor effects between concurrent agent rollouts.
    """
    node_id: str
    node_type: NodeType
    region: str
    
    # Capacity (prototype: simple accounting)
    # TODO (Production): Replace with cgroup v2 real-time resource tracking
    total_cpu: int = 96
    total_memory_mb: int = 384000    # 384 GB
    total_disk_mb: int = 2000000     # 2 TB
    available_cpu: int = 96
    available_memory_mb: int = 384000
    available_disk_mb: int = 2000000
    
    # Cost
    cost_per_hour: float = 0.0
    
    # State
    sandboxes: Dict[str, Sandbox] = field(default_factory=dict)
    is_healthy: bool = True
    preemption_probability: float = 0.0  # For spot instances
    
    # Metrics
    utilization: float = 0.0
    last_heartbeat: float = 0.0


@dataclass
class RolloutTask:
    """A coding agent rollout to be executed."""
    rollout_id: str
    agent_id: str
    task_description: str
    max_steps: int = 1000
    timeout_seconds: float = 3600.0
    priority: int = 0           # Higher = more important
    requires_gpu: bool = False
    
    # For resumption from snapshot
    resume_from_snapshot: Optional[AgentSnapshot] = None


@dataclass
class RolloutResult:
    """Result of a completed agent rollout."""
    rollout_id: str
    sandbox_id: str
    agent_id: str
    success: bool
    total_steps: int
    total_time_seconds: float
    final_state: Optional[AgentSnapshot] = None
    error: Optional[str] = None
    metrics: Dict = field(default_factory=dict)


# ============================================================
# PART 2: Sandbox Pool Manager
# ============================================================

class SandboxPoolManager:
    """
    Manages a pool of pre-warmed sandboxes for instant agent assignment.
    
    Key optimizations:
    1. Pre-warming: Keep N sandboxes ready to reduce cold-start latency
    2. Recycling: Clean and reuse sandboxes instead of destroying/creating
    3. Tiered allocation: Use spot instances for non-critical rollouts
    
    In production: interfaces with Firecracker API or containerd/gVisor
    """
    
    def __init__(
        self,
        target_warm_pool: int = 50,
        max_sandboxes: int = 10000,
        sandbox_cpu: int = 2,
        sandbox_memory_mb: int = 4096,
        warm_interval_seconds: float = 5.0,
    ):
        self.target_warm_pool = target_warm_pool
        self.max_sandboxes = max_sandboxes
        self.sandbox_cpu = sandbox_cpu
        self.sandbox_memory_mb = sandbox_memory_mb
        self.warm_interval = warm_interval_seconds
        
        # Sandbox tracking
        self._sandboxes: Dict[str, Sandbox] = {}
        self._warm_pool: deque = deque()  # Ready sandboxes
        self._active: Dict[str, Sandbox] = {}  # Running sandboxes
        self._recycling: deque = deque()  # Being cleaned
        
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Metrics
        self._total_created = 0
        self._total_recycled = 0
        self._total_failed = 0
        self._cold_starts = 0   # Assignments with no warm sandbox available
        self._warm_hits = 0     # Assignments served from warm pool
    
    def provision_sandbox(self, node: ComputeNode) -> Optional[Sandbox]:
        """
        Create a new sandbox on a compute node.
        In production: calls Firecracker API to create microVM.
        """
        # Check node capacity
        if (node.available_cpu < self.sandbox_cpu or 
            node.available_memory_mb < self.sandbox_memory_mb):
            return None
        
        sandbox_id = f"sbx-{uuid.uuid4().hex[:12]}"
        
        sandbox = Sandbox(
            sandbox_id=sandbox_id,
            node_id=node.node_id,
            state=SandboxState.WARMING,
            cpu_cores=self.sandbox_cpu,
            memory_mb=self.sandbox_memory_mb,
            created_at=time.time(),
        )
        
        # Simulate container creation latency
        # Firecracker: ~125ms, gVisor: ~200ms, Docker: ~500ms
        time.sleep(random.uniform(0.1, 0.15))  # Simulating Firecracker speed
        
        # Reserve resources on node
        node.available_cpu -= self.sandbox_cpu
        node.available_memory_mb -= self.sandbox_memory_mb
        node.sandboxes[sandbox_id] = sandbox
        
        sandbox.state = SandboxState.READY
        
        with self._lock:
            self._sandboxes[sandbox_id] = sandbox
            self._total_created += 1
        
        return sandbox
    
    def acquire_sandbox(self) -> Optional[Sandbox]:
        """
        Get a sandbox for a rollout. Prefers warm pool, falls back to cold start.
        Returns None if no capacity available.
        """
        with self._lock:
            # Try warm pool first
            if self._warm_pool:
                sandbox = self._warm_pool.popleft()
                sandbox.state = SandboxState.RUNNING
                sandbox.started_at = time.time()
                self._active[sandbox.sandbox_id] = sandbox
                self._warm_hits += 1
                return sandbox
            
            self._cold_starts += 1
        
        # No warm sandbox — will need cold provisioning
        return None
    
    def release_sandbox(self, sandbox: Sandbox, recycle: bool = True):
        """Return a sandbox to the pool for recycling or destruction."""
        with self._lock:
            self._active.pop(sandbox.sandbox_id, None)
            
            if recycle and len(self._sandboxes) <= self.max_sandboxes:
                sandbox.state = SandboxState.RECYCLING
                self._recycling.append(sandbox)
            else:
                sandbox.state = SandboxState.COMPLETED
                self._sandboxes.pop(sandbox.sandbox_id, None)
    
    def recycle_sandbox(self, sandbox: Sandbox):
        """
        Clean a sandbox for reuse. Faster than creating a new one.
        In production: resets filesystem to base image, clears env.
        """
        # Simulate cleanup (faster than fresh provision)
        time.sleep(random.uniform(0.02, 0.05))
        
        sandbox.agent_id = None
        sandbox.rollout_id = None
        sandbox.state = SandboxState.READY
        sandbox.last_snapshot = None
        sandbox.total_steps = 0
        sandbox.health_checks_passed = 0
        sandbox.health_checks_failed = 0
        
        with self._lock:
            self._warm_pool.append(sandbox)
            self._total_recycled += 1
    
    def add_to_warm_pool(self, sandbox: Sandbox):
        """Add a newly provisioned sandbox to the warm pool."""
        with self._lock:
            sandbox.state = SandboxState.READY
            self._warm_pool.append(sandbox)
    
    def get_metrics(self) -> Dict:
        with self._lock:
            return {
                "total_sandboxes": len(self._sandboxes),
                "warm_pool_size": len(self._warm_pool),
                "active_rollouts": len(self._active),
                "recycling_queue": len(self._recycling),
                "total_created": self._total_created,
                "total_recycled": self._total_recycled,
                "total_failed": self._total_failed,
                "warm_hit_rate": (
                    self._warm_hits / (self._warm_hits + self._cold_starts)
                    if (self._warm_hits + self._cold_starts) > 0 else 0
                ),
                "cold_starts": self._cold_starts,
            }


# ============================================================
# PART 3: State Streaming & Migration Engine
# ============================================================

class StateStreamingEngine:
    """
    Captures, transfers, and restores agent execution state across nodes.
    
    Enables:
    1. Cross-node migration (move agent from preempted spot to on-demand)
    2. Checkpoint/resume for long-running rollouts
    3. State forking (run multiple strategies from same checkpoint)
    
    In production: uses CRIU (Checkpoint/Restore In Userspace) for full
    process state capture, or custom snapshotting for lighter-weight migration.
    """
    
    def __init__(self, snapshot_store_path: str = "/tmp/snapshots"):
        self.store_path = snapshot_store_path
        os.makedirs(snapshot_store_path, exist_ok=True)
        
        self._snapshots: Dict[str, AgentSnapshot] = {}
        self._lock = threading.Lock()
        
        # Metrics
        self._total_snapshots = 0
        self._total_restorations = 0
        self._total_migrations = 0
        self._avg_snapshot_time_ms = 0
        self._avg_restore_time_ms = 0
    
    def capture_snapshot(self, sandbox: Sandbox) -> AgentSnapshot:
        """
        Capture the full execution state of a sandbox.
        In production: CRIU checkpoint or filesystem snapshot + process state.
        """
        start = time.perf_counter()
        
        sandbox.state = SandboxState.SNAPSHOTTING
        
        # Simulate state capture
        # Real implementation captures:
        # - Filesystem diff from base image
        # - Process memory pages (via CRIU)
        # - Open file descriptors
        # - Network connections
        # - Environment variables
        time.sleep(random.uniform(0.05, 0.15))  # 50-150ms for snapshot
        
        snapshot = AgentSnapshot(
            snapshot_id=f"snap-{uuid.uuid4().hex[:12]}",
            sandbox_id=sandbox.sandbox_id,
            agent_id=sandbox.agent_id or "",
            step=sandbox.total_steps,
            filesystem_hash=hashlib.sha256(
                f"{sandbox.sandbox_id}:{sandbox.total_steps}".encode()
            ).hexdigest()[:16],
            memory_size_mb=random.uniform(50, 500),
            env_variables={"WORKSPACE": "/home/agent", "TASK_ID": sandbox.rollout_id or ""},
            open_files=["/home/agent/main.py", "/home/agent/test.py"],
            git_state={
                "branch": "agent-work",
                "commit": hashlib.sha256(str(time.time()).encode()).hexdigest()[:8],
                "staged_files": random.randint(0, 5),
                "modified_files": random.randint(1, 10),
            },
            terminal_history=[
                "cd /home/agent",
                "git checkout -b agent-work",
                f"python main.py --step {sandbox.total_steps}",
            ],
            timestamp=time.time(),
        )
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Store snapshot
        with self._lock:
            self._snapshots[snapshot.snapshot_id] = snapshot
            self._total_snapshots += 1
            # Running average
            n = self._total_snapshots
            self._avg_snapshot_time_ms = (
                (self._avg_snapshot_time_ms * (n - 1) + elapsed_ms) / n
            )
        
        sandbox.last_snapshot = snapshot
        sandbox.snapshot_count += 1
        sandbox.state = SandboxState.RUNNING
        
        logger.debug(f"Snapshot captured: {snapshot.snapshot_id} "
                     f"({snapshot.memory_size_mb:.0f}MB, {elapsed_ms:.1f}ms)")
        
        return snapshot
    
    def restore_snapshot(
        self, snapshot: AgentSnapshot, target_sandbox: Sandbox
    ) -> bool:
        """
        Restore an agent's state into a different sandbox.
        In production: CRIU restore or filesystem overlay + process restart.
        """
        start = time.perf_counter()
        
        # Simulate state restoration
        time.sleep(random.uniform(0.05, 0.2))  # 50-200ms for restore
        
        # Apply snapshot to target sandbox
        target_sandbox.agent_id = snapshot.agent_id
        target_sandbox.total_steps = snapshot.step
        target_sandbox.last_snapshot = snapshot
        target_sandbox.state = SandboxState.RUNNING
        target_sandbox.started_at = time.time()
        
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        with self._lock:
            self._total_restorations += 1
            n = self._total_restorations
            self._avg_restore_time_ms = (
                (self._avg_restore_time_ms * (n - 1) + elapsed_ms) / n
            )
        
        logger.debug(f"Snapshot restored: {snapshot.snapshot_id} → "
                     f"{target_sandbox.sandbox_id} ({elapsed_ms:.1f}ms)")
        
        return True
    
    def migrate(
        self, sandbox: Sandbox, target_sandbox: Sandbox
    ) -> bool:
        """
        Full migration: snapshot source → restore to target → release source.
        Used when spot instance is about to be preempted.
        """
        sandbox.state = SandboxState.MIGRATING
        
        # Capture state from source
        snapshot = self.capture_snapshot(sandbox)
        
        # Restore to target
        success = self.restore_snapshot(snapshot, target_sandbox)
        
        if success:
            with self._lock:
                self._total_migrations += 1
            logger.info(f"Migration complete: {sandbox.sandbox_id} → "
                       f"{target_sandbox.sandbox_id} (step {snapshot.step})")
        
        return success
    
    def get_metrics(self) -> Dict:
        with self._lock:
            return {
                "total_snapshots": self._total_snapshots,
                "total_restorations": self._total_restorations,
                "total_migrations": self._total_migrations,
                "avg_snapshot_time_ms": round(self._avg_snapshot_time_ms, 1),
                "avg_restore_time_ms": round(self._avg_restore_time_ms, 1),
                "stored_snapshots": len(self._snapshots),
            }


# ============================================================
# PART 4: Spot Instance Optimizer
# ============================================================

class SpotInstanceOptimizer:
    """
    Optimizes sandbox placement across on-demand and spot instances
    to minimize cost while maintaining rollout reliability.
    
    Strategy:
    - Non-critical RL rollouts → spot instances (70-90% cheaper)
    - Critical evaluation runs → on-demand instances
    - Pre-emptively migrates when spot probability exceeds threshold
    """
    
    def __init__(
        self,
        preemption_threshold: float = 0.7,
        cost_weight: float = 0.6,
        reliability_weight: float = 0.4,
    ):
        self.preemption_threshold = preemption_threshold
        self.cost_weight = cost_weight
        self.reliability_weight = reliability_weight
        
        self._preemption_warnings: deque = deque(maxlen=1000)
        self._migrations_triggered = 0
        self._cost_saved = 0.0
    
    def select_node(
        self, 
        nodes: List[ComputeNode], 
        task: RolloutTask,
    ) -> Optional[ComputeNode]:
        """
        Select the optimal node for a rollout task.
        Balances cost vs reliability based on task priority.
        """
        eligible = [n for n in nodes if n.is_healthy and n.available_cpu >= 2]
        
        if not eligible:
            return None
        
        scored = []
        for node in eligible:
            # Cost score (lower is better, normalized)
            max_cost = max(n.cost_per_hour for n in eligible) or 1
            cost_score = 1.0 - (node.cost_per_hour / max_cost)
            
            # Reliability score (lower preemption is better)
            reliability_score = 1.0 - node.preemption_probability
            
            # High-priority tasks heavily weight reliability
            if task.priority > 5:
                weight_cost = 0.2
                weight_reliability = 0.8
            else:
                weight_cost = self.cost_weight
                weight_reliability = self.reliability_weight
            
            total_score = (weight_cost * cost_score + 
                          weight_reliability * reliability_score)
            
            scored.append((node, total_score))
        
        # Return highest scored node
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]
    
    def check_preemption_risk(self, nodes: List[ComputeNode]) -> List[ComputeNode]:
        """
        Identify nodes at risk of spot preemption.
        Returns nodes that should have their sandboxes migrated.
        """
        at_risk = []
        for node in nodes:
            if (node.node_type == NodeType.SPOT and 
                node.preemption_probability > self.preemption_threshold):
                at_risk.append(node)
                self._preemption_warnings.append({
                    "node_id": node.node_id,
                    "probability": node.preemption_probability,
                    "timestamp": time.time(),
                    "active_sandboxes": len(node.sandboxes),
                })
        return at_risk
    
    def get_metrics(self) -> Dict:
        return {
            "preemption_warnings": len(self._preemption_warnings),
            "migrations_triggered": self._migrations_triggered,
            "estimated_cost_saved": round(self._cost_saved, 2),
        }


# ============================================================
# PART 5: Sandbox Orchestrator (Main System)
# ============================================================

class SandboxOrchestrator:
    """
    Main orchestration system for distributed agent rollouts.
    
    Manages the full lifecycle:
    1. Receive rollout tasks from RL training loop
    2. Allocate sandboxes from warm pool or provision new ones
    3. Execute agent rollouts with periodic checkpointing
    4. Handle spot preemption via state migration
    5. Collect results for RL reward computation
    
    Usage:
        orchestrator = SandboxOrchestrator(
            num_nodes=10,
            sandboxes_per_node=50,
            warm_pool_target=100,
        )
        
        # Submit rollout tasks
        tasks = [RolloutTask(...) for _ in range(1000)]
        results = orchestrator.run(tasks)
    """
    
    def __init__(
        self,
        num_on_demand_nodes: int = 3,
        num_spot_nodes: int = 7,
        sandboxes_per_node: int = 40,
        warm_pool_target: int = 50,
        checkpoint_interval_steps: int = 100,
        max_concurrent_rollouts: int = 500,
    ):
        # Compute nodes
        self.nodes: List[ComputeNode] = []
        self._setup_nodes(num_on_demand_nodes, num_spot_nodes)
        
        # Sub-systems
        self.pool_manager = SandboxPoolManager(
            target_warm_pool=warm_pool_target,
            max_sandboxes=num_on_demand_nodes * sandboxes_per_node + 
                          num_spot_nodes * sandboxes_per_node,
        )
        self.state_engine = StateStreamingEngine()
        self.spot_optimizer = SpotInstanceOptimizer()
        
        # Config
        self.checkpoint_interval = checkpoint_interval_steps
        self.max_concurrent = max_concurrent_rollouts
        
        # Task management
        self._task_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._active_rollouts: Dict[str, Tuple[Sandbox, RolloutTask]] = {}
        self._results: List[RolloutResult] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Rollout execution pool
        # NOTE: Using ThreadPoolExecutor here because simulation is I/O-bound
        # (time.sleep). Production agent rollouts involve heavy CPU-bound work
        # (data serialization, log processing, state capture) that MUST use
        # ProcessPoolExecutor or distributed actors (Ray) to bypass GIL contention.
        # At Cognition's scale (100K+ concurrent rollouts), each rollout runs in
        # its own Firecracker microVM process anyway, making GIL irrelevant —
        # the orchestrator only needs to manage IPC, not execute rollout logic.
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent_rollouts)
        
        # Metrics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_migrated = 0
    
    def _setup_nodes(self, num_on_demand: int, num_spot: int):
        """Initialize compute nodes."""
        for i in range(num_on_demand):
            self.nodes.append(ComputeNode(
                node_id=f"node-od-{i}",
                node_type=NodeType.ON_DEMAND,
                region="us-west-2",
                cost_per_hour=3.06,  # p3.2xlarge equivalent
                preemption_probability=0.0,
                last_heartbeat=time.time(),
            ))
        
        for i in range(num_spot):
            self.nodes.append(ComputeNode(
                node_id=f"node-spot-{i}",
                node_type=NodeType.SPOT,
                region="us-west-2",
                cost_per_hour=0.92,  # 70% discount
                preemption_probability=random.uniform(0.05, 0.3),
                last_heartbeat=time.time(),
            ))
    
    def _pre_warm_pool(self):
        """Pre-warm sandbox pool for fast allocation."""
        logger.info(f"Pre-warming {self.pool_manager.target_warm_pool} sandboxes...")
        
        warm_count = 0
        for _ in range(self.pool_manager.target_warm_pool):
            # Prefer spot nodes for warm pool (cheaper to keep idle)
            spot_nodes = [n for n in self.nodes 
                         if n.node_type == NodeType.SPOT and n.available_cpu >= 2]
            
            if not spot_nodes:
                spot_nodes = [n for n in self.nodes if n.available_cpu >= 2]
            
            if not spot_nodes:
                break
            
            node = random.choice(spot_nodes)
            sandbox = self.pool_manager.provision_sandbox(node)
            
            if sandbox:
                self.pool_manager.add_to_warm_pool(sandbox)
                warm_count += 1
        
        logger.info(f"Pre-warmed {warm_count} sandboxes")
    
    def _execute_rollout(self, sandbox: Sandbox, task: RolloutTask) -> RolloutResult:
        """
        Execute a single agent rollout in a sandbox.
        Handles checkpointing and failure recovery.
        """
        sandbox.agent_id = task.agent_id
        sandbox.rollout_id = task.rollout_id
        sandbox.state = SandboxState.RUNNING
        
        start_time = time.time()
        start_step = 0
        
        # Resume from snapshot if provided
        if task.resume_from_snapshot:
            self.state_engine.restore_snapshot(task.resume_from_snapshot, sandbox)
            start_step = task.resume_from_snapshot.step
            logger.info(f"Resumed {task.rollout_id} from step {start_step}")
        
        try:
            # Simulate agent execution steps
            for step in range(start_step, task.max_steps):
                if self._stop_event.is_set():
                    break
                
                # Simulate one agent step (thinking + acting)
                step_time = random.uniform(0.001, 0.005)
                time.sleep(step_time)
                sandbox.total_steps = step + 1
                
                # Periodic checkpointing
                if (step + 1) % self.checkpoint_interval == 0:
                    self.state_engine.capture_snapshot(sandbox)
                
                # Simulate random completion (agent finished task)
                if random.random() < 0.005:  # ~0.5% chance per step
                    break
                
                # Simulate random failure (sandbox crash)
                if random.random() < 0.001:  # ~0.1% chance per step
                    raise RuntimeError(f"Sandbox crash at step {step}")
            
            elapsed = time.time() - start_time
            
            # Final snapshot
            final_snapshot = self.state_engine.capture_snapshot(sandbox)
            
            result = RolloutResult(
                rollout_id=task.rollout_id,
                sandbox_id=sandbox.sandbox_id,
                agent_id=task.agent_id,
                success=True,
                total_steps=sandbox.total_steps,
                total_time_seconds=elapsed,
                final_state=final_snapshot,
                metrics={
                    "steps_per_second": sandbox.total_steps / elapsed if elapsed > 0 else 0,
                    "checkpoints_taken": sandbox.snapshot_count,
                },
            )
            
            with self._lock:
                self._total_completed += 1
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Try to capture failure state for debugging
            try:
                failure_snapshot = self.state_engine.capture_snapshot(sandbox)
            except Exception:
                failure_snapshot = None
            
            result = RolloutResult(
                rollout_id=task.rollout_id,
                sandbox_id=sandbox.sandbox_id,
                agent_id=task.agent_id,
                success=False,
                total_steps=sandbox.total_steps,
                total_time_seconds=elapsed,
                final_state=failure_snapshot,
                error=str(e),
                metrics={"failed_at_step": sandbox.total_steps},
            )
            
            with self._lock:
                self._total_failed += 1
            
            logger.warning(f"Rollout {task.rollout_id} failed at step "
                          f"{sandbox.total_steps}: {e}")
            
            return result
    
    def _spot_preemption_monitor(self):
        """
        Background thread monitoring spot instances for preemption.
        When preemption is imminent, migrates sandboxes to safe nodes.
        """
        while not self._stop_event.is_set():
            # Simulate fluctuating preemption probabilities
            for node in self.nodes:
                if node.node_type == NodeType.SPOT:
                    # Random walk on preemption probability
                    node.preemption_probability += random.gauss(0, 0.05)
                    node.preemption_probability = max(0, min(1, node.preemption_probability))
            
            # Check for at-risk nodes
            at_risk = self.spot_optimizer.check_preemption_risk(self.nodes)
            
            for node in at_risk:
                # Find safe target nodes
                safe_nodes = [n for n in self.nodes 
                             if n.node_type == NodeType.ON_DEMAND 
                             and n.is_healthy
                             and n.available_cpu >= 2]
                
                if not safe_nodes:
                    logger.warning(f"No safe nodes for migration from {node.node_id}")
                    continue
                
                # Migrate all active sandboxes from at-risk node
                for sandbox_id, sandbox in list(node.sandboxes.items()):
                    if sandbox.state == SandboxState.RUNNING:
                        target_node = random.choice(safe_nodes)
                        target_sandbox = self.pool_manager.provision_sandbox(target_node)
                        
                        if target_sandbox:
                            success = self.state_engine.migrate(sandbox, target_sandbox)
                            if success:
                                self._total_migrated += 1
                                # Update active rollout tracking
                                with self._lock:
                                    if sandbox.rollout_id in self._active_rollouts:
                                        task = self._active_rollouts[sandbox.rollout_id][1]
                                        self._active_rollouts[sandbox.rollout_id] = (
                                            target_sandbox, task
                                        )
                                
                                self.pool_manager.release_sandbox(sandbox, recycle=False)
            
            self._stop_event.wait(timeout=2.0)
    
    def run(self, tasks: List[RolloutTask]) -> List[RolloutResult]:
        """
        Execute a batch of agent rollout tasks.
        """
        print("=" * 70)
        print("EPHEMERAL SANDBOX ORCHESTRATOR")
        print("Distributed Agent Rollout Infrastructure")
        print("=" * 70)
        print(f"\nConfig:")
        print(f"  Compute nodes: {len(self.nodes)} "
              f"({sum(1 for n in self.nodes if n.node_type == NodeType.ON_DEMAND)} on-demand, "
              f"{sum(1 for n in self.nodes if n.node_type == NodeType.SPOT)} spot)")
        print(f"  Rollout tasks: {len(tasks)}")
        print(f"  Max concurrent: {self.max_concurrent}")
        print(f"  Checkpoint interval: every {self.checkpoint_interval} steps\n")
        
        self._total_submitted = len(tasks)
        start_time = time.time()
        
        # Phase 1: Pre-warm sandbox pool
        print("--- Phase 1: Pre-warming sandbox pool ---")
        self._pre_warm_pool()
        pool_metrics = self.pool_manager.get_metrics()
        print(f"  Warm pool: {pool_metrics['warm_pool_size']} sandboxes ready\n")
        
        # Phase 2: Start spot preemption monitor
        print("--- Phase 2: Starting preemption monitor ---")
        preemption_thread = threading.Thread(
            target=self._spot_preemption_monitor, daemon=True
        )
        preemption_thread.start()
        
        # Phase 3: Execute rollouts
        print("--- Phase 3: Executing rollouts ---")
        futures: Dict[str, Future] = {}
        
        for task in tasks:
            # Acquire sandbox (warm pool or cold provision)
            sandbox = self.pool_manager.acquire_sandbox()
            
            if sandbox is None:
                # Cold start — provision on best available node
                node = self.spot_optimizer.select_node(self.nodes, task)
                if node:
                    sandbox = self.pool_manager.provision_sandbox(node)
            
            if sandbox is None:
                logger.warning(f"No capacity for {task.rollout_id} — skipping")
                continue
            
            with self._lock:
                self._active_rollouts[task.rollout_id] = (sandbox, task)
            
            future = self._executor.submit(self._execute_rollout, sandbox, task)
            futures[task.rollout_id] = future
        
        # Collect results
        print(f"\n  Waiting for {len(futures)} rollouts to complete...")
        
        for rollout_id, future in futures.items():
            try:
                result = future.result(timeout=60.0)
                self._results.append(result)
                
                # Release sandbox back to pool
                sandbox = self._active_rollouts.get(rollout_id, (None, None))[0]
                if sandbox:
                    self.pool_manager.release_sandbox(sandbox, recycle=True)
                    # Process recycling queue
                    while self.pool_manager._recycling:
                        s = self.pool_manager._recycling.popleft()
                        self.pool_manager.recycle_sandbox(s)
                        
            except Exception as e:
                logger.error(f"Rollout {rollout_id} collection failed: {e}")
        
        # Stop background threads
        self._stop_event.set()
        
        elapsed = time.time() - start_time
        
        # Summary
        successful = sum(1 for r in self._results if r.success)
        failed = sum(1 for r in self._results if not r.success)
        total_steps = sum(r.total_steps for r in self._results)
        
        print(f"\n{'=' * 70}")
        print("ORCHESTRATION SUMMARY")
        print(f"{'=' * 70}")
        print(f"  Total time:              {elapsed:.1f}s")
        print(f"  Rollouts submitted:      {self._total_submitted}")
        print(f"  Rollouts completed:      {successful}")
        print(f"  Rollouts failed:         {failed}")
        print(f"  Success rate:            {successful/(successful+failed)*100:.1f}%"
              if (successful + failed) > 0 else "")
        print(f"  Total agent steps:       {total_steps:,}")
        print(f"  Throughput:              {total_steps/elapsed:,.0f} steps/sec")
        print(f"  Migrations (preemption): {self._total_migrated}")
        
        pool_metrics = self.pool_manager.get_metrics()
        print(f"\n  Pool Metrics:")
        print(f"    Warm hit rate:         {pool_metrics['warm_hit_rate']:.1%}")
        print(f"    Cold starts:           {pool_metrics['cold_starts']}")
        print(f"    Sandboxes recycled:    {pool_metrics['total_recycled']}")
        
        state_metrics = self.state_engine.get_metrics()
        print(f"\n  State Streaming:")
        print(f"    Snapshots captured:    {state_metrics['total_snapshots']}")
        print(f"    Avg snapshot time:     {state_metrics['avg_snapshot_time_ms']:.1f}ms")
        print(f"    Avg restore time:      {state_metrics['avg_restore_time_ms']:.1f}ms")
        print(f"    Migrations completed:  {state_metrics['total_migrations']}")
        
        # Cleanup
        self._executor.shutdown(wait=True)
        
        return self._results


# ============================================================
# PART 6: Demo
# ============================================================

if __name__ == "__main__":
    orchestrator = SandboxOrchestrator(
        num_on_demand_nodes=2,
        num_spot_nodes=5,
        warm_pool_target=30,
        checkpoint_interval_steps=50,
        max_concurrent_rollouts=100,
    )
    
    # Generate rollout tasks (simulating RL training batch)
    tasks = []
    for i in range(50):
        tasks.append(RolloutTask(
            rollout_id=f"rollout-{i:04d}",
            agent_id=f"devin-v3.{random.randint(1,5)}",
            task_description=f"Fix bug #{random.randint(1000, 9999)} in repository",
            max_steps=random.randint(100, 300),
            priority=random.randint(1, 10),
        ))
    
    results = orchestrator.run(tasks)
