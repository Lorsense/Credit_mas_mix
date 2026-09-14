"""GPU accounting for a shared Agent pool and an independent value scorer.

These helpers deliberately have no Ray/Torch dependency so deployment layouts
can be validated before any distributed workers reserve resources.
"""
from __future__ import annotations

from collections.abc import Mapping
import numbers


def agent_gpu_layout(trainer: Mapping) -> list[int]:
    """Return actual Agent ranks per node (physical GPUs may include a scorer)."""
    explicit = trainer.get("agent_gpus_per_node")
    layout = list(explicit) if explicit is not None else [trainer["n_gpus_per_node"]] * int(trainer["nnodes"])
    if not layout or any(isinstance(n, bool) or not isinstance(n, numbers.Integral) or n <= 0 for n in layout):
        raise ValueError("trainer.agent_gpus_per_node must be a nonempty list of positive integers")
    if len(layout) != int(trainer["nnodes"]):
        raise ValueError("agent_gpus_per_node must contain one count per trainer.nnodes")
    if any(n > int(trainer["n_gpus_per_node"]) for n in layout):
        raise ValueError("An Agent node cannot request more GPUs than n_gpus_per_node")
    return [int(n) for n in layout]


def agent_world_size(trainer: Mapping) -> int:
    return sum(agent_gpu_layout(trainer))


def validate_pool_layout(node_resources: Mapping, pool_spec: Mapping, reserved_node: str | None = None,
                         *, cpus_per_gpu: float = 0.0, reserved_cpus: float = 0.0) -> None:
    """Check exact heterogeneous placement, optionally reserving one value GPU.

    Distinct node bundles belonging to one Agent pool must use distinct nodes.
    Checking only the total or repeating the first bundle size is insufficient
    for an 8+8 physical layout with a 7+8 Agent pool.
    """
    remaining = {str(node): float(info.get("GPU", info.get("NPU", 0))) for node, info in node_resources.items()}
    cpu_remaining = {str(node): float(info.get("CPU", 0)) for node, info in node_resources.items()}
    if reserved_node is not None:
        reserved_node = str(reserved_node)
        if remaining.get(reserved_node, 0) < 1:
            raise ValueError("The trainer node needs one available GPU for the value scorer")
        remaining[reserved_node] -= 1
        if cpu_remaining.get(reserved_node, 0) < reserved_cpus:
            raise ValueError("The value/coordinator node has insufficient available CPUs")
        cpu_remaining[reserved_node] -= reserved_cpus
    requests = []
    for name, layout in pool_spec.items():
        for count in layout:
            if isinstance(count, bool) or not isinstance(count, numbers.Integral) or count <= 0:
                raise ValueError("Pool GPU requests must be positive integers")
            requests.append((int(count), name))
    requests.sort(key=lambda item: item[0], reverse=True)
    used = {name: set() for name in pool_spec}

    def place(index):
        if index == len(requests):
            return True
        count, name = requests[index]
        candidates = sorted((node for node in remaining if node not in used[name] and remaining[node] >= count
                             and cpu_remaining[node] >= count * cpus_per_gpu),
                            key=lambda node: remaining[node])
        for node in candidates:
            remaining[node] -= count
            cpu_remaining[node] -= count * cpus_per_gpu
            used[name].add(node)
            if place(index + 1):
                return True
            used[name].remove(node)
            remaining[node] += count
            cpu_remaining[node] += count * cpus_per_gpu
        return False

    if not place(0):
        raise ValueError(f"Agent pool layout {dict(pool_spec)} does not fit available per-node GPU/CPU resources"
                         + (" after reserving the value GPU" if reserved_node is not None else ""))
