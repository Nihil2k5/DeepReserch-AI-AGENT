"""Node interface.

A node is a unit of work: given context, return a JSON-serialisable dict that
becomes its checkpoint. Nodes must be deterministic in the sense that matters
for resumption — the engine may replay a saved output instead of re-running, so
anything a node returns must be sufficient to continue without the node.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..context import NodeContext


class Node(ABC):
    #: Human-readable name, also used as the span name.
    name: str = "node"

    #: Checkpoint key this node writes. Set per instance by the engine.
    key: str = ""

    @abstractmethod
    def run(self, ctx: NodeContext) -> dict: ...
