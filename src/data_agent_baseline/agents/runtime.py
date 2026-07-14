from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from data_agent_baseline.benchmark.schema import AnswerTable


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_index: int
    thought: str
    action: str
    action_input: dict[str, Any]
    raw_response: str
    observation: dict[str, Any]
    ok: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentRuntimeState:
    steps: list[StepRecord] = field(default_factory=list)
    answer: AnswerTable | None = None
    failure_reason: str | None = None
    scratchpad: str = ""
    answer_rejections: int = 0
    # Rejections caused by the verifier demanding a phantom (non-existent) table/
    # column; tracked separately so they do not burn the real rejection budget.
    phantom_rejections: int = 0
    # Transient nudge injected into the next turn when the agent stalls (repeats a
    # failing action / loops). Cleared once consumed.
    pending_hint: str = ""
    # Auto-populated from successful inspect_video calls; never dropped by the
    # sliding-window summarizer so verified on-screen readings always persist.
    video_evidence: list[dict[str, Any]] = field(default_factory=list)
    # Cached once at task start for the answer-time source/column alignment check:
    # the knowledge-guide text (term -> authoritative table/column) and the list of
    # loose top-level CSV/JSON files that are likely pre-computed decoys.
    knowledge_text: str = ""
    decoy_files: list[str] = field(default_factory=list)
    # Provenance signatures of answers the verifier already rejected, so a
    # re-submission from the SAME (decoy) source is failed-closed instead of
    # slipping through once the rejection budget is spent.
    rejected_provenance: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class AgentRunResult:
    task_id: str
    answer: AnswerTable | None
    steps: list[StepRecord]
    failure_reason: str | None

    @property
    def succeeded(self) -> bool:
        return self.answer is not None and self.failure_reason is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer.to_dict() if self.answer is not None else None,
            "steps": [step.to_dict() for step in self.steps],
            "failure_reason": self.failure_reason,
            "succeeded": self.succeeded,
        }
