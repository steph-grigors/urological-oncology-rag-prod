"""
LLM-as-judge evaluation metrics.

Each judge is a standalone function that makes a single, focused LLM call
and returns a scalar score in [0, 1].  Judges are intentionally separate
so they can be run in parallel and individually replaced or ablated.

Judges (to be implemented):
    faithfulness(question, answer, context) -> float
        Score ∈ {0, 0.5, 1.0} — is every claim in `answer` grounded in
        `context`?  Uses chain-of-thought before the final score to improve
        reliability.  Model: gpt-4o-mini (cheap, adequate for binary grounding).

    answer_relevance(question, answer) -> float
        Does the answer actually address the question asked?

    context_precision(question, chunks) -> float
        For each retrieved chunk, is it relevant to the question?
        precision = relevant_chunks / total_chunks.

    context_recall(question, answer, ground_truth_answer) -> float
        Does the answer cover all the key facts in the ground truth?
        Requires `golden_set.GoldenQuery.ground_truth` to be populated.
        Returns 0.0 if ground truth is absent (do not penalise).

    clinical_safety(answer) -> float
        Does the answer contain any direct clinical advice that bypasses
        the intended scope (prescriptions, specific dosing without caveats)?
        Score 1.0 = safe, 0.0 = unsafe.  Uses a stricter model for this judge.

Public API (to be implemented):
    class JudgeSet:
        def __init__(self, openai_client, settings: Settings): ...

        def score_all(
            self,
            question: str,
            answer: str,
            chunks: list[SearchResult],
            ground_truth: str | None = None,
        ) -> JudgeScores: ...

    JudgeScores(dataclass)
        faithfulness: float
        answer_relevance: float
        context_precision: float
        context_recall: float
        clinical_safety: float
        overall: float   # unweighted mean of the above
"""
