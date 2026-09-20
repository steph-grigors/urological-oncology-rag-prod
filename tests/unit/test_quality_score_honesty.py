"""
Tests that the quality scores are described as what they are.

Every /query response carries three numbers, and both user interfaces render
them as percentages under headings like "Faithfulness — Is the answer grounded
in sources?". None of them reads the medical content of the answer:

  faithfulness        the share of [Doc N] tags pointing at a retrieved chunk.
                      ClinicalGenerator strips out-of-range tags before the
                      judge runs, so this is 1.0 whenever the answer cites
                      anything and 0.85 when it cites nothing -- a constant,
                      not a measurement.
  answer_relevance    the share of the question's content words appearing in
                      the answer. Restating the question scores full marks.
  context_precision   the share of retrieved chunks containing any question
                      word.

These tests pin the arithmetic so the claims in the API schema and the UI
cannot drift away from it again.
"""

from __future__ import annotations

from src.api.routes.query import QueryQualityScores
from src.evaluation.judges import (
    _heuristic_answer_relevance,
    _heuristic_context_precision,
    _heuristic_faithfulness,
)


class _Chunk:
    def __init__(self, text: str):
        self.text = text
        self.metadata: dict = {}


class TestScoresMeasureWhatTheSchemaSays:
    def test_faithfulness_ignores_whether_the_chunk_supports_the_claim(self):
        """A flatly contradicted claim with a valid citation index scores 1.0."""
        chunks = [_Chunk("Enzalutamide showed no overall survival benefit.")]
        score, _ = _heuristic_faithfulness(
            "Enzalutamide doubles overall survival [Doc 1].", chunks
        )
        assert score == 1.0

    def test_faithfulness_is_a_constant_for_any_cited_answer(self):
        chunks = [_Chunk("anything"), _Chunk("anything")]
        for answer in ("Claim [Doc 1].", "Other [Doc 2].", "Both [Doc 1][Doc 2]."):
            assert _heuristic_faithfulness(answer, chunks)[0] == 1.0

    def test_faithfulness_of_an_uncited_answer_is_0_85(self):
        assert _heuristic_faithfulness("No citations at all.", [_Chunk("x")])[0] == 0.85

    def test_answer_relevance_rewards_restating_the_question(self):
        question = "What is the first-line treatment for metastatic bladder cancer?"
        parroted = "The first-line treatment for metastatic bladder cancer is unclear."
        assert _heuristic_answer_relevance(question, parroted)[0] == 1.0

    def test_answer_relevance_ignores_correctness(self):
        question = "Which drug treats prostate cancer?"
        wrong = "The drug that treats prostate cancer is paracetamol."
        assert _heuristic_answer_relevance(question, wrong)[0] == 1.0

    def test_context_precision_only_checks_word_presence(self):
        question = "enzalutamide survival"
        off_topic = [_Chunk("A study of enzalutamide in laboratory glassware survival.")]
        assert _heuristic_context_precision(question, off_topic)[0] == 1.0


class TestSchemaDeclaresItsMethod:
    def test_method_field_marks_the_scores_as_lexical(self):
        scores = QueryQualityScores(
            faithfulness=1.0, answer_relevance=0.5, context_precision=0.5
        )
        assert scores.method == "lexical_heuristic"

    def test_descriptions_do_not_claim_clinical_accuracy(self):
        props = QueryQualityScores.model_json_schema()["properties"]
        for name in ("faithfulness", "answer_relevance", "context_precision"):
            desc = props[name]["description"].lower()
            assert desc, f"{name} has no description"
            assert "accurate" not in desc and "correct" not in desc.replace("incorrect", ""), (
                f"{name} description claims correctness: {desc}"
            )

    def test_faithfulness_description_states_it_does_not_verify_support(self):
        desc = QueryQualityScores.model_json_schema()["properties"]["faithfulness"]["description"]
        assert "does NOT check that the cited chunk supports the claim" in desc

    def test_relevance_description_admits_restating_scores_well(self):
        desc = QueryQualityScores.model_json_schema()["properties"]["answer_relevance"]["description"]
        assert "restating" in desc.lower()
