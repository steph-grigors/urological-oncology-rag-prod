"""
Evaluation Module
Measures RAG system quality using various metrics
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time

from openai import OpenAI
from tqdm import tqdm
import os


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    faithfulness: float  # Is answer grounded in retrieved context?
    relevance: float     # Does answer address the question?
    context_precision: float  # Are retrieved chunks relevant?
    context_recall: float     # Did we retrieve all relevant info?
    latency: float       # Response time in seconds


class RAGEvaluator:
    """Evaluates RAG system quality"""

    def __init__(self, rag_retriever=None):
        self.rag_retriever = rag_retriever

        # Initialize OpenAI for LLM-as-judge
        api_key = os.getenv("OPENAI_API_KEY")
        self.openai_client = OpenAI(api_key=api_key)

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str
    ) -> float:
        """
        Evaluate if answer is faithful to the context (no hallucination)

        Returns score between 0 and 1
        """
        prompt = f"""You are evaluating the faithfulness of an AI-generated answer.

Question: {question}

Context (from research papers):
{context}

Generated Answer:
{answer}

Evaluation Task:
Rate the faithfulness of the answer on a scale of 0 to 1, where:
- 1.0 = All claims in the answer are directly supported by the context
- 0.5 = Some claims are supported, others are inferred or unsupported
- 0.0 = Answer contains claims not found in or contradicted by the context

Consider:
1. Are all factual claims in the answer present in the context?
2. Does the answer add information not in the context?
3. Does the answer misrepresent information from the context?

Respond with ONLY a number between 0 and 1."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))  # Clamp between 0 and 1

        except Exception as e:
            print(f"Error evaluating faithfulness: {e}")
            return 0.5  # Default middle score

    def evaluate_relevance(
        self,
        question: str,
        answer: str
    ) -> float:
        """
        Evaluate if answer is relevant to the question

        Returns score between 0 and 1
        """
        prompt = f"""You are evaluating the relevance of an AI-generated answer.

Question: {question}

Generated Answer:
{answer}

Evaluation Task:
Rate how well the answer addresses the question on a scale of 0 to 1, where:
- 1.0 = Answer directly and completely addresses the question
- 0.5 = Answer partially addresses the question or includes tangential information
- 0.0 = Answer does not address the question at all

Consider:
1. Does the answer directly respond to what was asked?
2. Is the answer complete?
3. Does the answer stay on topic?

Respond with ONLY a number between 0 and 1."""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10
            )

            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"Error evaluating relevance: {e}")
            return 0.5

    def evaluate_context_precision(
        self,
        question: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> float:
        """
        Evaluate if retrieved chunks are relevant to the question
        Precision = relevant_chunks / total_retrieved_chunks
        """
        if not retrieved_chunks:
            return 0.0

        relevant_count = 0

        for chunk in retrieved_chunks[:5]:  # Check top 5
            context_text = chunk['text'][:500]  # First 500 chars

            prompt = f"""Is this context relevant to answering the question?

Question: {question}

Context:
{context_text}

Answer with ONLY 'yes' or 'no'."""

            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=5
                )

                answer = response.choices[0].message.content.strip().lower()
                if 'yes' in answer:
                    relevant_count += 1

            except:
                continue

        return relevant_count / min(5, len(retrieved_chunks))

    def evaluate_query(
    self,
    question: str,
    answer: str,
    sources: List[Dict[str, Any]],
    latency: float
    ) -> EvaluationMetrics:
        """
        Evaluate a complete RAG response

        Args:
            question: User's question
            answer: Generated answer
            sources: Retrieved source chunks
            latency: Response time in seconds

        Returns:
            EvaluationMetrics object
        """
        # Prepare context string - handle both 'text' and 'text_preview' keys
        context_parts = []
        for i, source in enumerate(sources[:5]):
            # Try text_preview first (from formatted response), then text (from raw chunks)
            text = source.get('text_preview', source.get('text', ''))[:500]
            context_parts.append(f"[Source {i+1}]\n{text}")

        context = "\n\n".join(context_parts)

        # Evaluate faithfulness
        print(f"  Evaluating faithfulness...")
        faithfulness = self.evaluate_faithfulness(question, answer, context)

        # Evaluate relevance
        print(f"  Evaluating relevance...")
        relevance = self.evaluate_relevance(question, answer)

        # Convert sources to format expected by context_precision
        chunks_for_precision = []
        for source in sources:
            chunk = {
                'text': source.get('text_preview', source.get('text', ''))
            }
            chunks_for_precision.append(chunk)

        # Evaluate context precision
        print(f"  Evaluating context precision...")
        context_precision = self.evaluate_context_precision(question, chunks_for_precision)

        # Context recall is harder without ground truth, set to 0.0 for now
        context_recall = 0.0

        print(f"  ✅ Metrics: F={faithfulness:.2f}, R={relevance:.2f}, CP={context_precision:.2f}")

        return EvaluationMetrics(
            faithfulness=faithfulness,
            relevance=relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            latency=latency
        )

    def evaluate_test_set(
        self,
        test_queries: List[str],
        save_results: bool = True,
        output_dir: str = "data/evaluation"
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on a test set of queries

        Args:
            test_queries: List of test questions
            save_results: Whether to save detailed results
            output_dir: Directory to save results

        Returns:
            Dictionary with aggregated metrics
        """
        if not self.rag_retriever:
            raise ValueError("RAG retriever not provided")

        print("="*70)
        print(" "*20 + "📊 RAG EVALUATION")
        print("="*70)
        print(f"\nEvaluating {len(test_queries)} queries...")
        print()

        results = []

        for question in tqdm(test_queries, desc="Evaluating queries"):
            # Time the query
            start_time = time.time()

            try:
                # Get RAG response
                response = self.rag_retriever.query(question)

                latency = time.time() - start_time

                # Evaluate
                metrics = self.evaluate_query(
                    question=question,
                    answer=response['answer'],
                    sources=response.get('sources', []),
                    latency=latency
                )

                results.append({
                    'question': question,
                    'answer': response['answer'],
                    'metrics': asdict(metrics),
                    'num_sources': response['num_sources']
                })

            except Exception as e:
                print(f"\n⚠️  Error evaluating '{question[:50]}...': {e}")
                continue

        # Calculate aggregate statistics
        aggregate_metrics = self._calculate_aggregate_metrics(results)

        # Print results
        self._print_evaluation_results(aggregate_metrics)

        # Save if requested
        if save_results:
            self._save_results(results, aggregate_metrics, output_dir)

        return aggregate_metrics

    def _calculate_aggregate_metrics(
        self,
        results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate average metrics across all queries"""
        if not results:
            return {}

        metrics_keys = ['faithfulness', 'relevance', 'context_precision', 'latency']

        aggregates = {}
        for key in metrics_keys:
            values = [r['metrics'][key] for r in results]
            aggregates[f'avg_{key}'] = sum(values) / len(values)
            aggregates[f'min_{key}'] = min(values)
            aggregates[f'max_{key}'] = max(values)

        aggregates['total_queries'] = len(results)

        return aggregates

    def _print_evaluation_results(self, metrics: Dict[str, Any]):
        """Pretty print evaluation results"""
        print("\n" + "="*70)
        print("📊 EVALUATION RESULTS")
        print("="*70)
        print()

        print(f"Total Queries Evaluated: {metrics.get('total_queries', 0)}")
        print()

        print("Average Metrics:")
        print(f"  Faithfulness:       {metrics.get('avg_faithfulness', 0):.3f} "
              f"(min: {metrics.get('min_faithfulness', 0):.3f}, "
              f"max: {metrics.get('max_faithfulness', 0):.3f})")
        print(f"  Relevance:          {metrics.get('avg_relevance', 0):.3f} "
              f"(min: {metrics.get('min_relevance', 0):.3f}, "
              f"max: {metrics.get('max_relevance', 0):.3f})")
        print(f"  Context Precision:  {metrics.get('avg_context_precision', 0):.3f} "
              f"(min: {metrics.get('min_context_precision', 0):.3f}, "
              f"max: {metrics.get('max_context_precision', 0):.3f})")
        print(f"  Latency (seconds):  {metrics.get('avg_latency', 0):.2f} "
              f"(min: {metrics.get('min_latency', 0):.2f}, "
              f"max: {metrics.get('max_latency', 0):.2f})")
        print()

        # Overall grade
        avg_score = (
            metrics.get('avg_faithfulness', 0) +
            metrics.get('avg_relevance', 0) +
            metrics.get('avg_context_precision', 0)
        ) / 3

        if avg_score >= 0.8:
            grade = "A (Excellent)"
            emoji = "🌟"
        elif avg_score >= 0.7:
            grade = "B (Good)"
            emoji = "✅"
        elif avg_score >= 0.6:
            grade = "C (Acceptable)"
            emoji = "👍"
        else:
            grade = "D (Needs Improvement)"
            emoji = "⚠️"

        print(f"{emoji} Overall Quality Score: {avg_score:.3f} - Grade: {grade}")
        print()
        print("="*70)

    def _save_results(
        self,
        results: List[Dict[str, Any]],
        aggregates: Dict[str, Any],
        output_dir: str
    ):
        """Save evaluation results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed results
        detailed_file = output_path / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Save aggregate metrics
        aggregate_file = output_path / "aggregate_metrics.json"
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregates, f, indent=2)

        print(f"💾 Results saved to: {output_dir}/")

