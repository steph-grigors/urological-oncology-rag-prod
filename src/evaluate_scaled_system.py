"""
Evaluation Script for Scaled RAG System
Evaluates quality across all 4 cancer types
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

from src.retrieval_optimized import OptimizedRAGRetriever, OptimizedRetrievalConfig
from src.evaluation import RAGEvaluator


def load_test_queries(filepath: str = "data/evaluation/test_queries_scaled.json") -> List[Dict]:
    """Load test queries"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data['queries']


def evaluate_system():
    """Run full system evaluation"""
    
    print("="*70)
    print(" "*15 + "🔬 SYSTEM EVALUATION")
    print("="*70)
    print()
    
    # Load retriever
    print("📚 Loading RAG system...")
    retriever = OptimizedRAGRetriever(
        chroma_db_dir="chroma_db_scaled",
        collection_name="urological_oncology_papers",
        config=OptimizedRetrievalConfig(
            top_k=5,
            max_context_length=3000,
            max_tokens=500
        )
    )
    
    print(f"   ✅ Loaded: {retriever.collection.count()} documents")
    print()
    
    # Load evaluator
    print("🔬 Loading evaluator...")
    evaluator = RAGEvaluator(rag_retriever=retriever)
    print("   ✅ Evaluator ready")
    print()
    
    # Load test queries
    print("📝 Loading test queries...")
    test_queries = load_test_queries()
    print(f"   ✅ Loaded {len(test_queries)} test queries")
    print()
    
    # Evaluate each query
    results = []
    
    print("🔍 Running evaluation...")
    print()
    
    for i, query_data in enumerate(test_queries, 1):
        question = query_data['question']
        topic = query_data['topic']
        
        print(f"Query {i}/{len(test_queries)}: {topic.upper()}")
        print(f"  Q: {question[:60]}...")
        
        try:
            start_time = time.time()
            
            # Get answer
            response = retriever.query(
                question=question,
                return_sources=True,
                use_cache=False
            )
            
            latency = time.time() - start_time
            
            # Prepare context
            context = "\n\n".join([
                f"[Doc {i+1}]\n{source['text_preview'][:500]}"
                for i, source in enumerate(response['sources'])
            ])
            
            # Evaluate
            faithfulness = evaluator.evaluate_faithfulness(
                question, response['answer'], context
            )
            
            relevance = evaluator.evaluate_relevance(
                question, response['answer']
            )
            
            chunks = [{'text': s['text_preview']} for s in response['sources']]
            precision = evaluator.evaluate_context_precision(question, chunks)
            
            result = {
                'query_id': i,
                'question': question,
                'topic': topic,
                'answer': response['answer'],
                'num_sources': response['num_sources'],
                'latency': latency,
                'faithfulness': faithfulness,
                'relevance': relevance,
                'context_precision': precision,
                'overall_quality': (faithfulness + relevance + precision) / 3,
                'sources': [
                    {
                        'title': s['title'],
                        'topic': s.get('topic', 'N/A'),
                        'section': s['section'],
                        'pmid': s['pmid']
                    }
                    for s in response['sources']
                ]
            }
            
            results.append(result)
            
            print(f"  ✅ Faithfulness: {faithfulness:.1%} | Relevance: {relevance:.1%} | Precision: {precision:.1%}")
            print(f"     Latency: {latency:.2f}s | Quality: {result['overall_quality']:.1%}")
            print()
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print()
            continue
    
    # Calculate aggregate metrics
    print("="*70)
    print("📊 AGGREGATE METRICS")
    print("="*70)
    print()
    
    if not results:
        print("❌ No successful evaluations")
        return
    
    # Overall metrics
    avg_faithfulness = sum(r['faithfulness'] for r in results) / len(results)
    avg_relevance = sum(r['relevance'] for r in results) / len(results)
    avg_precision = sum(r['context_precision'] for r in results) / len(results)
    avg_quality = sum(r['overall_quality'] for r in results) / len(results)
    avg_latency = sum(r['latency'] for r in results) / len(results)
    
    print(f"Overall Quality:      {avg_quality:.1%}")
    print(f"  Faithfulness:       {avg_faithfulness:.1%}")
    print(f"  Relevance:          {avg_relevance:.1%}")
    print(f"  Context Precision:  {avg_precision:.1%}")
    print()
    print(f"Performance:")
    print(f"  Avg Latency:        {avg_latency:.2f}s")
    print(f"  Min Latency:        {min(r['latency'] for r in results):.2f}s")
    print(f"  Max Latency:        {max(r['latency'] for r in results):.2f}s")
    print()
    
    # Per-topic breakdown
    print("Per-Topic Breakdown:")
    for topic in ['prostate', 'bladder', 'kidney', 'testicular']:
        topic_results = [r for r in results if r['topic'] == topic]
        if topic_results:
            topic_quality = sum(r['overall_quality'] for r in topic_results) / len(topic_results)
            print(f"  {topic.capitalize():12} {len(topic_results)} queries | Quality: {topic_quality:.1%}")
    
    print()
    print("="*70)
    
    # Save results
    output_dir = Path("data/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detailed results
    detailed_file = output_dir / "scaled_system_detailed.json"
    with open(detailed_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Aggregate metrics
    aggregate = {
        'evaluation_date': datetime.now().isoformat(),
        'total_queries': len(results),
        'avg_faithfulness': avg_faithfulness,
        'avg_relevance': avg_relevance,
        'avg_context_precision': avg_precision,
        'avg_quality': avg_quality,
        'avg_latency': avg_latency,
        'min_latency': min(r['latency'] for r in results),
        'max_latency': max(r['latency'] for r in results),
        'per_topic': {}
    }
    
    for topic in ['prostate', 'bladder', 'kidney', 'testicular']:
        topic_results = [r for r in results if r['topic'] == topic]
        if topic_results:
            aggregate['per_topic'][topic] = {
                'queries': len(topic_results),
                'avg_quality': sum(r['overall_quality'] for r in topic_results) / len(topic_results),
                'avg_latency': sum(r['latency'] for r in topic_results) / len(topic_results)
            }
    
    aggregate_file = output_dir / "scaled_system_metrics.json"
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"💾 Results saved:")
    print(f"   Detailed: {detailed_file}")
    print(f"   Aggregate: {aggregate_file}")
    print()
    
    return aggregate


if __name__ == "__main__":
    evaluate_system()
