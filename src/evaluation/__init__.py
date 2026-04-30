"""
Evaluation package.

Provides automated quality measurement of the RAG pipeline against a
curated golden query set.  All evaluation logic is offline (not in the
hot path of live queries).

Entry point: `runner.run_evaluation(golden_set_path, output_dir)`.
"""
