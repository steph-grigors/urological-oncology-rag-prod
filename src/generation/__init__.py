"""
Generation package.

Converts a `RetrievalResult` into a cited, confidence-gated clinical answer.

Flow:
    retrieval_result
        → confidence.gate()          check if evidence meets threshold
        → prompts.build_prompt()     assemble system + user messages
        → generator.generate()       LLM call (Anthropic or OpenAI)
        → generator.check_citations() verify [Doc N] references are grounded
        → GenerationResult           typed response with answer + sources
"""
