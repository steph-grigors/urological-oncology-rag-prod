"""
Optimized Retrieval Module
High-performance RAG with caching and streaming
"""

import os
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import time
from functools import lru_cache

import chromadb
from chromadb.config import Settings
from openai import OpenAI


@dataclass
class OptimizedRetrievalConfig:
    """Configuration for optimized retrieval"""
    top_k: int = 5
    max_context_length: int = 3000  # Limit context size (was unlimited)
    max_tokens: int = 500  # Limit response length (was 1000)
    temperature: float = 0.2  # Lower = faster + more deterministic
    use_cache: bool = True
    cache_dir: str = "data/cache"


class QueryCache:
    """Simple disk-based query cache"""

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, query: str, top_k: int, model: str) -> str:
        """Generate cache key from query parameters"""
        content = f"{query}|{top_k}|{model}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, query: str, top_k: int, model: str) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        cache_key = self._get_cache_key(query, top_k, model)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    # Check if cache is less than 24 hours old
                    if time.time() - cached.get('timestamp', 0) < 86400:
                        return cached.get('response')
            except:
                pass
        return None

    def set(self, query: str, top_k: int, model: str, response: Dict[str, Any]):
        """Cache response"""
        cache_key = self._get_cache_key(query, top_k, model)
        cache_file = self.cache_dir / f"{cache_key}.json"

        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'response': response
                }, f)
        except:
            pass  # Fail silently if cache write fails

    def clear(self):
        """Clear all cache"""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()


class OptimizedRAGRetriever:
    """High-performance RAG retriever with caching and optimization"""

    def __init__(
        self,
        chroma_db_dir: str = "chroma_db_scaled",  # ← New DB
        collection_name: str = "urological_oncology_papers",  # ← New collection
        config: OptimizedRetrievalConfig = None
    ):
        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name
        self.config = config or OptimizedRetrievalConfig()

        # Initialize cache
        self.cache = QueryCache(self.config.cache_dir) if self.config.use_cache else None

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        self.openai_client = OpenAI(api_key=api_key)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=False
            )
        )

        # Get collection
        self.collection = self.chroma_client.get_collection(
            name=self.collection_name
        )

        print(f"✅ Optimized RAG Retriever initialized")
        print(f"   Collection: {self.collection_name}")
        print(f"   Documents: {self.collection.count()}")
        print(f"   Cache enabled: {self.config.use_cache}")
        print(f"   Max context: {self.config.max_context_length} chars")
        print(f"   Max tokens: {self.config.max_tokens}")

    @lru_cache(maxsize=1000)
    def generate_query_embedding(self, query: str) -> tuple:
        """
        Generate embedding for query (cached in memory)
        Returns tuple so it's hashable for lru_cache
        """
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
            encoding_format="float"
        )
        return tuple(response.data[0].embedding)

    def retrieve_chunks(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve most relevant chunks (same as before but with cached embeddings)"""
        k = top_k or self.config.top_k

        # Generate query embedding (cached)
        query_embedding = list(self.generate_query_embedding(query))

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        chunks = []
        for i in range(len(results['ids'][0])):
            chunk = {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],  # ← This contains all metadata
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]
            }
            chunks.append(chunk)

        return chunks

    def format_context_optimized(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Format context with length limit for faster LLM processing
        """
        context_parts = []
        total_length = 0

        for i, chunk in enumerate(chunks, 1):
            metadata = chunk['metadata']

            # Shorter format - remove verbose fields
            context = f"""[Doc {i}] {metadata.get('title', 'N/A')[:80]}
Section: {metadata.get('section_name', 'N/A')}
PMID: {metadata.get('pmid', 'N/A')}

{chunk['text']}
"""

            # Check length limit
            if total_length + len(context) > self.config.max_context_length:
                # Truncate this chunk to fit
                remaining = self.config.max_context_length - total_length
                if remaining > 200:  # Only include if meaningful
                    context = context[:remaining] + "..."
                    context_parts.append(context.strip())
                break

            context_parts.append(context.strip())
            total_length += len(context)

        return "\n\n".join(context_parts)

    def generate_answer_optimized(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        model: str = "gpt-4o-mini",
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate answer with optimizations:
        - Shorter context
        - Fewer max_tokens
        - Lower temperature
        - Optional streaming
        """
        # Format context (optimized)
        context = self.format_context_optimized(chunks)

        # Shorter, more focused system prompt
        system_prompt = """You are a medical research assistant specializing in prostate cancer.

Provide concise, evidence-based answers citing sources by [Doc N] format.
If information is insufficient, state this clearly."""

        user_prompt = f"""Context:
{context}

Question: {query}

Provide a focused answer with citations."""

        # Generate answer
        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=stream
        )

        if stream:
            return response  # Return stream object

        answer = response.choices[0].message.content

        return {
            'answer': answer,
            'sources': chunks,
            'model': model,
            'query': query,
            'context_length': len(context)
        }

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
        model: str = "gpt-4o-mini",
        return_sources: bool = True,
        use_cache: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Optimized RAG pipeline with caching
        """
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        k = top_k or self.config.top_k

        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(question, k, model)
            if cached:
                print("✨ Cache hit!")
                return cached

        start_time = time.time()

        # Retrieve relevant chunks
        chunks = self.retrieve_chunks(
            query=question,
            top_k=k,
            filter_metadata=filter_metadata
        )

        retrieval_time = time.time() - start_time

        # Generate answer
        generation_start = time.time()
        result = self.generate_answer_optimized(
            query=question,
            chunks=chunks,
            model=model
        )
        generation_time = time.time() - generation_start

        total_time = time.time() - start_time

        # Format response
        response = {
            'question': question,
            'answer': result['answer'],
            'model': model,
            'num_sources': len(chunks),
            'latency': {
                'total': total_time,
                'retrieval': retrieval_time,
                'generation': generation_time
            },
            'context_length': result.get('context_length', 0)
        }

        if return_sources:
            response['sources'] = [
                {
                    'title': chunk['metadata'].get('title'),
                    'section': chunk['metadata'].get('section_name'),
                    'pmid': chunk['metadata'].get('pmid'),
                    'doi': chunk['metadata'].get('doi'),
                    'topic': chunk['metadata'].get('topic'),  # ← ADD THIS
                    'similarity': chunk['similarity'],
                    'text_preview': chunk['text'][:500] + "..."  # Show more text
                }
                for chunk in chunks
    ]

        # Cache response
        if use_cache and self.cache:
            self.cache.set(question, k, model, response)

        return response

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        model: str = "gpt-4o-mini"
    ) -> Iterator[str]:
        """
        Stream response for better UX
        Yields answer chunks as they're generated
        """
        # Retrieve chunks
        chunks = self.retrieve_chunks(
            query=question,
            top_k=top_k or self.config.top_k
        )

        # Get streaming response
        context = self.format_context_optimized(chunks)

        system_prompt = """You are a medical research assistant specializing in prostate cancer.

Provide concise, evidence-based answers citing sources by [Doc N] format."""

        user_prompt = f"""Context:
{context}

Question: {question}

Provide a focused answer with citations."""

        stream = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content



