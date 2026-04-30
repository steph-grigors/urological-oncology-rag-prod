"""
Scaled Embeddings Generation
Generate embeddings for 41,970 chunks across 4 cancer types
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime

import chromadb
from chromadb.config import Settings
from openai import OpenAI


class ScaledEmbeddingsGenerator:
    """Generate embeddings and store in ChromaDB for multiple topics"""

    def __init__(
        self,
        input_dir: str = "data/processed_fulltext",
        chroma_db_dir: str = "chroma_db_scaled",
        collection_name: str = "urological_oncology_papers"
    ):
        self.input_dir = Path(input_dir)
        self.chroma_db_dir = chroma_db_dir
        self.collection_name = collection_name

        self.topics = ["prostate", "bladder", "kidney", "testicular"]

        # Initialize OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        self.openai_client = OpenAI(api_key=api_key)

        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_db_dir),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Create or get collection
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            print(f"🗑️  Deleted existing collection: {self.collection_name}")
        except:
            pass

        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"description": "Urological oncology papers - 4 cancer types, 815 papers"}
        )

        print(f"✅ Created new collection: {self.collection_name}")

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                encoding_format="float"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"⚠️  Embedding error: {e}")
            return None

    def generate_embeddings_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings in batches"""
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1

            if batch_num % 10 == 0 or batch_num == 1:
                print(f"   Batch {batch_num}/{total_batches} ({len(batch)} chunks)...", end=" ")

            try:
                response = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                    encoding_format="float"
                )

                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)

                if batch_num % 10 == 0 or batch_num == 1:
                    print("✅")

                # Rate limiting
                time.sleep(0.05)

            except Exception as e:
                print(f"❌ Error: {e}")
                # Generate individually as fallback
                for text in batch:
                    emb = self.generate_embedding(text)
                    if emb:
                        embeddings.append(emb)
                    else:
                        embeddings.append([0.0] * 1536)  # Placeholder
                    time.sleep(0.1)

        return embeddings

    def process_topic(self, topic: str) -> int:
        """Process chunks for a topic"""
        print(f"\n{'='*70}")
        print(f"🔢 Generating embeddings for: {topic.upper()}")
        print(f"{'='*70}")

        # Load chunks
        chunks_file = self.input_dir / f"{topic}_chunks.json"

        if not chunks_file.exists():
            print(f"❌ No chunks found for {topic}")
            return 0

        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"📄 Loaded {len(chunks)} chunks")

        # Extract texts
        texts = [chunk['text'] for chunk in chunks]

        # Generate embeddings
        print(f"🔢 Generating embeddings...")
        start_time = time.time()
        embeddings = self.generate_embeddings_batch(texts)
        elapsed = time.time() - start_time

        print(f"   ✅ Generated {len(embeddings)} embeddings in {elapsed:.1f}s")

        if len(embeddings) != len(chunks):
            print(f"⚠️  Warning: {len(embeddings)} embeddings vs {len(chunks)} chunks")

        # Prepare for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Create truly unique ID using global counter
            pmc_id = chunk['metadata'].get('pmc_id', 'unknown')
            pmid = chunk['metadata'].get('pmid', 'unknown')
            section_name = chunk.get('section_name', 'unknown').replace(' ', '_')[:30]
            chunk_idx = chunk.get('chunk_index', i)
            # Use global index i to ensure uniqueness
            chunk_id = f"{topic}_{pmc_id}_{pmid}_{section_name}_{chunk_idx}_{i}"

            # Prepare metadata
            metadata = {
                'pmc_id': chunk['metadata'].get('pmc_id', ''),
                'pmid': chunk['metadata'].get('pmid', ''),
                'title': chunk['metadata'].get('title', '')[:200],
                'topic': topic,
                'section_name': chunk.get('section_name', 'Unknown'),
                'chunk_index': chunk.get('chunk_index', 0),
                'total_chunks': chunk.get('total_chunks', 1),
                'journal': chunk['metadata'].get('journal', '')[:100],
                'year': chunk['metadata'].get('year', ''),
                'doi': chunk['metadata'].get('doi', '')[:100]
            }

            ids.append(chunk_id)
            documents.append(chunk['text'])
            metadatas.append(metadata)
            embeddings_list.append(embedding)

        # Add to ChromaDB in batches
        print(f"💾 Storing in ChromaDB...")

        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end_idx = min(i + batch_size, len(ids))

            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                embeddings=embeddings_list[i:end_idx]
            )

            if (i + batch_size) % 5000 == 0 or end_idx == len(ids):
                print(f"   Stored {end_idx}/{len(ids)} chunks")

        print(f"✅ Completed {topic}: {len(chunks)} chunks")

        return len(chunks)

    def process_all(self):
        """Process all topics"""
        start_time = time.time()

        print("="*70)
        print(" "*12 + "🔢 SCALED EMBEDDINGS GENERATION")
        print("="*70)
        print(f"\nModel: text-embedding-3-small")
        print(f"Topics: {', '.join(self.topics)}")
        print()

        total_chunks = 0
        results = {}

        for topic in self.topics:
            chunks_count = self.process_topic(topic)
            results[topic] = chunks_count
            total_chunks += chunks_count

        elapsed = time.time() - start_time

        # Verify collection
        collection_count = self.collection.count()

        # Summary
        print("\n" + "="*70)
        print("📊 EMBEDDINGS SUMMARY")
        print("="*70)

        for topic, count in results.items():
            print(f"✅ {topic.capitalize():12} {count:6} chunks")

        print(f"\n{'Total chunks':20} {total_chunks}")
        print(f"{'In ChromaDB':20} {collection_count}")
        print(f"{'Time elapsed':20} {elapsed/60:.1f} minutes")
        print(f"{'Avg per chunk':20} {elapsed/total_chunks:.2f}s" if total_chunks > 0 else "")

        # Estimate cost
        avg_tokens = 200  # Rough estimate
        total_tokens = total_chunks * avg_tokens
        cost = total_tokens * 0.00002 / 1000
        print(f"{'Est. cost':20} ${cost:.2f}")

        print("="*70)

        # Save summary
        summary = {
            'total_chunks': total_chunks,
            'collection_count': collection_count,
            'topics': results,
            'duration_seconds': elapsed,
            'estimated_cost_usd': round(cost, 2),
            'generated_at': datetime.now().isoformat(),
            'collection_name': self.collection_name,
            'chroma_db_dir': self.chroma_db_dir
        }

        summary_file = Path("data") / "embeddings_summary_scaled.json"
        summary_file.parent.mkdir(exist_ok=True)

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n💾 Summary saved to: {summary_file}")

        return results


def main():
    """Run scaled embeddings generation"""

    # Verify OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set")
        return

    generator = ScaledEmbeddingsGenerator()
    results = generator.process_all()

    print("\n✅ Embeddings generation complete!")
    print(f"📁 ChromaDB saved in: chroma_db_scaled/")


if __name__ == "__main__":
    main()
