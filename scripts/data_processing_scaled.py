"""
Full-Text Processing with Section Awareness
Process 815 papers with sections into ~75,000 chunks
"""

import json
from pathlib import Path
from typing import List, Dict
import re
from datetime import datetime


class FullTextProcessor:
    """Process full-text papers with section-aware chunking"""

    def __init__(
        self,
        input_dir: str = "data/papers_fulltext",
        output_dir: str = "data/processed_fulltext"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.topics = ["prostate", "bladder", "kidney", "testicular"]

        # Chunking parameters - section-aware
        self.chunk_size = 200  # Words per chunk
        self.overlap = 30      # Overlap between chunks

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]\']', '', text)
        return text.strip()

    def chunk_section(
        self,
        section_text: str,
        section_name: str,
        paper_metadata: Dict
    ) -> List[Dict]:
        """Chunk a single section"""
        words = section_text.split()
        chunks = []

        # If section is short, keep as single chunk
        if len(words) <= self.chunk_size:
            return [{
                'text': section_text,
                'section_name': section_name,
                'metadata': paper_metadata,
                'chunk_index': 0,
                'total_chunks': 1
            }]

        # Create overlapping chunks within section
        i = 0
        chunk_index = 0

        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)

            chunks.append({
                'text': chunk_text,
                'section_name': section_name,
                'metadata': paper_metadata,
                'chunk_index': chunk_index,
                'total_chunks': 0  # Will update later
            })

            i += self.chunk_size - self.overlap
            chunk_index += 1

        # Update total_chunks
        for chunk in chunks:
            chunk['total_chunks'] = len(chunks)

        return chunks

    def process_paper(self, paper: Dict) -> List[Dict]:
        """Process a single full-text paper"""
        # Prepare metadata
        metadata = {
            'pmc_id': paper.get('pmc_id', ''),
            'pmid': paper.get('pmid', ''),
            'title': paper.get('title', ''),
            'authors': paper.get('authors', [])[:3],
            'journal': paper.get('journal', ''),
            'year': paper.get('year', ''),
            'doi': paper.get('doi', ''),
            'topic': paper.get('topic', ''),
            'num_sections': len(paper.get('sections', []))
        }

        all_chunks = []

        # Process each section
        for section in paper.get('sections', []):
            section_name = section.get('section_name', 'Unknown')
            section_content = section.get('content', '')

            if not section_content:
                continue

            # Clean section content
            section_content = self.clean_text(section_content)

            # Chunk the section
            section_chunks = self.chunk_section(
                section_content,
                section_name,
                metadata
            )

            all_chunks.extend(section_chunks)

        return all_chunks

    def process_topic(self, topic: str) -> int:
        """Process all papers for a topic"""
        print(f"\n{'='*70}")
        print(f"⚙️  Processing: {topic.upper()}")
        print(f"{'='*70}")

        # Load papers
        input_file = self.input_dir / topic / f"{topic}_fulltext.json"

        if not input_file.exists():
            print(f"❌ No papers found for {topic}")
            return 0

        with open(input_file, 'r', encoding='utf-8') as f:
            papers = json.load(f)

        print(f"📄 Loaded {len(papers)} papers")

        # Calculate total sections
        total_sections = sum(p.get('num_sections', 0) for p in papers)
        avg_sections = total_sections / len(papers) if papers else 0

        print(f"📑 Total sections: {total_sections} (avg: {avg_sections:.1f} per paper)")

        # Process papers
        all_chunks = []

        for i, paper in enumerate(papers, 1):
            if i % 50 == 0:
                print(f"   Progress: {i}/{len(papers)} papers")

            chunks = self.process_paper(paper)
            all_chunks.extend(chunks)

        print(f"✅ Created {len(all_chunks)} chunks")
        print(f"📊 Avg chunks per paper: {len(all_chunks) / len(papers):.1f}")

        # Save chunks
        output_file = self.output_dir / f"{topic}_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)

        print(f"💾 Saved to {output_file}")

        # Calculate statistics
        avg_chunk_length = sum(len(c['text'].split()) for c in all_chunks) / len(all_chunks)

        # Count chunks per section type
        section_counts = {}
        for chunk in all_chunks:
            section = chunk.get('section_name', 'Unknown')
            section_counts[section] = section_counts.get(section, 0) + 1

        stats = {
            'topic': topic,
            'papers': len(papers),
            'sections': total_sections,
            'chunks': len(all_chunks),
            'avg_sections_per_paper': round(avg_sections, 1),
            'avg_chunks_per_paper': round(len(all_chunks) / len(papers), 1),
            'avg_chunk_length': round(avg_chunk_length, 1),
            'top_sections': sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:10],
            'processed_at': datetime.now().isoformat()
        }

        stats_file = self.output_dir / f"{topic}_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        return len(all_chunks)

    def process_all(self):
        """Process all topics"""
        print("="*70)
        print(" "*12 + "⚙️  FULL-TEXT PROCESSING")
        print("="*70)
        print("\nChunking strategy: Section-aware with overlap")
        print(f"Chunk size: {self.chunk_size} words")
        print(f"Overlap: {self.overlap} words")
        print()

        total_papers = 0
        total_chunks = 0
        results = {}

        for topic in self.topics:
            chunks_count = self.process_topic(topic)
            results[topic] = chunks_count
            total_chunks += chunks_count

            # Count papers
            input_file = self.input_dir / topic / f"{topic}_fulltext.json"
            if input_file.exists():
                with open(input_file) as f:
                    papers = json.load(f)
                    total_papers += len(papers)

        # Summary
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)

        for topic, chunks in results.items():
            # Load paper count
            input_file = self.input_dir / topic / f"{topic}_fulltext.json"
            if input_file.exists():
                with open(input_file) as f:
                    papers = json.load(f)
                paper_count = len(papers)
                avg = chunks / paper_count if paper_count > 0 else 0
                print(f"✅ {topic.capitalize():12} {paper_count:3} papers → {chunks:6} chunks (avg: {avg:.1f}/paper)")

        print(f"\n{'Total papers':20} {total_papers}")
        print(f"{'Total chunks':20} {total_chunks}")
        print(f"{'Avg chunks/paper':20} {total_chunks/total_papers:.1f}")
        print("="*70)

        # Save overall summary
        summary = {
            'total_papers': total_papers,
            'total_chunks': total_chunks,
            'avg_chunks_per_paper': round(total_chunks / total_papers, 1) if total_papers > 0 else 0,
            'topics': results,
            'chunk_size': self.chunk_size,
            'overlap': self.overlap,
            'processed_at': datetime.now().isoformat()
        }

        summary_file = self.output_dir / "processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return results


def main():
    """Run full-text processing"""
    processor = FullTextProcessor()
    results = processor.process_all()

    print("\n✅ Processing complete!")
    print(f"📁 Chunks saved in: data/processed_fulltext/")


if __name__ == "__main__":
    main()
