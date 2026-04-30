"""
Full-Text Data Collection from PubMed Central
Collects 1000 full-text papers with sections across 4 cancer types
"""

import os
from pathlib import Path
from typing import List, Dict, Optional
import time
from datetime import datetime
import xml.etree.ElementTree as ET
import requests

from Bio import Entrez

# Configuration
Entrez.email = os.getenv("ENTREZ_EMAIL", "your.email@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")


class FullTextCollector:
    """Collect full-text papers from PMC with section parsing"""

    def __init__(self, output_dir: str = "data/papers_fulltext"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define topics with search queries
        self.topics = {
            "prostate": {
                "query": (
                    '("prostate cancer"[Title/Abstract] OR "prostatic neoplasms"[MeSH Terms]) '
                    'AND (treatment[Title/Abstract] OR therapy[Title/Abstract] OR diagnosis[Title/Abstract])'
                ),
                "target_papers": 250
            },
            "bladder": {
                "query": (
                    '("bladder cancer"[Title/Abstract] OR "urinary bladder neoplasms"[MeSH Terms]) '
                    'AND (treatment[Title/Abstract] OR immunotherapy[Title/Abstract] OR BCG[Title/Abstract])'
                ),
                "target_papers": 250
            },
            "kidney": {
                "query": (
                    '("kidney cancer"[Title/Abstract] OR "renal cell carcinoma"[Title/Abstract]) '
                    'AND ("targeted therapy"[Title/Abstract] OR immunotherapy[Title/Abstract])'
                ),
                "target_papers": 250
            },
            "testicular": {
                "query": (
                    '("testicular cancer"[Title/Abstract] OR "testis neoplasms"[MeSH Terms]) '
                    'AND (treatment[Title/Abstract] OR chemotherapy[Title/Abstract])'
                ),
                "target_papers": 250
            }
        }

        print(f"📧 Email: {Entrez.email}")
        print(f"🔑 API Key: {'✅ Set' if Entrez.api_key else '❌ Not set'}")

    def search_pmc_papers(self, query: str, max_results: int = 500) -> List[str]:
        """Search PMC (not PubMed) for papers with full-text"""
        print(f"\n🔍 Searching PMC for full-text papers...")

        try:
            # Search PMC directly for open access papers
            search_query = (
                f"{query} "
                f"AND (2015:2025[PDAT]) "  # Last 10 years
                f"AND (Review[PT] OR Journal Article[PT])"
            )

            # Use PMC database for full-text
            handle = Entrez.esearch(
                db="pmc",  # PMC database (not pubmed)
                term=search_query,
                retmax=max_results,
                sort="relevance",
                usehistory="y"
            )

            results = Entrez.read(handle)
            handle.close()

            pmc_ids = results["IdList"]
            print(f"   ✅ Found {len(pmc_ids)} PMC papers with full-text")

            return pmc_ids

        except Exception as e:
            print(f"   ❌ Error: {e}")
            return []

    def fetch_pmc_fulltext(self, pmc_id: str) -> Optional[Dict]:
        """Fetch full-text XML from PMC and parse sections"""
        try:
            # Fetch XML
            handle = Entrez.efetch(
                db="pmc",
                id=pmc_id,
                rettype="xml",
                retmode="xml"
            )

            xml_content = handle.read()
            handle.close()

            # Parse XML
            root = ET.fromstring(xml_content)

            # Extract metadata
            article = root.find(".//article")
            if article is None:
                return None

            # Get front matter (metadata)
            front = article.find(".//front")
            if front is None:
                return None

            # Extract title
            title_group = front.find(".//title-group/article-title")
            title = self._get_text(title_group) if title_group is not None else ""

            # Extract abstract
            abstract_elem = front.find(".//abstract")
            abstract = self._get_text(abstract_elem) if abstract_elem is not None else ""

            # Extract authors
            authors = []
            for contrib in front.findall(".//contrib[@contrib-type='author']"):
                surname = contrib.find(".//surname")
                given_names = contrib.find(".//given-names")
                if surname is not None:
                    author = self._get_text(surname)
                    if given_names is not None:
                        author += f" {self._get_text(given_names)}"
                    authors.append(author)

            # Extract journal info
            journal = front.find(".//journal-title")
            journal_name = self._get_text(journal) if journal is not None else ""

            # Extract year
            pub_date = front.find(".//pub-date[@pub-type='epub']") or front.find(".//pub-date")
            year = ""
            if pub_date is not None:
                year_elem = pub_date.find(".//year")
                year = self._get_text(year_elem) if year_elem is not None else ""

            # Extract DOI
            article_id = front.find(".//article-id[@pub-id-type='doi']")
            doi = self._get_text(article_id) if article_id is not None else ""

            # Extract PMID
            pmid_elem = front.find(".//article-id[@pub-id-type='pmid']")
            pmid = self._get_text(pmid_elem) if pmid_elem is not None else ""

            # Extract body sections
            body = article.find(".//body")
            sections = []

            if body is not None:
                for sec in body.findall(".//sec"):
                    section_title = sec.find(".//title")
                    section_name = self._get_text(section_title) if section_title is not None else "Unknown"

                    # Get all paragraphs in section
                    paragraphs = []
                    for p in sec.findall(".//p"):
                        para_text = self._get_text(p)
                        if para_text:
                            paragraphs.append(para_text)

                    if paragraphs:
                        sections.append({
                            'section_name': section_name,
                            'content': ' '.join(paragraphs)
                        })

            # If no body sections, use abstract as fallback
            if not sections and abstract:
                sections.append({
                    'section_name': 'Abstract',
                    'content': abstract
                })

            # Build paper dict
            paper = {
                'pmc_id': pmc_id,
                'pmid': pmid,
                'title': title,
                'abstract': abstract,
                'authors': authors[:10],  # First 10 authors
                'journal': journal_name,
                'year': year,
                'doi': doi,
                'sections': sections,
                'num_sections': len(sections)
            }

            return paper

        except Exception as e:
            print(f"   ⚠️  Failed to fetch PMC{pmc_id}: {e}")
            return None

    def _get_text(self, element) -> str:
        """Extract text from XML element recursively"""
        if element is None:
            return ""

        text_parts = []

        # Get direct text
        if element.text:
            text_parts.append(element.text)

        # Get text from children
        for child in element:
            text_parts.append(self._get_text(child))
            if child.tail:
                text_parts.append(child.tail)

        return ' '.join(text_parts).strip()

    def collect_papers_for_topic(self, topic: str, config: Dict) -> int:
        """Collect full-text papers for a topic"""
        print(f"\n{'='*70}")
        print(f"📚 Collecting full-text papers for: {topic.upper()}")
        print(f"{'='*70}")

        topic_dir = self.output_dir / topic
        topic_dir.mkdir(exist_ok=True)

        target = config['target_papers']

        # Search for more papers than needed (some won't have full-text)
        search_count = target * 3  # Search 3x more to account for failures
        pmc_ids = self.search_pmc_papers(config['query'], search_count)

        if not pmc_ids:
            print(f"❌ No papers found for {topic}")
            return 0

        print(f"\n📥 Fetching full-text for {len(pmc_ids)} papers...")
        print(f"   Target: {target} papers with full-text sections")

        papers = []
        failed_count = 0

        for i, pmc_id in enumerate(pmc_ids, 1):
            if len(papers) >= target:
                print(f"\n✅ Reached target: {target} papers")
                break

            if i % 10 == 0:
                print(f"   Progress: {len(papers)}/{target} papers (tried {i}/{len(pmc_ids)})")

            paper = self.fetch_pmc_fulltext(pmc_id)

            if paper and paper.get('sections') and len(paper['sections']) > 1:
                # Only keep papers with multiple sections (full-text)
                paper['topic'] = topic
                paper['collected_at'] = datetime.now().isoformat()
                papers.append(paper)
            else:
                failed_count += 1

            # Rate limiting
            time.sleep(0.1 if Entrez.api_key else 0.34)

        print(f"\n📊 Results for {topic}:")
        print(f"   ✅ Collected: {len(papers)} full-text papers")
        print(f"   ⚠️  Failed/No full-text: {failed_count}")
        print(f"   📄 Avg sections per paper: {sum(p['num_sections'] for p in papers) / len(papers):.1f}" if papers else "")

        # Save papers
        if papers:
            import json

            output_file = topic_dir / f"{topic}_fulltext.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)

            print(f"💾 Saved to {output_file}")

            # Save summary
            summary = {
                'topic': topic,
                'count': len(papers),
                'target': target,
                'collected_at': datetime.now().isoformat(),
                'avg_sections': sum(p['num_sections'] for p in papers) / len(papers)
            }

            summary_file = topic_dir / f"{topic}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

        return len(papers)

    def collect_all(self):
        """Collect full-text papers for all topics"""
        start_time = time.time()

        print("="*70)
        print(" "*12 + "🚀 FULL-TEXT DATA COLLECTION")
        print("="*70)
        print(f"\nTarget: {sum(t['target_papers'] for t in self.topics.values())} full-text papers")
        print(f"Topics: {', '.join(self.topics.keys())}")
        print(f"Source: PubMed Central (PMC)")
        print()

        total_collected = 0
        results = {}

        for topic, config in self.topics.items():
            count = self.collect_papers_for_topic(topic, config)
            results[topic] = count
            total_collected += count

            if topic != list(self.topics.keys())[-1]:
                print("\n⏸️  Pausing 5 seconds before next topic...")
                time.sleep(5)

        elapsed = time.time() - start_time

        # Summary
        print("\n" + "="*70)
        print("📊 COLLECTION SUMMARY")
        print("="*70)

        for topic, count in results.items():
            target = self.topics[topic]['target_papers']
            percentage = (count / target * 100) if target > 0 else 0
            status = "✅" if percentage >= 80 else "⚠️"
            print(f"{status} {topic.capitalize():12} {count:4}/{target:4} ({percentage:.1f}%)")

        print(f"\n{'Total':15} {total_collected:4} full-text papers")
        print(f"{'Time elapsed':15} {elapsed/60:.1f} minutes")
        print(f"{'Avg per paper':15} {elapsed/total_collected:.1f}s" if total_collected > 0 else "")
        print("="*70)

        # Save overall summary
        import json

        summary = {
            'total_papers': total_collected,
            'topics': results,
            'collected_at': datetime.now().isoformat(),
            'duration_seconds': elapsed,
            'source': 'PubMed Central (PMC)'
        }

        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        return results


def main():
    """Run full-text data collection"""

    if not os.getenv("NCBI_API_KEY"):
        print("⚠️  Warning: NCBI_API_KEY not set. Collection will be slower.")
        response = input("\nContinue anyway? (y/n): ").lower()
        if response != 'y':
            return

    collector = FullTextCollector()
    results = collector.collect_all()

    print("\n✅ Full-text collection complete!")
    print(f"📁 Papers saved in: data/papers_fulltext/")


if __name__ == "__main__":
    main()
