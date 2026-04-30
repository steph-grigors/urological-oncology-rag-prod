"""
Data Quality Assessment Module
Analyzes collected papers for completeness and quality
Updated for scaled multi-topic corpus
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import List, Dict, Any


class DataQualityAssessor:
    """Assesses quality of collected medical papers"""

    def __init__(self, topic: str = "prostate", data_dir: str = "data/papers_fulltext"):
        """
        Initialize quality assessor for a specific topic

        Args:
            topic: Cancer type (prostate, bladder, kidney, testicular)
            data_dir: Base directory containing topic folders
        """
        self.topic = topic
        self.data_dir = Path(data_dir) / topic
        self.papers = []

    def load_papers(self) -> List[Dict[Any, Any]]:
        """Load papers from topic-specific JSON file"""
        print(f"📂 Loading papers from {self.data_dir}...")

        # Load from the single JSON file per topic
        papers_file = self.data_dir / f"{self.topic}_fulltext.json"

        if not papers_file.exists():
            print(f"   ❌ File not found: {papers_file}")
            return []

        try:
            with open(papers_file, 'r', encoding='utf-8') as f:
                self.papers = json.load(f)
            print(f"   ✅ Loaded {len(self.papers)} papers\n")
        except Exception as e:
            print(f"   ⚠️ Failed to load {papers_file}: {e}")

        return self.papers

    def check_missing_fields(self) -> Dict[str, int]:
        """Check for missing critical fields"""
        print(f"🔍 Checking for missing fields...")

        missing = {
            'title': 0,
            'abstract': 0,
            'sections': 0,
            'authors': 0,
            'year': 0,
            'journal': 0,
            'pmc_id': 0,
            'pmid': 0
        }

        for paper in self.papers:
            if not paper.get('title') or paper.get('title') == 'No title':
                missing['title'] += 1
            if not paper.get('abstract'):
                missing['abstract'] += 1
            if not paper.get('sections') or len(paper.get('sections', [])) == 0:
                missing['sections'] += 1
            if not paper.get('authors') or len(paper.get('authors', [])) == 0:
                missing['authors'] += 1
            if not paper.get('year'):
                missing['year'] += 1
            if not paper.get('journal'):
                missing['journal'] += 1
            if not paper.get('pmc_id'):
                missing['pmc_id'] += 1
            if not paper.get('pmid'):
                missing['pmid'] += 1

        for field, count in missing.items():
            percentage = (count / len(self.papers)) * 100 if self.papers else 0
            status = "✅" if count == 0 else "⚠️"
            print(f"   {status} Papers missing {field}: {count} ({percentage:.1f}%)")

        print()
        return missing

    def analyze_sections(self) -> Dict[str, int]:
        """Analyze section distribution across papers"""
        print(f"📑 Analyzing section distribution...")

        all_section_names = []
        section_counts_per_paper = []

        for paper in self.papers:
            sections = paper.get('sections', [])
            section_names = [s.get('section_name', 'Unknown') for s in sections]
            all_section_names.extend(section_names)
            section_counts_per_paper.append(len(sections))

        if not section_counts_per_paper:
            print("   ⚠️ No sections found in papers")
            print()
            return {}

        section_distribution = Counter(all_section_names)

        print(f"   Total unique section names: {len(section_distribution)}")
        print(f"   Average sections per paper: {sum(section_counts_per_paper)/len(section_counts_per_paper):.1f}")
        print(f"\n   Top 10 most common sections:")

        for section, count in section_distribution.most_common(10):
            percentage = (count / len(self.papers)) * 100
            print(f"      {section}: {count} papers ({percentage:.1f}%)")

        # Check for important sections
        print(f"\n   Standard section coverage:")
        important_patterns = ['introduction', 'method', 'result', 'discussion']

        for pattern in important_patterns:
            # Count sections containing this pattern (case-insensitive)
            count = sum(1 for s in all_section_names if pattern in s.lower())
            percentage = (count / len(self.papers)) * 100
            status = "✅" if percentage > 50 else "⚠️"
            print(f"      {status} *{pattern}*: {count} occurrences ({percentage:.1f}%)")

        print()
        return dict(section_distribution)

    def analyze_text_lengths(self) -> Dict[str, Any]:
        """Analyze text length statistics"""
        print(f"📏 Analyzing text lengths...")

        total_text_lengths = []
        section_counts = []
        abstract_lengths = []

        for paper in self.papers:
            # Calculate total text length from all sections
            sections = paper.get('sections', [])
            total_length = sum(len(s.get('content', '')) for s in sections)
            total_text_lengths.append(total_length)
            section_counts.append(len(sections))

            # Abstract length
            abstract = paper.get('abstract', '')
            abstract_lengths.append(len(abstract))

        if not total_text_lengths:
            print("   ⚠️ No text data found")
            print()
            return {}

        stats = {
            'total_text': {
                'avg': sum(total_text_lengths) / len(total_text_lengths),
                'min': min(total_text_lengths),
                'max': max(total_text_lengths),
                'median': sorted(total_text_lengths)[len(total_text_lengths)//2]
            },
            'sections': {
                'avg': sum(section_counts) / len(section_counts),
                'min': min(section_counts),
                'max': max(section_counts)
            },
            'abstracts': {
                'avg': sum(abstract_lengths) / len(abstract_lengths),
                'min': min(abstract_lengths),
                'max': max(abstract_lengths)
            }
        }

        print(f"   Total text statistics (all sections):")
        print(f"      Average: {stats['total_text']['avg']:,.0f} characters")
        print(f"      Median:  {stats['total_text']['median']:,.0f} characters")
        print(f"      Range:   {stats['total_text']['min']:,} - {stats['total_text']['max']:,}")

        print(f"\n   Section count statistics:")
        print(f"      Average: {stats['sections']['avg']:.1f} sections per paper")
        print(f"      Range:   {stats['sections']['min']} - {stats['sections']['max']}")

        print(f"\n   Abstract statistics:")
        print(f"      Average: {stats['abstracts']['avg']:,.0f} characters")
        print(f"      Range:   {stats['abstracts']['min']:,} - {stats['abstracts']['max']:,}")

        print()
        return stats

    def identify_problematic_papers(self) -> List[Dict[str, Any]]:
        """Identify papers with potential issues"""
        print(f"⚠️ Identifying problematic papers...")

        problematic = []

        for paper in self.papers:
            issues = []

            # Calculate total text length
            sections = paper.get('sections', [])
            total_text_length = sum(len(s.get('content', '')) for s in sections)

            if total_text_length < 5000:
                issues.append(f"Very short ({total_text_length:,} chars)")

            # Check sections
            if not sections:
                issues.append("No sections")
            elif len(sections) < 3:
                issues.append(f"Only {len(sections)} sections")

            # Check abstract
            abstract = paper.get('abstract', '')
            if not abstract:
                issues.append("No abstract")
            elif len(abstract) < 100:
                issues.append("Very short abstract")

            # Check title
            title = paper.get('title', '')
            if not title or title == 'No title':
                issues.append("Missing title")

            # Check important sections
            section_names = [s.get('section_name', '').lower() for s in sections]
            has_methods = any('method' in s for s in section_names)
            has_results = any('result' in s for s in section_names)

            if not has_methods:
                issues.append("No methods section")
            if not has_results:
                issues.append("No results section")

            if issues:
                problematic.append({
                    'pmc_id': paper.get('pmc_id', 'Unknown'),
                    'pmid': paper.get('pmid', 'Unknown'),
                    'title': title[:60] + '...' if len(title) > 60 else title,
                    'issues': issues,
                    'total_text_length': total_text_length,
                    'section_count': len(sections)
                })

        if problematic:
            print(f"   Found {len(problematic)} papers with potential issues:")
            print(f"   (Showing top 10 most problematic)\n")

            # Sort by number of issues
            problematic.sort(key=lambda x: len(x['issues']), reverse=True)

            for i, p in enumerate(problematic[:10], 1):
                print(f"   {i}. PMC{p['pmc_id']} / PMID{p['pmid']}")
                print(f"      Title: {p['title']}")
                print(f"      Issues: {', '.join(p['issues'])}")
                print(f"      Length: {p['total_text_length']:,} chars, {p['section_count']} sections")
                print()
        else:
            print(f"   ✅ No problematic papers found!")

        print()
        return problematic

    def analyze_publication_years(self) -> Dict[str, int]:
        """Analyze publication year distribution"""
        print(f"📅 Analyzing publication years...")

        years = []
        missing_years = 0

        for paper in self.papers:
            year = paper.get('year', '')
            if year and str(year).isdigit():
                years.append(str(year))
            else:
                missing_years += 1

        year_distribution = Counter(years)

        print(f"   Papers with valid years: {len(years)}/{len(self.papers)}")
        print(f"   Papers with missing/invalid years: {missing_years}")

        if year_distribution:
            print(f"\n   Publication year distribution:")
            for year, count in sorted(year_distribution.items(), reverse=True):
                percentage = (count / len(self.papers)) * 100
                bar = '█' * int(percentage / 2)
                print(f"      {year}: {count:2d} papers {bar} ({percentage:.1f}%)")

        print()
        return dict(year_distribution)

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        print("="*70)
        print(f" "*15 + f"📊 DATA QUALITY REPORT - {self.topic.upper()}")
        print("="*70)
        print()

        # Basic stats
        print(f"📦 Dataset Overview:")
        print(f"   Topic: {self.topic}")
        print(f"   Total papers: {len(self.papers)}")
        print(f"   Data directory: {self.data_dir}")
        print()

        # Run all assessments
        missing_fields = self.check_missing_fields()
        section_stats = self.analyze_sections()
        text_stats = self.analyze_text_lengths()
        problematic = self.identify_problematic_papers()
        year_stats = self.analyze_publication_years()

        # Overall quality score
        print("="*70)
        print("🎯 Overall Quality Assessment:")
        print("="*70)

        quality_score = 100

        # Deduct points for issues
        if missing_fields.get('title', 0) > 0:
            quality_score -= 20
        if missing_fields.get('abstract', 0) > len(self.papers) * 0.1:
            quality_score -= 15
        if missing_fields.get('sections', 0) > 0:
            quality_score -= 20
        if len(problematic) > len(self.papers) * 0.3:
            quality_score -= 15
        if text_stats and text_stats.get('total_text', {}).get('avg', 0) < 10000:
            quality_score -= 10

        quality_score = max(0, quality_score)

        if quality_score >= 90:
            grade = "A (Excellent)"
            emoji = "🌟"
        elif quality_score >= 80:
            grade = "B (Good)"
            emoji = "✅"
        elif quality_score >= 70:
            grade = "C (Acceptable)"
            emoji = "👍"
        elif quality_score >= 60:
            grade = "D (Needs Improvement)"
            emoji = "⚠️"
        else:
            grade = "F (Poor)"
            emoji = "❌"

        print(f"\n   {emoji} Quality Score: {quality_score}/100 - Grade: {grade}")
        print()

        # Recommendations
        print("💡 Recommendations:")
        if len(problematic) > 0:
            print(f"   • {len(problematic)} papers have potential issues - review individually")
        if missing_fields.get('abstract', 0) > 0:
            print(f"   • {missing_fields['abstract']} papers missing abstracts")
        if text_stats and text_stats.get('total_text', {}).get('avg', 0) < 15000:
            print(f"   • Average paper length is shorter than typical research articles")
        if quality_score >= 90:
            print(f"   • ✅ Data quality is excellent! Ready for use.")

        print()
        print("="*70)

        return {
            'topic': self.topic,
            'total_papers': len(self.papers),
            'missing_fields': missing_fields,
            'problematic_count': len(problematic),
            'problematic_papers': problematic[:10],  # Top 10
            'quality_score': quality_score,
            'grade': grade,
            'text_stats': text_stats,
            'section_stats': dict(list(section_stats.items())[:20]) if section_stats else {},  # Top 20 sections
            'year_stats': year_stats
        }

    def save_report(self, output_file: str = None):
        """Save quality report to JSON file"""
        if output_file is None:
            output_file = f"data/quality_report_{self.topic}.json"

        report = self.generate_summary_report()

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

        print(f"💾 Report saved to: {output_file}")
        print()


def main():
    """Run quality assessment on all topics"""
    topics = ['prostate', 'bladder', 'kidney', 'testicular']

    all_reports = {}

    for topic in topics:
        print(f"\n{'='*70}")
        print(f"{'='*70}")
        print(f" "*20 + f"ASSESSING {topic.upper()}")
        print(f"{'='*70}")
        print(f"{'='*70}\n")

        assessor = DataQualityAssessor(topic=topic)
        assessor.load_papers()

        if assessor.papers:
            assessor.save_report()
            all_reports[topic] = assessor.generate_summary_report()
        else:
            print(f"⚠️ Skipping {topic} - no papers found\n")

    # Overall summary
    if all_reports:
        print("\n" + "="*70)
        print(" "*15 + "📊 OVERALL CORPUS SUMMARY")
        print("="*70)
        print()

        total_papers = sum(r['total_papers'] for r in all_reports.values())
        avg_quality = sum(r['quality_score'] for r in all_reports.values()) / len(all_reports)

        print(f"Total papers across all topics: {total_papers}")
        print(f"Average quality score: {avg_quality:.1f}/100")
        print()
        
        for topic, report in all_reports.items():
            print(f"   {topic.capitalize():12} {report['total_papers']:3d} papers | Quality: {report['quality_score']}/100 ({report['grade']})")

        print()
        print("="*70)


if __name__ == "__main__":
    main()
