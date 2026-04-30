"""
Root pytest conftest — ensures the project root is on sys.path so that
`from src.ingestion.chunk import ...` works in all test modules.
Run tests from the project root: pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
