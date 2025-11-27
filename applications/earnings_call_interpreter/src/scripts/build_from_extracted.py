import os
import sys

# Make project root importable as a package root
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.graph.ingest_extracted_facts import build_kg_and_index_from_extracted


def main():
    build_kg_and_index_from_extracted("data/processed")


if __name__ == "__main__":
    main()
