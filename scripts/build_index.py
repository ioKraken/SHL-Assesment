"""Build the FAISS index from the catalog data.

Run this script after scraping the catalog to pre-build the index:
    python scripts/build_index.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.catalog import catalog_store
from app.retriever import retriever


def main():
    print("Loading catalog...")
    catalog_store.load()
    print(f"Loaded {len(catalog_store.assessments)} assessments")

    print("\nBuilding FAISS index...")
    retriever.initialize()

    print("\n--- Testing retrieval ---")
    test_queries = [
        "Java developer programming test",
        "personality assessment for manager",
        "customer service cognitive ability",
        "sales representative behavioral assessment",
        "data analyst numerical reasoning",
        "Python programming knowledge test",
    ]

    for query in test_queries:
        results = retriever.retrieve(query, top_k=5)
        print(f"\nQuery: \"{query}\"")
        for a, score in results:
            print(f"  [{score:.3f}] [{', '.join(a.test_types)}] {a.name}")

    print("\n✅ Index built and tested successfully!")


if __name__ == "__main__":
    main()
