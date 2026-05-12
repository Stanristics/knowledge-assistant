"""
Retrieval evaluation — tests the RAG pipeline against
synthetic Q&A pairs. Run: pytest tests/test_rag.py -v
"""

import pytest
import os
import tempfile
from pathlib import Path

# Minimal test document so tests run without real PDFs
SAMPLE_TEXT = """
Company: Acme GmbH
Founded: 2018
Headquarters: Berlin, Germany
CEO: Maria Hoffmann

Product Overview:
Acme GmbH builds B2B SaaS tools for supply chain optimisation.
Our flagship product, SupplyIQ, uses machine learning to predict
demand and reduce inventory holding costs by up to 30%.

Pricing:
- Starter plan: €299/month, up to 5 users
- Growth plan: €799/month, up to 25 users
- Enterprise plan: custom pricing

Support:
Email support@acme-gmbh.de or call +49 30 12345678.
Office hours: Monday to Friday, 09:00 to 18:00 CET.
"""

TEST_PAIRS = [
    {
        "question": "Who is the CEO of Acme GmbH?",
        "expected_keywords": ["Maria Hoffmann"],
    },
    {
        "question": "What does SupplyIQ do?",
        "expected_keywords": ["demand", "inventory"],
    },
    {
        "question": "How much does the Growth plan cost?",
        "expected_keywords": ["799"],
    },
    {
        "question": "Where is the company headquartered?",
        "expected_keywords": ["Berlin"],
    },
]


@pytest.fixture(scope="module")
def vectorstore(tmp_path_factory):
    """Build a test vectorstore from the sample document."""
    from app.rag_engine import build_vectorstore
    from langchain.schema import Document

    tmp_dir = str(tmp_path_factory.mktemp("chroma"))
    docs = [Document(page_content=SAMPLE_TEXT, metadata={"source": "test_doc.txt", "page": 1})]
    return build_vectorstore(docs, tmp_dir)


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM tests",
)
def test_retrieval_hit_rate(vectorstore):
    """Hit rate should be >= 75% on basic factual questions."""
    from app.rag_engine import build_chain, evaluate_retrieval

    chain = build_chain(vectorstore, model_name="gpt-3.5-turbo", temperature=0.0)
    report = evaluate_retrieval(chain, TEST_PAIRS)

    print(f"\nHit rate: {report['hit_rate']} ({report['hits']}/{report['total']})")
    for r in report["details"]:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['question']}")

    assert report["hit_rate"] >= 0.75, (
        f"Hit rate {report['hit_rate']} below threshold 0.75"
    )


def test_vectorstore_has_chunks(vectorstore):
    """Vectorstore should contain at least one chunk after indexing."""
    count = vectorstore._collection.count()
    assert count > 0, "Vectorstore is empty after indexing"


def test_retriever_returns_results(vectorstore):
    """Retriever should return documents for a known query."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    docs = retriever.get_relevant_documents("CEO of Acme")
    assert len(docs) > 0, "Retriever returned no documents"
    assert any("Maria" in d.page_content for d in docs), (
        "Expected to find CEO name in retrieved chunks"
    )
