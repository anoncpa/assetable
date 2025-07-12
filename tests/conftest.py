"""
Pytest configuration and fixtures for assetable tests.
"""
from pathlib import Path
import pytest
from tests.test_integration import TestDocumentCreation


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ollama: mark test as requiring Ollama server"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add asyncio marker to async tests
        if "asyncio" in item.keywords:
            item.add_marker(pytest.mark.asyncio)

        # Add ollama marker to tests that use Ollama
        if any(cls in str(item.fspath) for cls in ["ollama", "vision", "ai_steps"]):
            item.add_marker(pytest.mark.ollama)

@pytest.fixture(scope="session")
def temp_workspace():
    """Create a temporary workspace for integration tests."""
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def sample_pdf_simple(temp_workspace):
    """Create a simple PDF for testing."""
    pdf_path = temp_workspace / "sample_simple.pdf"
    TestDocumentCreation.create_simple_test_pdf(pdf_path, pages=3)
    return pdf_path

@pytest.fixture(scope="session")
def sample_pdf_complex(temp_workspace):
    """Create a complex PDF for testing."""
    pdf_path = temp_workspace / "sample_complex.pdf"
    TestDocumentCreation.create_complex_test_pdf(pdf_path, num_pages=5)
    return pdf_path
