"""
Pytest configuration and fixtures for assetable tests.
"""

import pytest


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
