def pytest_addoption(parser):
    parser.addoption("--live-llm", action="store_true", default=False, help="Run LLM integration tests")
    parser.addoption("--live-rss", action="store_true", default=False, help="Run RSS fetching tests")
