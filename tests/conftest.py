def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: test that performs a live network call; skipped unless "
        "THESISBOARD_RUN_NETWORK_TESTS=1 is set.",
    )
