from src.market_sources import (
    CATEGORY_ORDER,
    MARKET_SOURCES,
    MarketSource,
    sources_for_category,
)


def test_config_loads_with_sources():
    assert 15 <= len(MARKET_SOURCES) <= 18
    for source in MARKET_SOURCES:
        assert isinstance(source, MarketSource)
        assert source.id and source.name and source.category
        assert source.description and source.best_for and source.caution
        assert source.priority >= 1


def test_each_source_has_exactly_one_url_form():
    for source in MARKET_SOURCES:
        assert (source.url is None) ^ (source.url_template is None), source.id


def test_all_categories_are_represented():
    categories = {source.category for source in MARKET_SOURCES}
    assert set(CATEGORY_ORDER) <= categories


def test_source_ids_are_unique():
    ids = [source.id for source in MARKET_SOURCES]
    assert len(ids) == len(set(ids))


def test_url_template_fills_ticker():
    ticker_specific = [s for s in MARKET_SOURCES if s.is_ticker_specific()]
    assert ticker_specific  # there are ticker-specific sources
    for source in ticker_specific:
        url = source.resolve_url("nvda")
        assert "{ticker}" not in url
        assert "NVDA" in url  # uppercased


def test_no_source_has_empty_or_broken_url():
    for source in MARKET_SOURCES:
        url = source.resolve_url("TEST")
        assert url, source.id
        assert url.startswith("http"), source.id


def test_static_sources_ignore_ticker():
    for source in MARKET_SOURCES:
        if not source.is_ticker_specific():
            assert source.resolve_url("ANY") == source.url


def test_sources_for_category_sorted_by_priority():
    for category in CATEGORY_ORDER:
        priorities = [source.priority for source in sources_for_category(category)]
        assert priorities == sorted(priorities)
        assert priorities  # each category has at least one source
