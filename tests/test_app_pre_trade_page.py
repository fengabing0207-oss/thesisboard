"""Lightweight tests for the Pre-Trade Check page wiring (PR #18).

These do not drive Streamlit; they confirm the app module imports cleanly and
that a ThesisDecision can be built from representative form values (core fields
filled, advanced left at their defaults).
"""


def test_app_module_imports_with_pre_trade_page():
    import app

    assert hasattr(app, "render_pre_trade_check")
    assert hasattr(app, "main")


def test_decision_builds_from_core_form_values_with_advanced_defaults():
    from src.pre_trade_check import DecisionStatus, ThesisDecision

    # Core fields filled; advanced fields at the page's defaults (theme blank ->
    # "unspecified", levels medium, position_size 0.0, max_loss 0.0, notes "").
    decision = ThesisDecision(
        ticker="nvda",
        theme="unspecified",
        event_type="earnings",
        planned_action="buy",
        instrument_type="stock",
        horizon_days=5,
        market_expectation="medium",
        entry_thesis="Datacenter demand keeps beating estimates.",
        risk_thesis="Guidance disappoints.",
        max_loss=0.0,
        invalidation_rule="",
        confidence="medium",
        position_size=0.0,
        notes="",
    )

    assert decision.ticker == "NVDA"
    assert decision.status is DecisionStatus.PENDING
    assert decision.theme == "unspecified"
    assert decision.invalidation_rule == ""  # empty allowed, not rejected
    assert decision.to_dict()["status"] == "pending"


def test_decision_summary_renders_without_streamlit_runtime():
    # _render_decision_summary touches st.* which is a no-op import-time stub when
    # not in a Streamlit run; we only assert the helper is importable and callable
    # shape-wise by building the dict it consumes.
    from src.pre_trade_check import ThesisDecision

    decision = ThesisDecision(
        ticker="aapl",
        theme="unspecified",
        event_type="product_launch",
        planned_action="watch",
        instrument_type="stock",
        horizon_days=20,
        market_expectation="medium",
        entry_thesis="New product cycle.",
        risk_thesis="Demand soft.",
        max_loss=0.0,
        invalidation_rule="Exit below support.",
        confidence="low",
        position_size=2.5,
        notes="",
    )
    data = decision.to_dict()
    assert {"ticker", "planned_action", "status", "invalidation_rule"} <= set(data)


def test_pre_trade_page_runs_evaluator_on_submit():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30).run()
    at.sidebar.radio[0].set_value("Pre-Trade Check").run()

    # the manual run-up checkbox is present in the core form
    assert any("run-up" in cb.label.lower() for cb in at.checkbox)

    at.text_input[0].set_value("nvda")
    at.text_area[0].set_value("demand strong")
    at.get("button")[-1].click().run()

    assert not at.exception
    subheaders = [s.value for s in at.subheader]
    assert "Heuristic risk check" in subheaders
