from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


@pytest.mark.slow
def test_cockpit_renders_and_runs_a_simulation():
    app = AppTest.from_file(str(APP), default_timeout=300)
    app.run()
    assert not app.exception
    assert any("LBO Deal Cockpit" in title.value for title in app.title)
    assert any("Gross IRR" in metric.label for metric in app.metric)

    # Move the exit multiple down; the headline IRR must fall.
    before = next(metric.value for metric in app.metric if metric.label == "Gross IRR")
    exit_slider = next(slider for slider in app.slider if "Exit multiple" in slider.label)
    exit_slider.set_value(exit_slider.value - 1.5).run()
    assert not app.exception
    after = next(metric.value for metric in app.metric if metric.label == "Gross IRR")
    assert float(after.strip("%")) < float(before.strip("%"))

    app.button[0].click().run()  # run the Monte Carlo on the risk tab
    assert not app.exception
    assert "trials" in app.session_state
