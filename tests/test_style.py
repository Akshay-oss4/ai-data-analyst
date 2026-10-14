"""Guards for the style sheet and the header bar."""
from src.ui.style import CSS, build_top_bar_html


def test_top_bar_escapes_file_names_and_filters():
    html = build_top_bar_html("<img src=x onerror=alert(1)>.csv", (1200, 8), ["region in <b>West</b>"], demo=True)
    assert "<img" not in html and "&lt;img" in html
    assert "<b>West</b>" not in html
    assert "1,200 rows" in html and "Demo mode" in html and "1 filter active" in html


def test_css_braces_are_balanced():
    assert CSS.count("{") == CSS.count("}")
    assert "stMetric" in CSS