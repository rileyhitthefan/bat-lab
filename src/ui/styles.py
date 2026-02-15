"""Load and inject CSS styles for the BatLab Streamlit app."""

from pathlib import Path

import streamlit as st

# CSS files live alongside this module (src/ui/)
_UI_DIR = Path(__file__).resolve().parent


def inject_styles(dark_theme: bool = False) -> None:
    """Inject CSS into the Streamlit app.

    Args:
        dark_theme: If True, also inject dark theme styles (for logged-in main app).
    """
    base_css = _UI_DIR / "styles.css"
    if base_css.exists():
        st.markdown(
            f"<style>\n{base_css.read_text()}\n</style>",
            unsafe_allow_html=True,
        )

    if dark_theme:
        dark_css = _UI_DIR / "theme_dark.css"
        if dark_css.exists():
            st.markdown(
                f"<style>\n{dark_css.read_text()}\n</style>",
                unsafe_allow_html=True,
            )
