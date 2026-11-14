"""Shared Plotly theme: one validated palette, recessive axes, consistent typography.

Categorical colours are assigned in a fixed order (never cycled) and follow the
entity, not its rank. Magnitude uses a single-hue sequential ramp.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
SEQUENTIAL_BLUE = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#1c5cab"],
    [1.0, "#0d366b"],
]

# Fixed entity -> colour mappings, reused by every figure.
WEEK_STATUS_COLORS = {"Weekday": CATEGORICAL[0], "Weekend": CATEGORICAL[1]}
LOAD_TYPE_COLORS = {
    "Light_Load": CATEGORICAL[0],
    "Medium_Load": CATEGORICAL[1],
    "Maximum_Load": CATEGORICAL[2],
}

TEMPLATE_NAME = "steelops"


def register_template() -> None:
    axis = {
        "showgrid": True,
        "gridcolor": GRID,
        "gridwidth": 1,
        "zeroline": False,
        "linecolor": GRID,
        "ticks": "",
        "tickfont": {"color": TEXT_SECONDARY, "size": 11},
        "title": {"font": {"color": TEXT_SECONDARY, "size": 12}},
    }
    pio.templates[TEMPLATE_NAME] = go.layout.Template(
        layout={
            "font": {"family": "Inter, Helvetica, Arial, sans-serif", "color": TEXT_PRIMARY},
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "colorway": CATEGORICAL,
            "title": {"font": {"size": 16}, "x": 0.015, "xanchor": "left", "xref": "container"},
            "xaxis": axis,
            "yaxis": axis,
            "legend": {"orientation": "h", "y": 1.02, "yanchor": "bottom", "x": 0},
            "hovermode": "x unified",
            "margin": {"l": 60, "r": 20, "t": 80, "b": 50},
        }
    )
    pio.templates.default = TEMPLATE_NAME
