"""Interactive limits of functions of two variables (Python 3.10+).

Install:
    python -m pip install "dash>=2.17,<4" "plotly>=5.22,<7" "numpy>=1.24"
Run:
    python multivariable_limits.py
Then open the local link printed in the terminal. Stop with Ctrl+C.

Use --port to choose a different local port if needed.
The server binds to localhost; no account, external data, or notebook is needed.
All application code, styles, and mathematical examples are in this file.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update


COLORS = ("#126fbd", "#d35426", "#8b42a4")
PRESETS = {
    "A": {
        "title": "A · Agreeing axes, disagreeing diagonal",
        "formula": "f(x,y) = ((x² − y²)/(x² + y²))²",
        "text": "On either axis the squared ratio is 1. On y = x the numerator cancels, giving 0. The axes agree, but the diagonal disproves the overall limit.",
        "exact": "Supplied exact results: y = 0 → 1; x = 0 → 1; y = x → 0. Overall limit: does not exist.",
        "defaults": ("xaxis", "yaxis", "diagonal"), "zrange": (-0.08, 1.08),
    },
    "B": {
        "title": "B · Balance fourth powers",
        "formula": "f(x,y) = x²y²/(x⁴ + 3y⁴)",
        "text": "Either axis makes the numerator zero. Substituting y = x gives t⁴/(4t⁴) = 1/4. Two different path limits disprove the overall limit.",
        "exact": "Supplied exact results: y = 0 → 0; x = 0 → 0; y = x → 1/4. Overall limit: does not exist.",
        "defaults": ("xaxis", "yaxis", "diagonal"), "zrange": (-0.04, 0.34),
    },
    "C": {
        "title": "C · Straight lines miss a curved ridge",
        "formula": "f(x,y) = x³y/(x⁶ + y²)",
        "text": "Every straight line through the origin has limit 0: for y = kx, f = kx²/(x⁴ + k²) → 0 when k ≠ 0; k = 0 and the vertical axis give 0 identically. Match y² = x⁶ to choose y = x³: then f = 1/2. Even agreement on every straight line is inconclusive.",
        "exact": "Supplied exact results: either axis → 0; y = x → 0; y = x³ → 1/2. Overall limit: does not exist.",
        "defaults": ("xaxis", "diagonal", "none"), "zrange": (-0.58, 0.58),
    },
    "D": {
        "title": "D · Respect the domain",
        "formula": "f(x,y) = (x − 4y)/(6y + 7x)",
        "text": "The axes give different constants, 1/7 and −2/3. The entire line 6y + 7x = 0 is outside the domain; it is not an admissible approach path. Values can grow without bound near this line.",
        "exact": "Supplied exact results: y = 0 → 1/7; x = 0 → −2/3; y = x → −3/13. Overall limit: does not exist.",
        "defaults": ("xaxis", "yaxis", "diagonal"), "zrange": (-3.3, 3.3),
    },
    "E": {
        "title": "E · A proof that covers every approach",
        "formula": "f(x,y) = x²y/(x² + y²)",
        "text": "Since 0 ≤ x²/(x² + y²) ≤ 1 on the domain, |x²y/(x² + y²)| ≤ |y| ≤ √(x² + y²) → 0. The squeeze theorem proves the overall limit is 0. The inequality covers every approach; plotted paths only illustrate this proof.",
        "exact": "Supplied exact results: both axes, y = x, and y = x³ → 0. Overall limit: 0, by the displayed inequality.",
        "defaults": ("xaxis", "diagonal", "cubic"), "zrange": (-0.6, 0.6),
    },
}


def evaluate(example, x, y):
    """Evaluate on the true domain; undefined values stay NaN, never zero."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    if example == "A":
        numerator, denominator = x*x - y*y, x*x + y*y
    elif example == "B":
        numerator, denominator = x*x*y*y, x**4 + 3*y**4
    elif example == "C":
        numerator, denominator = x**3*y, x**6 + y*y
    elif example == "D":
        numerator, denominator = x - 4*y, 6*y + 7*x
    elif example == "E":
        numerator, denominator = x*x*y, x*x + y*y
    else:
        raise ValueError(f"Unknown example: {example}")
    valid = denominator != 0
    if example == "D":
        # Floating-point cancellation must not turn the excluded line into a path.
        valid &= np.abs(denominator) > 8*np.finfo(float).eps*(np.abs(6*y) + np.abs(7*x))
    result = np.full(x.shape, np.nan)
    np.divide(numerator, denominator, out=result, where=valid)
    return result**2 if example == "A" else result


@dataclass(frozen=True)
class Path:
    kind: str
    k: float = 1.0
    p: int = 1

    @property
    def label(self):
        return {"xaxis": "y = 0", "yaxis": "x = 0", "diagonal": "y = x",
                "cubic": "y = x³", "singular": "y = −7x/6 (domain test)"}.get(
                    self.kind, f"x = t, y = {self.k:g}t^{self.p}")

    def coordinates(self, t):
        t = np.asarray(t, dtype=float)
        if self.kind == "yaxis":
            return np.zeros_like(t), t
        k, p = {"xaxis": (0, 1), "diagonal": (1, 1), "cubic": (1, 3),
                "singular": (-7/6, 1)}.get(self.kind, (self.k, self.p))
        return t, k*t**p

    def rejection(self, example):
        if example == "D" and (self.kind == "singular" or (
            self.kind == "custom" and self.p == 1 and abs(6*self.k + 7) < 1e-13
        )):
            return "Rejected: 6y + 7x = 0 at every point on this path. It lies entirely outside the domain."
        return None


EXACT = {
    "A": {"xaxis": "1", "yaxis": "1", "diagonal": "0", "cubic": "1"},
    "B": {"xaxis": "0", "yaxis": "0", "diagonal": "1/4", "cubic": "0"},
    "C": {"xaxis": "0", "yaxis": "0", "diagonal": "0", "cubic": "1/2"},
    "D": {"xaxis": "1/7", "yaxis": "−2/3", "diagonal": "−3/13", "cubic": "1/7"},
    "E": {"xaxis": "0", "yaxis": "0", "diagonal": "0", "cubic": "0"},
}


@lru_cache(maxsize=5)
def surface_mesh(example):
    """Column-wise ridge refinement for C; a full singularity gap for D."""
    xs = np.linspace(-1, 1, 181)
    base_y = np.linspace(-3, 3, 181)
    if example == "C":
        # Include ±x³ and neighbors exactly for every x, even near the origin.
        ratios = np.array([-3, -2, -1.5, -1.1, -1, -.9, -.5, 0, .5, .9, 1, 1.1, 1.5, 2, 3])
        ys = np.sort(np.concatenate([
            np.broadcast_to(base_y[:, None], (len(base_y), len(xs))),
            np.clip(ratios[:, None]*xs[None, :]**3, -3, 3),
        ]), axis=0)
        xx = np.broadcast_to(xs, ys.shape)
    else:
        xx, ys = np.meshgrid(xs, base_y)
    zz = evaluate(example, xx, ys)
    if example == "D":
        # Wider than a cell's change in 6y+7x: no cell bridges the singular line.
        zz = np.where(np.abs(6*ys + 7*xx) < .32, np.nan, zz)
        zz = np.where(np.abs(zz) <= 3, zz, np.nan)
    return xx, ys, zz


def make_figures(example, paths, s, negative, opacity):
    """Build linked views; path formulas never interpolate the surface mesh."""
    preset = PRESETS[example]
    zmin, zmax = preset["zrange"]
    floor = zmin - .12*(zmax-zmin)
    xx, yy, zz = surface_mesh(example)
    surface = go.Figure(go.Surface(
        x=xx, y=yy, z=zz, opacity=opacity, colorscale="Greys", showscale=False,
        cmin=zmin, cmax=zmax, connectgaps=False, name="z = f(x,y)",
        hovertemplate="x=%{x:.4g}<br>y=%{y:.4g}<br>f=%{z:.5g}<extra>Surface</extra>",
        lighting=dict(ambient=.8, diffuse=.5),
    ))
    plane, outputs = go.Figure(), go.Figure()
    magnitude = np.geomspace(1, 1e-5, 401)
    sign = -1 if negative else 1
    now = sign*10**(-float(s))
    rows = []
    for slot, path in paths:
        color = COLORS[slot]
        reason = path.rejection(example)
        if reason:
            rows.append(html.Div([html.Strong(f"Path {slot+1} · {path.label}"), html.P(reason)], className="value rejected"))
            continue
        x, y = path.coordinates(sign*magnitude)
        z = evaluate(example, x, y)
        if example == "D":
            # An adjustable curve can cross the excluded line between samples.
            # Break the traces instead of drawing a false segment through a pole.
            denominator = 6*y + 7*x
            crossing = np.signbit(denominator[1:]) != np.signbit(denominator[:-1])
            z = z.copy()
            z[1:][crossing] = np.nan
        nx, ny = path.coordinates(now)
        nz = float(evaluate(example, nx, ny))
        nx, ny = float(nx), float(ny)
        # Omit out-of-window heights, rather than fabricating capped outputs.
        visible_z = np.where((z >= zmin) & (z <= zmax), z, np.nan)
        surface.add_trace(go.Scatter3d(
            x=x, y=y, z=visible_z, mode="lines", line=dict(color=color, width=7),
            name=f"{slot+1}: lifted · {path.label}", connectgaps=False,
            hovertemplate="x=%{x:.5g}<br>y=%{y:.5g}<br>f=%{z:.6g}<extra>Lifted path</extra>",
        ))
        surface.add_trace(go.Scatter3d(
            x=x, y=y, z=np.full_like(x, floor), mode="lines",
            line=dict(color=color, width=3, dash="dash"), name=f"{slot+1}: input projection",
            hovertemplate="x=%{x:.5g}<br>y=%{y:.5g}<extra>Projection (not function height)</extra>",
        ))
        surface.add_trace(go.Scatter3d(
            x=[nx], y=[ny], z=[nz if zmin <= nz <= zmax else None], mode="markers",
            marker=dict(size=6, color=color, line=dict(color="white", width=2)),
            showlegend=False, name="Moving surface point",
        ))
        surface.add_trace(go.Scatter3d(
            x=[nx], y=[ny], z=[floor], mode="markers",
            marker=dict(size=5, symbol="diamond", color=color), showlegend=False,
            name="Moving input projection",
        ))
        plane.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=3),
                                   name=f"{slot+1}: {path.label}"))
        plane.add_trace(go.Scatter(x=[nx], y=[ny], mode="markers", marker=dict(size=11, color=color), showlegend=False))
        outputs.add_trace(go.Scatter(x=magnitude, y=z, mode="lines", line=dict(color=color, width=3, dash=("solid", "dash", "dot")[slot]),
                                     name=f"{slot+1}: {path.label}", connectgaps=False))
        outputs.add_trace(go.Scatter(x=[abs(now)], y=[nz], mode="markers", marker=dict(size=11, color=color), showlegend=False))
        exact = EXACT[example].get(path.kind)
        evidence = f"Exact path limit (algebra): {exact}" if exact else "Adjustable path: numerical observations only; samples are not a proof."
        if example == "E" and not exact:
            evidence += " The preset’s global inequality proves 0 for this path too."
        display = f"t = {now:.6g} · x = {nx:.6g} · y = {ny:.6g} · f = {nz:.8g}" if np.isfinite(nz) else f"t = {now:.6g} · x = {nx:.6g} · y = {ny:.6g} · f undefined"
        rows.append(html.Div([
            html.Strong(f"Path {slot+1} · {path.label}", style={"color": color}),
            html.Div(display, className="numbers"), html.Small(evidence),
            html.Small("Height is outside the displayed 3D range; read its actual value here and in the output graph.")
            if np.isfinite(nz) and not zmin <= nz <= zmax else None,
        ], className="value"))
    plane.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(size=12, color="#172d39", symbol="x"), name="Target (0,0)"))
    if example == "D":
        plane.add_trace(go.Scatter(x=[-1, 1], y=[7/6, -7/6], mode="lines", line=dict(color="#555", dash="dot"), name="Excluded: 6y + 7x = 0"))
    for fig in (surface, plane, outputs):
        fig.update_layout(template="plotly_white", font=dict(family="Arial, sans-serif", size=12),
                          margin=dict(l=45, r=20, t=25, b=45),
                          legend=dict(orientation="h", y=-.14, x=0), uirevision="limits-stable")
    surface.update_layout(
        height=580, margin=dict(l=0, r=0, t=10, b=80),
        scene=dict(xaxis=dict(title="x (input)", range=[-1.05, 1.05]),
                   yaxis=dict(title="y (input)", range=[-3.15, 3.15]),
                   zaxis=dict(title="z = f(x,y)", range=[floor-.03, zmax]),
                   aspectmode="cube", uirevision="keep-camera",
                   camera=dict(eye=dict(x=1.6, y=1.7, z=1.2))),
    )
    plane.update_layout(height=350, xaxis=dict(title="x", range=[-1.1, 1.1]),
                        yaxis=dict(title="y", range=[-3.2, 3.2]), dragmode="pan")
    outputs.update_layout(height=350, xaxis=dict(type="log", range=[0, -5], title="|t| · logarithmic scale · approach →"),
                          yaxis=dict(title="f(x(t), y(t))"))
    outputs.add_vline(x=abs(now), line_dash="dot", line_color="#777")
    return surface, plane, outputs, rows or [html.P("Choose a path in any slot to begin.")]


app = Dash(__name__, title="Multivariable limits lab")
app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:#18313f;font-family:Arial,sans-serif;line-height:1.5}
main{max-width:1500px;margin:auto;padding:24px}h1{font-size:32px;margin:0}h2{font-size:21px;margin:0 0 12px}h3{font-size:17px;margin:0 0 8px}
p{margin:8px 0 14px}.eyebrow{letter-spacing:2px;text-transform:uppercase;color:#557180;font-size:12px;font-weight:bold}
.card{background:white;border:1px solid #dbe3e8;border-radius:12px;padding:20px;margin:16px 0}
.rule{background:#173c50;color:white;padding:16px 20px;border-radius:10px;font-weight:bold;margin-top:18px}
.formula{font-family:Georgia,serif;font-size:24px;padding:12px 0}.exact{border-left:4px solid #277760;background:#eef8f3;padding:12px}
.slots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.slot{border-top:4px solid;padding-top:12px}
.parameters{display:flex;gap:10px;align-items:center;margin-top:12px}.parameters>div{flex:1;min-width:0}
input[type=number]{width:100%;padding:9px;border:1px solid #bbcbd4;border-radius:4px}label{display:block;font-weight:bold;font-size:14px;margin-bottom:6px}
.transport{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:12px 0 20px}button{background:#eaf1f5;border:1px solid #adbfca;color:#173c50;border-radius:6px;padding:10px 18px;font-weight:bold;cursor:pointer}button:hover{background:#dceaf2}
.views{display:grid;grid-template-columns:minmax(0,1.4fr) minmax(0,1fr);gap:18px}.views>.card{min-width:0;margin:0}.side>div+div{margin-top:16px}
.value{padding:12px 0;border-bottom:1px solid #e5ebee}.numbers{font-family:monospace;font-size:14px;margin:6px 0;overflow-wrap:anywhere}small{display:block;color:#506773}.rejected{background:#fff1e9;padding:12px}
.notes{color:#506773;font-size:13px}.guided{background:#f4f0fb;padding:14px;border-radius:8px;margin-top:12px}.guided button{margin:4px 8px 4px 0}
@media(max-width:950px){.views{grid-template-columns:1fr}.slots{grid-template-columns:1fr}main{padding:12px}h1{font-size:26px}.card{padding:14px}}
</style></head><body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>"""

OPTIONS = [
    {"label": "No path", "value": "none"}, {"label": "Horizontal axis · y = 0", "value": "xaxis"},
    {"label": "Vertical axis · x = 0", "value": "yaxis"}, {"label": "Diagonal · y = x", "value": "diagonal"},
    {"label": "Curved ridge · y = x³", "value": "cubic"},
    {"label": "Adjustable · x = t, y = ktᵖ", "value": "custom"},
    {"label": "Domain test · y = −7x/6", "value": "singular"},
]


def slot_controls(i):
    return html.Div([
        html.Label(f"Path {i+1}", htmlFor=f"path-{i}"),
        dcc.Dropdown(id=f"path-{i}", options=OPTIONS, value=PRESETS["A"]["defaults"][i], clearable=False),
        html.Div([
            html.Div([html.Label("k (−3 to 3)", htmlFor=f"k-{i}"),
                      dcc.Input(id=f"k-{i}", type="number", value=1, min=-3, max=3, step="any", debounce=True)]),
            html.Div([html.Label("p", htmlFor=f"p-{i}"),
                      dcc.Dropdown(id=f"p-{i}", options=[1, 2, 3, 4], value=1, clearable=False)]),
        ], className="parameters"),
        html.Small("k and p apply only to the adjustable family."),
    ], className="slot", style={"borderColor": COLORS[i]})


app.layout = html.Main([
    html.Div("An interactive calculus lab", className="eyebrow"),
    html.H1("One target. Many ways to approach."),
    html.P("Explore limits of functions of two variables at (0,0). The surface is z = f(x,y): x and y are the two independent inputs; z is the output height, not a third independent variable."),
    html.Div("Two different path limits prove nonexistence. Agreement along selected paths is inconclusive.", className="rule"),
    html.Section([
        html.Label("Choose an example", htmlFor="preset"),
        dcc.Dropdown(id="preset", options=[{"label": v["title"], "value": k} for k, v in PRESETS.items()], value="A", clearable=False),
        html.Div(id="lesson"),
        html.Div([
            html.Strong("Guided comparison · example C"),
            html.P("Start with two agreeing straight paths. Predict what happens when denominator powers balance, then reveal the curved path."),
            html.Button("1 · Start with straight paths", id="guide-start", n_clicks=0),
            html.Button("2 · Reveal y = x³", id="guide-reveal", n_clicks=0),
        ], id="guide", className="guided", style={"display": "none"}),
    ], className="card"),
    html.Section([
        html.H2("Choose up to three paths"),
        html.Div([slot_controls(i) for i in range(3)], className="slots"),
        dcc.Checklist(id="negative", options=[{"label": " Approach with negative t (graph uses |t|)", "value": "negative"}], value=[], style={"marginTop": "18px"}),
        html.Div(id="input-warning", role="status"),
    ], className="card"),
    html.Section([
        html.H2("Approach the origin"),
        html.Div([html.Button("Play", id="play", n_clicks=0), html.Button("Reset", id="reset", n_clicks=0), html.Strong(id="time-label")], className="transport"),
        html.Label("s controls t = ±10⁻ˢ; the origin is never evaluated", htmlFor="s"),
        dcc.Slider(id="s", min=0, max=5, step=.025, value=0, marks={i: f"10⁻{i}" if i else "1" for i in range(6)}, updatemode="drag"),
        html.Label("Surface opacity", htmlFor="opacity", style={"marginTop": "22px"}),
        dcc.Slider(id="opacity", min=.1, max=1, step=.05, value=.55, marks={.1: "0.1", .55: "0.55", 1: "1"}, updatemode="drag"),
        dcc.Interval(id="clock", interval=180, disabled=True, n_intervals=0),
    ], className="card"),
    html.Div([
        html.Section([
            html.H2("Surface and lifted paths"),
            html.P("Drag to rotate · scroll to zoom. Solid colored curves lie on z = f(x,y). Dashed curves and diamond markers are input projections on a reference plane below the surface; that plane is not z = 0.", className="notes"),
            dcc.Graph(id="surface", config={"displaylogo": False, "scrollZoom": True}),
            html.Div(id="display-note", className="notes"),
        ], className="card"),
        html.Section([
            html.Div([html.H2("Inputs · top-down xy view"), dcc.Graph(id="plane", config={"displaylogo": False}), html.Small("Both axes are input coordinates. Their displayed scales differ; read the tick labels.")]),
            html.Div([html.H2("Outputs · along each path"), dcc.Graph(id="outputs", config={"displaylogo": False}), html.Small("Moving right approaches |t| = 10⁻⁵, never zero. Curves show samples, not proofs; this graph does not clip output values.")]),
        ], className="card side"),
    ], className="views"),
    html.Section([html.H2("Live coordinates and evidence"), html.Div(id="values", **{"aria-live": "off"})], className="card"),
    html.Section([
        html.H2("Why choose this path?"),
        html.Ol([
            html.Li("Set a variable to zero to simplify: axes often make the numerator vanish or leave a constant ratio."),
            html.Li("Try y = x to expose cancellation or balance terms of equal degree (examples A and B)."),
            html.Li("Match denominator powers when straight lines are inconclusive: x⁶ + y² suggests y² = x⁶, hence y = ±x³ (example C)."),
        ]),
        html.P("A path must stay in the function’s domain arbitrarily close to the target. A missing value at the target is allowed; a path with no domain points is not. Each plotted path uses t ≠ 0. Try both signs to explore approaches from opposite sides."),
        html.P("Agreement among any finite selection is inconclusive. Establish existence with a bound, continuity theorem, or another argument covering every approach. Exact path limits above come from algebra, not from the final sampled value.", className="exact"),
    ], className="card"),
])


@app.callback(
    [Output(f"path-{i}", "value") for i in range(3)],
    Input("preset", "value"), Input("guide-start", "n_clicks"), Input("guide-reveal", "n_clicks"),
)
def choose_paths(example, _start, _reveal):
    if example == "C" and ctx.triggered_id == "guide-reveal":
        return "xaxis", "diagonal", "cubic"
    return PRESETS[example]["defaults"]


@app.callback(Output("lesson", "children"), Output("guide", "style"), Output("display-note", "children"), Input("preset", "value"))
def lesson(example):
    p = PRESETS[example]
    note = "Surface window: −1 ≤ x ≤ 1, −3 ≤ y ≤ 3. Undefined points are gaps; no value is assigned at the origin. Input projections sit below the displayed output range."
    if example == "C":
        note += " Adaptive columns include y = ±x³ and nearby samples to resolve the narrow ridge. Lifted paths are evaluated independently of this mesh."
    if example == "D":
        note += " Display only: a band |6y + 7x| < 0.32 is masked so the mesh cannot bridge the singular line; surface heights outside [−3,3] are omitted, not capped. The true excluded set is only 6y + 7x = 0. The output graph has no height clipping and breaks at detected singular crossings; readouts show actual values."
    return [html.Div(p["formula"], className="formula"), html.P(p["text"]), html.Div(p["exact"], className="exact")], {"display": "block" if example == "C" else "none"}, note


@app.callback(
    Output("s", "value"), Output("clock", "disabled"), Output("play", "children"),
    Input("play", "n_clicks"), Input("reset", "n_clicks"), Input("clock", "n_intervals"), Input("preset", "value"),
    State("s", "value"), State("clock", "disabled"), prevent_initial_call=True,
)
def transport(_play, _reset, _ticks, _preset, s, disabled):
    trigger = ctx.triggered_id
    if trigger in ("reset", "preset"):
        return 0, True, "Play"
    if trigger == "play":
        if disabled:
            return (0 if s >= 5 else no_update), False, "Pause"
        return no_update, True, "Play"
    if trigger == "clock" and not disabled:
        value = min(5, round(s + .05, 5))
        return value, value >= 5, "Play" if value >= 5 else "Pause"
    return no_update, no_update, no_update


@app.callback(
    Output("surface", "figure"), Output("plane", "figure"), Output("outputs", "figure"),
    Output("values", "children"), Output("time-label", "children"), Output("input-warning", "children"),
    Input("preset", "value"), Input("s", "value"), Input("negative", "value"), Input("opacity", "value"),
    *[Input(f"{field}-{i}", "value") for i in range(3) for field in ("path", "k", "p")],
)
def update_views(example, s, negative, opacity, *settings):
    paths, warnings = [], []
    for i in range(3):
        kind, k, p = settings[3*i:3*i+3]
        if kind == "none":
            continue
        if kind == "custom" and (k is None or not np.isfinite(k) or not -3 <= k <= 3 or p not in (1, 2, 3, 4)):
            warnings.append(f"Path {i+1}: enter k between −3 and 3 and an integer p from 1 to 4. This slot is omitted until corrected.")
            continue
        paths.append((i, Path(kind, float(k) if k is not None else 1, int(p) if p is not None else 1)))
    figures = make_figures(example, paths, s, bool(negative), opacity)
    sign = "−" if negative else "+"
    return (*figures, f"s = {s:.3f} · t = {sign}{10**(-s):.6g}", html.P(" ".join(warnings), className="rejected") if warnings else "")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port, debug=False)
