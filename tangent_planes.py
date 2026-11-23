"""Interactive level surfaces, tangent planes, and normal lines (Python 3.10+).

Install:
    python -m pip install "numpy>=1.24" "dash>=2.17,<4" "plotly>=5.22,<7"
Run:
    python tangent_planes.py
Open the local link printed in the terminal. Use --port to change the port.
"""

from __future__ import annotations

import argparse
from functools import lru_cache

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate


SURFACE = "#3e8c85"
PLANE = "#425fbb"
NORMAL = "#a84c21"
INK = "#203047"


def graph_height(x, y):
    return (x**2 + 2*y**2) / 4


def geometry(example, level, anchor, azimuth, elevation, x, y, direction):
    """Return the contact point, gradient, and tangent directions."""
    if example == "sphere":
        radius = np.sqrt(level)
        if anchor == "worked":
            p = (radius / 3) * np.array([1., 2., 2.])
            azimuth, elevation = np.arctan2(2, 1), np.arcsin(2 / 3)
        elif anchor == "top":
            p = np.array([0., 0., radius])
            azimuth, elevation = 0., np.pi / 2
        else:
            azimuth, elevation = np.deg2rad([azimuth, elevation])
            horizontal = np.cos(elevation)
            p = radius * np.array([
                horizontal * np.cos(azimuth), horizontal * np.sin(azimuth), np.sin(elevation)
            ])
        gradient = 2 * p
        e1 = np.array([
            np.sin(elevation) * np.cos(azimuth), np.sin(elevation) * np.sin(azimuth),
            -np.cos(elevation),
        ])
    else:
        p = np.array([x, y, graph_height(x, y)])
        gradient = np.array([x / 2, y, -1.])
        e1 = np.array([1., 0., gradient[0]])
        e1 /= np.linalg.norm(e1)

    unit_normal = gradient / np.linalg.norm(gradient)
    e2 = np.cross(unit_normal, e1)
    if example == "graph":
        e2 = -e2  # The second tangent direction points toward increasing y.
    angle = np.deg2rad(direction)
    u = np.cos(angle) * e1 + np.sin(angle) * e2
    return p, gradient, unit_normal, e1, e2, u


def surface_path(example, p, u, s):
    """Sample a curve C on F=c with C(0)=P and C′(0)=u."""
    s = np.asarray(s)[..., None]
    if example == "sphere":
        radius = np.linalg.norm(p)
        return np.cos(s / radius) * p + radius * np.sin(s / radius) * u
    xy = p[:2] + s * u[:2]
    z = graph_height(xy[..., 0], xy[..., 1])
    return np.concatenate((xy, z[..., None]), axis=-1)


@lru_cache(maxsize=40)
def surface_mesh(example, level):
    if example == "sphere":
        azimuth, elevation = np.meshgrid(np.linspace(0, 2*np.pi, 61),
                                         np.linspace(-np.pi / 2, np.pi / 2, 37))
        radius = np.sqrt(level)
        return (
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        )
    x, y = np.meshgrid(np.linspace(-2.5, 2.5, 41), np.linspace(-2.5, 2.5, 41))
    return x, y, graph_height(x, y)


def line(points, color, width, name, dash="solid"):
    x, y, z = np.asarray(points).T
    return go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=color, width=width, dash=dash),
                        name=name, hoverinfo="skip")


def marker(point, color, name, symbol="circle"):
    x, y, z = point
    return go.Scatter3d(
        x=[x], y=[y], z=[z], mode="markers", name=name,
        marker=dict(size=6, color=color, symbol=symbol, line=dict(color="white", width=2)),
        hovertemplate=f"{name}<br>(%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}})<extra></extra>",
    )


def make_figure(example, level, p, unit_normal, e1, e2, u, distance, show_path):
    is_sphere = example == "sphere"
    x, y, z = surface_mesh(example, level if is_sphere else 0)
    fig = go.Figure(go.Surface(
        x=x, y=y, z=z, opacity=.42, showscale=False, name="Level surface",
        colorscale=[[0, SURFACE], [1, SURFACE]], lighting=dict(ambient=.72, diffuse=.78),
        hovertemplate="Surface<br>(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
    ))

    a, b = 1.85 * e1, 1.85 * e2
    corners = np.array([p-a-b, p+a-b, p+a+b, p-a+b])
    x, y, z = corners.T
    fig.add_trace(go.Mesh3d(x=x, y=y, z=z, i=[0, 0], j=[1, 2], k=[2, 3],
                            color=PLANE, opacity=.30, name="Tangent plane", hoverinfo="skip"))
    fig.add_trace(line(np.vstack((corners, corners[0])), PLANE, 3, "Plane boundary"))

    # For a sphere the visible normal line extends through its center at every level.
    inward = np.sqrt(level) + .55 if is_sphere else 2.2
    fig.add_trace(line([p-max(inward, 2.2)*unit_normal, p+2.35*unit_normal],
                       NORMAL, 5, "Normal line", "dash"))
    tip = p + 1.15 * unit_normal
    fig.add_trace(line([p, tip], NORMAL, 6, "Gradient direction"))
    fig.add_trace(go.Cone(
        x=[tip[0]], y=[tip[1]], z=[tip[2]],
        u=[unit_normal[0]], v=[unit_normal[1]], w=[unit_normal[2]],
        anchor="tip", sizemode="absolute", sizeref=.29, showscale=False,
        colorscale=[[0, NORMAL], [1, NORMAL]], hoverinfo="skip", name="Gradient direction",
    ))

    if show_path:
        curve = surface_path(example, p, u, np.linspace(-1, 1, 81))
        fig.add_trace(line(curve, "#24665f", 6, "Curve on the level surface"))

    q, n = p + distance*u, p + distance*unit_normal
    fig.add_trace(line([p-1.7*u, p+1.7*u], PLANE, 5, "Tangent direction u"))
    corner = .28
    fig.add_trace(line([p+corner*u, p+corner*(u+unit_normal), p+corner*unit_normal],
                       INK, 4, "Right angle"))
    if is_sphere:
        fig.add_trace(marker(np.zeros(3), "#637388", "Sphere center"))
    if distance != 0:
        fig.add_trace(marker(q, PLANE, "Q · on tangent plane", "square"))
        fig.add_trace(marker(n, NORMAL, "N · on normal line", "diamond"))
    fig.add_trace(marker(p, INK, "P · contact point"))

    limits = ((-5.5, 5.5),) * 3 if is_sphere else ((-4, 4), (-4, 4), (-2.6, 5.4))
    axes = {
        f"{axis}axis": dict(title=axis, range=bounds, nticks=6, backgroundcolor="#edf2f7",
                            gridcolor="#d2dce7", zerolinecolor="#a6b5c8", showbackground=True)
        for axis, bounds in zip("xyz", limits)
    }
    fig.update_layout(
        paper_bgcolor="#f5f8fb", font=dict(color=INK, family="Arial, sans-serif"),
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, uirevision="geometry",
        scene=dict(**axes, aspectmode="data", camera=dict(
            eye=dict(x=1.45, y=-.73, z=1.24), projection=dict(type="orthographic"),
        )),
    )
    return fig


def number(value):
    return f"{value:.4g}".replace("-", "−") if abs(value) >= 1e-10 else "0"


def coordinates(values):
    return "(" + ", ".join(map(number, values)) + ")"


def linear_expression(coefficients, variables):
    terms = []
    for coefficient, variable in zip(coefficients, variables):
        if number(coefficient) == "0":
            continue
        magnitude = number(abs(coefficient))
        term = variable if variable and magnitude == "1" else magnitude + variable
        if not terms:
            terms.append(("−" if coefficient < 0 else "") + term)
        else:
            terms.append((" − " if coefficient < 0 else " + ") + term)
    return "".join(terms) or "0"


def readout(example, level, p, gradient, unit_normal, u, distance, show_path):
    if example == "sphere":
        normal = p  # Half the gradient gives the simpler line equation from the lesson.
        offset = level
        if np.allclose(p[:2], 0):
            normal, offset = np.array([0., 0., 1.]), p[2]
        plane = f"{linear_expression(normal, 'xyz')} = {number(offset)}"
        surface = f"F(x, y, z) = x² + y² + z² = {number(level)}"
        context = f"Radius √c = {number(np.sqrt(level))}. The normal line passes through the center."
    else:
        normal = gradient
        fx, fy = gradient[:2]
        plane = f"z = {linear_expression([fx, fy, p[2]-fx*p[0]-fy*p[1]], ('x', 'y', ''))}"
        surface = "F(x, y, z) = (x² + 2y²)/4 − z = 0"
        context = "Writing z = f(x, y) as F = f(x, y) − z gives ∇F = (fₓ, fᵧ, −1)."

    return [
        html.Div([
            html.H2("At the contact point"), html.Div(surface, className="formula"),
            html.P(context),
            html.Div(f"P = {coordinates(p)}", className="formula"),
            html.Div(f"∇F(P) = {coordinates(gradient)}", className="formula"),
            html.Div(f"F(P) = {number(level if example == 'sphere' else 0)}", className="formula"),
        ], className="readout"),
        html.Div([
            html.H2("Tangent plane", className="tangent"),
            html.Div(plane, className="formula"),
            html.P("A displacement in the plane is perpendicular to the gradient."),
            html.Div(f"u = {coordinates(u)}", className="formula"),
            html.Div(f"∇F(P) · u = {number(gradient @ u)}", className="formula"),
            html.Div(f"Q = P + su = {coordinates(p + distance*u)}", className="formula"),
        ], className="readout"),
        html.Div([
            html.H2("Normal line", className="normal"),
            html.Div(f"(x, y, z) = {coordinates(p)} + t{coordinates(normal)}", className="formula"),
            html.P("The line follows the gradient. The probe uses the unit normal n̂."),
            html.Div(f"∇F(P) · n̂ = {number(gradient @ unit_normal)}", className="formula"),
            html.Div(f"N = P + sn̂ = {coordinates(p + distance*unit_normal)}", className="formula"),
        ], className="readout"),
        html.Div([
            html.H2("Walking on the surface"),
            html.P(f"The green curve has initial direction u and keeps F = "
                   f"{number(level if example == 'sphere' else 0)}. Its tangent lies in the plane at P. "
                   "The straight probe in the plane follows that direction only to first order."),
        ], className="readout") if show_path else None,
    ]


CSS = """
* { box-sizing: border-box; }
body { margin: 0; color: #203047; background: #f5f8fb;
       font: 15px/1.5 system-ui, sans-serif; }
main { max-width: 1450px; padding: 28px clamp(18px, 4vw, 58px) 34px; margin: auto; }
header { padding-bottom: 19px; border-bottom: 1px solid #d8e2ec; }
h1 { font: 500 clamp(2.1rem, 4.6vw, 3.7rem)/1.08 Georgia, serif;
     letter-spacing: -.035em; margin: 0 0 8px; }
p { margin: 0; color: #495c72; }
header p { max-width: 68ch; }
.workspace { display: grid; grid-template-columns: minmax(0, 1fr) minmax(310px, .43fr);
             gap: 20px; align-items: start; margin-top: 18px; }
.geometry-panel { min-width: 0; overflow: hidden; border: 1px solid #d8e2ec; border-radius: 12px; }
.controls { padding: 18px 20px 6px; background: white; border-bottom: 1px solid #d8e2ec; }
.choices { display: flex; flex-wrap: wrap; gap: 12px 23px; }
fieldset { min-width: 0; border: 0; margin: 0; padding: 0; }
legend { font-weight: 700; padding: 0; margin: 0 0 4px; }
.choices fieldset { margin-bottom: 12px; }
.choices label { display: inline-flex; align-items: center; margin-right: 12px; }
input[type=radio], input[type=checkbox] { accent-color: #425fbb; margin: 0 5px 0 0; }
.sliders { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 0 18px; }
.sliders fieldset { margin-bottom: 7px; }
.sliders legend { font-size: .9rem; }
.sliders .rc-slider-handle { border-color: #425fbb; }
.sliders .rc-slider-track { background: #425fbb; }
.path-toggle { padding-bottom: 7px; }
.path-toggle label { display: inline-flex; align-items: center; }
.plot-frame { height: clamp(370px, 56vh, 590px); }
.legend { display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 9px 16px 12px;
          font-size: .87rem; background: #f5f8fb; }
.legend span { display: inline-flex; gap: 6px; align-items: center; }
.legend i { width: 14px; height: 4px; background: currentColor; border-radius: 3px; }
.legend .normal i { height: 0; border-top: 3px dashed currentColor; background: none; }
.readout { border: 1px solid #d8e2ec; border-radius: 12px; background: white;
           padding: 15px 17px; margin-bottom: 12px; }
.readout h2 { font-size: 1.08rem; margin: 0 0 6px; }
.readout p { margin: 3px 0 7px; font-size: .92rem; }
.formula { font: .91rem/1.6 ui-monospace, monospace; overflow-wrap: anywhere; }
.tangent { color: #425fbb; }
.normal { color: #a84c21; }
@media (max-width: 870px) { .workspace { grid-template-columns: minmax(0, 1fr); } }
@media (max-width: 460px) {
    .plot-frame { height: clamp(330px, 92vw, 420px); }
    .sliders { grid-template-columns: minmax(0, 1fr); }
    .choices { gap: 0; }
}
"""


def slider(label, id_, minimum, maximum, step, value, marks, style=None):
    return html.Fieldset([
        html.Legend(label, id=f"{id_}-label"),
        dcc.Slider(minimum, maximum, step, id=id_, value=value, marks=marks,
                   updatemode="drag", tooltip=dict(placement="bottom")),
    ], id=f"{id_}-control", style=style)


app = Dash(__name__)
app.title = "Tangent Planes and Normal Lines"
app.index_string = app.index_string.replace("</head>", f"<style>{CSS}</style></head>")
HIDDEN = {"display": "none"}
app.layout = html.Main([
    dcc.Store(id="camera"),
    html.Header([
        html.H1("Tangent planes and normal lines"),
        html.P("Choose a level surface and move its contact point P. Drag to rotate the scene, "
               "then use the probe to compare motion in the plane and along the normal line."),
    ]),
    html.Div([
        html.Section([
            html.Div([
                html.Div([
                    html.Fieldset([
                        html.Legend("Surface"),
                        dcc.RadioItems({"sphere": "Sphere", "graph": "Graph z = f(x, y)"},
                                       "sphere", id="example", inline=True),
                    ]),
                    html.Fieldset([
                        html.Legend("Contact point on the sphere"),
                        dcc.RadioItems({"worked": "Worked example", "top": "Top", "custom": "Choose angles"},
                                       "worked", id="anchor", inline=True),
                    ], id="anchor-control"),
                ], className="choices"),
                html.Div([
                    slider("Level c", "level", 1, 9, .25, 9, {1: "1", 5: "5", 9: "9"}),
                    slider("Azimuth", "azimuth", -180, 180, 1, 63,
                           {-180: "−180°", 0: "0°", 180: "180°"}, HIDDEN),
                    slider("Elevation", "elevation", -90, 90, 1, 42,
                           {-90: "−90°", 0: "0°", 90: "90°"}, HIDDEN),
                    slider("Contact x", "contact-x", -1.25, 1.25, .05, 1,
                           {-1.25: "−1.25", 0: "0", 1.25: "1.25"}, HIDDEN),
                    slider("Contact y", "contact-y", -1.25, 1.25, .05, 1,
                           {-1.25: "−1.25", 0: "0", 1.25: "1.25"}, HIDDEN),
                    slider("Tangent direction angle", "direction", 0, 360, 1, 35,
                           {0: "0°", 90: "90°", 180: "180°", 270: "270°", 360: "360°"}),
                    slider("Probe distance s = 1.05", "distance", -1.7, 1.7, .05, 1.05,
                           {-1.7: "−1.7", 0: "0", 1.7: "1.7"}),
                ], className="sliders"),
                dcc.Checklist([{"label": "Show a path on the level surface", "value": "path"}],
                              [], id="path-toggle", className="path-toggle"),
            ], className="controls"),
            html.Div(dcc.Graph(id="geometry", config={"displaylogo": False, "scrollZoom": True},
                               responsive=True, style={"height": "100%"}), className="plot-frame"),
            html.Div([
                html.Span([html.I(style={"color": color}), label], className=class_name)
                for color, label, class_name in (
                    (SURFACE, "Level surface", ""), (PLANE, "Tangent plane · Q", ""),
                    (NORMAL, "Normal line · N", "normal"), (INK, "Contact point · P", ""),
                )
            ], className="legend"),
        ], className="geometry-panel"),
        html.Aside(id="values", **{"aria-live": "polite"}),
    ], className="workspace"),
])


@app.callback(
    Output("anchor-control", "style"), Output("level-control", "style"),
    Output("azimuth-control", "style"), Output("elevation-control", "style"),
    Output("contact-x-control", "style"), Output("contact-y-control", "style"),
    Input("example", "value"), Input("anchor", "value"),
)
def show_controls(example, anchor):
    sphere = {} if example == "sphere" else HIDDEN
    angles = {} if example == "sphere" and anchor == "custom" else HIDDEN
    graph = {} if example == "graph" else HIDDEN
    return sphere, sphere, angles, angles, graph, graph


@app.callback(Output("camera", "data"), Input("geometry", "relayoutData"), prevent_initial_call=True)
def remember_camera(relayout):
    if relayout and "scene.camera" in relayout:
        return relayout["scene.camera"]
    raise PreventUpdate


@app.callback(
    Output("geometry", "figure"), Output("values", "children"), Output("distance-label", "children"),
    Input("example", "value"), Input("level", "value"), Input("anchor", "value"),
    Input("azimuth", "value"), Input("elevation", "value"),
    Input("contact-x", "value"), Input("contact-y", "value"),
    Input("direction", "value"), Input("distance", "value"), Input("path-toggle", "value"),
    State("camera", "data"),
)
def update_scene(example, level, anchor, azimuth, elevation, x, y, direction, distance, path, camera):
    p, gradient, unit_normal, e1, e2, u = geometry(
        example, level, anchor, azimuth, elevation, x, y, direction
    )
    show_path = "path" in path
    figure = make_figure(example, level, p, unit_normal, e1, e2, u, distance, show_path)
    if camera:
        figure.update_scenes(camera=camera)
    values = readout(example, level, p, gradient, unit_normal, u, distance, show_path)
    return figure, values, f"Probe distance s = {number(distance)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8050, help="Local server port (default: 8050)")
    args = parser.parse_args()
    app.run(host="127.0.0.1", port=args.port)
