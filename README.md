# AIP Advanced Math

Interactive Python visualizations and 2026 midterm mock papers for AIP Advanced Math students. Use the visualizations to explore 3D geometry and multivariable calculus, and the mock papers for revision.

## 2026 Midterm Mock Papers

Two mock papers and their answer sets are available as PDFs. Open or download them directly; Python is only needed for the visualizations.

| Paper | Questions | Answers |
| --- | --- | --- |
| Mock A | [Questions](tests/Advanced_Math_Mock_A_Questions.pdf) | [Answers](tests/Advanced_Math_Mock_A_Answers.pdf) |
| Mock B | [Questions](tests/Advanced_Math_Mock_B_Questions.pdf) | [Answers](tests/Advanced_Math_Mock_B_Answers.pdf) |

The `tests/` folder currently stores these four study PDFs. It does not contain an automated test suite.

## Getting Started with the Visualizations

Use Python 3.10 or newer. Download or clone this repository, open a terminal in its folder, and install the packages used by the visualizations:

```bash
python -m pip install "numpy>=1.24" matplotlib "dash>=2.17,<4" "plotly>=5.22,<7"
```

Choose an application and run its command:

| Application | Run command | Opens in |
| --- | --- | --- |
| [Spherical coordinates](spherical_coordinates.py) | `python spherical_coordinates.py` | A Matplotlib window |
| [Quadric surfaces](quadric_surfaces.py) | `python quadric_surfaces.py` | A Matplotlib window |
| [Multivariable limits](multivariable_limits.py) | `python multivariable_limits.py` | A local website in your browser |
| [Tangent planes and normal lines](tangent_planes.py) | `python tangent_planes.py` | A local website in your browser |

Each application is self-contained in its named `.py` file. Its opening docstring lists the packages needed if you prefer to download and run only that app. The Matplotlib applications require a desktop environment to use their interactive windows.

### Spherical Coordinates

Rotate the 3D scene and adjust radius and angle sliders to connect spherical coordinates with Cartesian coordinates. Lesson controls show the vertical triangle, horizontal triangle, or both. The convention is **theta = azimuth** and **phi = angle measured down from the positive z-axis**.

### Quadric Surfaces

Explore ten surfaces, including planes, cylinders, spheres, ellipsoids, paraboloids, cones, and hyperboloids. Adjust their parameters and choose a slicing axis and position to compare a 3D surface with its corresponding 2D cross-section.

Both Matplotlib applications can save their initial view as a PNG instead of opening a window:

```bash
python spherical_coordinates.py --snapshot spherical.png
python quadric_surfaces.py --snapshot quadrics.png
```

### Multivariable Limits

Open the local link printed in the terminal after starting the server (by default, `http://127.0.0.1:8050`). Stop the server with Ctrl+C. To choose another port:

```bash
python multivariable_limits.py --port 8051
```

The app includes five examples, three linked views, adjustable approach paths, and a guided comparison showing why agreement along every straight line is still inconclusive. It also illustrates excluded domain paths and a squeeze-theorem proof for a limit that exists.

### Tangent Planes and Normal Lines

Rotate a 3D level surface, move the contact point, and compare two probes that move the same signed distance along a tangent direction and the normal line. The app updates the gradient, equations, and directional derivatives as you explore. Start with the sphere `x² + y² + z² = 9` at `P = (1, 2, 2)`, switch to the top of the sphere, or try the graph `z = (x² + 2y²)/4`. Toggle the surface path to see a curve with the same initial tangent direction.

Open the local link printed in the terminal (by default, `http://127.0.0.1:8050`). Use `python tangent_planes.py --port 8051` to run both browser apps at the same time.

## Contributing

Contributions are welcome: add a visualization, improve an explanation or study resource, fix a bug, or suggest a topic through an issue or pull request.

Keep scripts easy to run, label mathematical quantities clearly, and include dependencies and instructions. When updating mock papers, keep question and answer filenames paired and update the links above.

## License

See [LICENSE](LICENSE) for the MIT license.
