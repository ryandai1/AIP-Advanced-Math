# AIP Advanced Math

A collection of Python visualizations for AIP students taking Advanced Math. From vectors and 3D geometry to multivariable calculus, these interactive tools help make abstract concepts easier to see and understand.

This repository is built for current and future students—a shared resource that can grow with each class.

## Getting Started

Use Python 3.10 or newer. Download or clone this repository, open a terminal in
its folder, and install the packages used by the visualizations:

```bash
python -m pip install "numpy>=1.24" matplotlib "dash>=2.17,<4" "plotly>=5.22,<7"
```

Choose an application and run its command:

| Application | Run command | Opens in |
| --- | --- | --- |
| Spherical coordinates | `python spherical_coordinates.py` | A Matplotlib window |
| Quadric surfaces | `python quadric_surfaces.py` | A Matplotlib window |
| Multivariable limits | `python multivariable_limits.py` | Your browser at <http://127.0.0.1:8050> |

For the limits app, open the browser address after starting the server. Stop it
with Ctrl+C; use `python multivariable_limits.py --port 8051` if port 8050 is busy.
It includes five examples, three linked views, adjustable approach paths, and a
guided comparison showing why agreeing straight paths are inconclusive.

Each application is self-contained in its named `.py` file. Its opening docstring
lists the packages needed if you prefer to download and run only that app.

## Optional Checks

The `tests/` folder contains automated checks for the multivariable limits app;
it is not another application and is not needed to launch the lab.

```bash
python -m unittest discover -s tests -v
```

## Contributing

Everyone is welcome to contribute! Add a visualization, improve an explanation, fix a bug, or suggest a topic through an issue or pull request.

Keep scripts easy to run, label mathematical quantities clearly, and include any dependencies and instructions.
