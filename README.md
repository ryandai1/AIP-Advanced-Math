# AIP Advanced Math

A collection of Python visualizations for AIP students taking Advanced Math. From vectors and 3D geometry to multivariable calculus, these interactive tools help make abstract concepts easier to see and understand.

This repository is built for current and future students—a shared resource that can grow with each class.

## Getting Started

Download a `.py` file, install its required packages, and run it.

```bash
pip install numpy matplotlib
python spherical_coordinates.py
```

### Multivariable limits lab

Explore five examples with linked 3D surfaces, input paths, and output graphs.
Compare up to three paths, animate approaches from either sign of `t`, and use
the guided curved-path comparison to see why agreeing straight lines are
inconclusive. Exact algebraic limits and a global existence proof are clearly
separated from numerical observations.

```bash
python -m pip install "dash>=2.17,<4" "plotly>=5.22,<7" "numpy>=1.24"
python multivariable_limits.py
```

Open <http://127.0.0.1:8050>. Use `--port 8051` to choose another port.
The complete application is in `multivariable_limits.py`.
Run its mathematical and application checks with
`python -m unittest discover -s tests -v`.

## Contributing

Everyone is welcome to contribute! Add a visualization, improve an explanation, fix a bug, or suggest a topic through an issue or pull request.

Keep scripts easy to run, label mathematical quantities clearly, and include any dependencies and instructions.
