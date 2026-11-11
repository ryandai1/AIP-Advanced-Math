"""Run with: python -m unittest discover -s tests -v."""

import unittest
import warnings

import numpy as np

import multivariable_limits as lab


class LimitsTests(unittest.TestCase):
    def test_required_path_values(self):
        expected = {
            "A": {"xaxis": 1, "yaxis": 1, "diagonal": 0},
            "B": {"xaxis": 0, "yaxis": 0, "diagonal": .25},
            "C": {"xaxis": 0, "yaxis": 0, "cubic": .5},
            "D": {"xaxis": 1/7, "yaxis": -2/3, "diagonal": -3/13},
        }
        for example, paths in expected.items():
            for kind, value in paths.items():
                for sign in (-1, 1):
                    with self.subTest(example=example, path=kind, sign=sign):
                        t = sign*np.geomspace(1, 1e-5, 101)
                        x, y = lab.Path(kind).coordinates(t)
                        np.testing.assert_allclose(lab.evaluate(example, x, y), value, atol=1e-14)

    def test_small_t_and_exact_limit_table(self):
        # All named limits, including nonconstant traces, approach supplied algebraic results.
        for example, entries in lab.EXACT.items():
            for kind, exact in entries.items():
                from fractions import Fraction
                expected = float(Fraction(exact.replace("−", "-")))
                for sign in (-1, 1):
                    x, y = lab.Path(kind).coordinates(sign*1e-5)
                    self.assertAlmostEqual(float(lab.evaluate(example, x, y)), expected, delta=1e-5)
        t = np.array([1., .1, 1e-5])
        x, y = lab.Path("diagonal").coordinates(t)
        np.testing.assert_allclose(lab.evaluate("C", x, y), t*t/(1+t**4))
        for k in (-3, -.5, 0, 1, 3):
            x, y = lab.Path("custom", k, 1).coordinates(1e-5)
            self.assertLess(abs(float(lab.evaluate("C", x, y))), 1e-8)

    def test_existence_bound(self):
        rng = np.random.default_rng(37)
        x, y = rng.uniform(-1, 1, (2, 1000))*np.geomspace(1, 1e-5, 1000)
        self.assertTrue(np.all(np.abs(lab.evaluate("E", x, y)) <= np.abs(y) + 1e-15))

    def test_undefined_points_without_warnings(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for example in lab.PRESETS:
                self.assertTrue(np.isnan(lab.evaluate(example, 0, 0)))
                lab.surface_mesh(example)
            x = np.array([-1., -.001, 0, .001, 1])
            self.assertTrue(np.all(np.isnan(lab.evaluate("D", x, -7*x/6))))
            self.assertIsNotNone(lab.Path("singular").rejection("D"))
            self.assertIsNotNone(lab.Path("custom", -7/6, 1).rejection("D"))
            self.assertIsNone(lab.Path("custom", -7/6, 2).rejection("D"))
            # Nearby points are valid; masking for display does not change evaluation.
            self.assertTrue(np.isfinite(lab.evaluate("D", 1, -7/6 + 1e-6)))

    def test_mesh_gap_and_ridge(self):
        x, y, z = lab.surface_mesh("D")
        self.assertTrue(np.all(np.abs(z[np.isfinite(z)]) <= 3))
        # No four-corner finite surface cell spans both sides of the singular line.
        d = 6*y+7*x
        corners = np.stack([d[:-1, :-1], d[1:, :-1], d[:-1, 1:], d[1:, 1:]])
        finite = np.isfinite(z[:-1, :-1]) & np.isfinite(z[1:, :-1]) & np.isfinite(z[:-1, 1:]) & np.isfinite(z[1:, 1:])
        self.assertFalse(np.any(finite & (corners.min(axis=0) < 0) & (corners.max(axis=0) > 0)))
        x, y, z = lab.surface_mesh("C")
        on_ridge = (x != 0) & (y == x**3)
        self.assertGreater(np.count_nonzero(on_ridge), 170)
        np.testing.assert_allclose(z[on_ridge], .5)

    def test_figures_and_callbacks(self):
        for example in lab.PRESETS:
            paths = [(i, lab.Path(kind)) for i, kind in enumerate(lab.PRESETS[example]["defaults"]) if kind != "none"]
            for negative in (True, False):
                figures = lab.make_figures(example, paths, 5, negative, .5)
                self.assertEqual(figures[2].layout.xaxis.type, "log")
                self.assertEqual(figures[0].layout.scene.uirevision, "keep-camera")
                for figure in figures[:3]:
                    figure.to_json()
        values = lab.update_views("A", 1, [], .5, "custom", None, 1, "none", 1, 1, "none", 1, 1)
        self.assertIn("enter k", str(values[-1]))
        rejected = lab.make_figures("D", [(0, lab.Path("singular"))], 1, False, .5)
        self.assertEqual(len(rejected[0].data), 1)  # Surface only; invalid path never drawn.
        crossing = lab.make_figures("D", [(0, lab.Path("custom", -2, 2))], 1, False, .5)
        self.assertTrue(np.any(np.isnan(crossing[2].data[0].y)))
        client = lab.app.server.test_client()
        for url in ("/", "/_dash-layout", "/_dash-dependencies"):
            self.assertEqual(client.get(url).status_code, 200)


if __name__ == "__main__":
    unittest.main()
