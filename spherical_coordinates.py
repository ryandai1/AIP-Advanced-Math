"""Interactive spherical-coordinate lesson (desktop Python).

Install: python -m pip install numpy matplotlib
Run:     python spherical_coordinates.py
Drag the 3D scene to rotate it; use sliders and lesson buttons below.
Convention: theta is azimuth; phi is measured DOWN from positive z.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def spherical_to_cartesian(rho, theta_degrees, phi_degrees):
    theta, phi = np.deg2rad([theta_degrees, phi_degrees])
    return np.array([rho*np.sin(phi)*np.cos(theta),
                     rho*np.sin(phi)*np.sin(theta), rho*np.cos(phi)])


class SphericalLesson:
    def __init__(self):
        self.fig = plt.figure(figsize=(13, 9), facecolor='#f5f7fb')
        self.fig.suptitle('Spherical coordinates | two triangles, three coordinates',
                          fontsize=18, fontweight='bold', y=.97)
        self.ax = self.fig.add_axes([.02, .28, .63, .63], projection='3d')
        self.panel = self.fig.add_axes([.68, .45, .30, .44])
        self.panel.axis('off')
        self.readout = self.fig.text(.69, .425, '', va='top', fontsize=11, color='#23304a')
        self.sliders = []
        specs = [(r'$\rho$  distance', 0, 6, 4),
                 (r'$\theta$  azimuth (degrees)', 0, 360, 45),
                 (r'$\phi$  from +z (degrees)', 0, 180, 55)]
        for i, (label, lo, hi, initial) in enumerate(specs):
            widget = Slider(self.fig.add_axes([.27, .20-i*.055, .56, .024]),
                            label, lo, hi, valinit=initial, valfmt='%1.1f',
                            color=['#7356c8', '#e4a33b', '#268cb8'][i])
            widget.on_changed(self.update)
            self.sliders.append(widget)
        self.steps = RadioButtons(self.fig.add_axes([.69, .245, .28, .105]),
                                  ('1  Vertical triangle', '2  Horizontal triangle',
                                   '3  Combine both'), active=2)
        self.steps.on_clicked(self.update)
        self.reset_button = Button(self.fig.add_axes([.87, .15, .10, .045]), 'Reset')
        self.reset_button.on_clicked(self.reset)
        self.fig.text(.04, .025,
                      'Drag the plot to rotate. Sliders move P. Colors: purple = radius, '
                      'blue = vertical triangle, gold = horizontal triangle.', fontsize=10)
        self.ax.view_init(elev=24, azim=-60)
        self.update()

    def reset(self, event=None):
        for slider in self.sliders:
            slider.reset()
        self.ax.view_init(elev=24, azim=-60)
        self.update()

    def update(self, event=None):
        rho, theta_deg, phi_deg = [s.val for s in self.sliders]
        theta, phi = np.deg2rad([theta_deg, phi_deg])
        p = spherical_to_cartesian(rho, theta_deg, phi_deg)
        x, y, z = p
        r = rho*np.sin(phi)
        o, q = np.zeros(3), np.array([x, y, 0])
        foot = np.array([x, 0, 0])
        zpoint = np.array([0, 0, z])
        azim, elev = self.ax.azim, self.ax.elev
        self.ax.clear()
        self.ax.view_init(elev=elev, azim=azim)
        ax = self.ax
        ax.set_facecolor('#f5f7fb')
        lim = max(1.5, rho*1.18)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim),
               xlabel='x', ylabel='y', zlabel='z')
        ax.set_box_aspect((1, 1, 1))
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((.94, .96, .99, .35))
        for endpoint, name in [([lim*.95, 0, 0], '+x'), ([0, lim*.95, 0], '+y'),
                               ([0, 0, lim*.95], '+z')]:
            a = np.array(endpoint)
            ax.plot([0, a[0]], [0, a[1]], [0, a[2]], color='#9aa5b6', lw=1)
            ax.text(*a, name, fontsize=10)

        def line(a, b, color, label=None, dashed=False, width=2.5):
            ax.plot(*np.array([a, b]).T, color=color,
                    ls='--' if dashed else '-', lw=width)
            if label and np.linalg.norm(b-a) > .15:
                ax.text(*((a+b)/2), label, color=color, fontsize=11)

        def triangle(vertices, color):
            ax.add_collection3d(Poly3DCollection([vertices], facecolor=color,
                                                 alpha=.16, edgecolor='none'))

        # Faint sphere marks all positions having the selected radius.
        t = np.linspace(0, 2*np.pi, 25)
        f = np.linspace(0, np.pi, 13)
        ax.plot_wireframe(rho*np.outer(np.cos(t), np.sin(f)),
                          rho*np.outer(np.sin(t), np.sin(f)),
                          rho*np.outer(np.ones_like(t), np.cos(f)),
                          color='#b7bfce', alpha=.20, linewidth=.5)
        line(o, p, '#7356c8', r'$\rho$', width=3.5)
        ax.scatter(*p, color='#7356c8', s=65, depthshade=False)
        ax.text(*(p+np.array([.15, .15, .15])), 'P (x, y, z)', fontweight='bold')
        ax.text(0, 0, 0, 'O')
        show_vertical = self.steps.value_selected != '2  Horizontal triangle'
        show_horizontal = self.steps.value_selected != '1  Vertical triangle'
        if show_vertical:
            triangle([o, q, p], '#268cb8')
            line(o, q, '#268cb8', 'r')
            line(q, p, '#268cb8', 'z')
            line(zpoint, p, '#268cb8', dashed=True, width=1)
            # Phi arc starts at +z, in the vertical plane containing P.
            arc = np.linspace(0, phi, 65)
            rad = max(.35, .23*rho)
            ax.plot(rad*np.sin(arc)*np.cos(theta), rad*np.sin(arc)*np.sin(theta),
                    rad*np.cos(arc), color='#268cb8', lw=2)
            ax.text(rad*np.sin(phi/2)*np.cos(theta),
                    rad*np.sin(phi/2)*np.sin(theta), rad*np.cos(phi/2),
                    r'$\phi$', color='#268cb8', fontsize=15)
        if show_horizontal:
            triangle([o, foot, q], '#e4a33b')
            line(o, q, '#e4a33b', 'r' if not show_vertical else None)
            line(o, foot, '#c88b26', 'x')
            line(foot, q, '#c88b26', 'y')
            line(q, p, '#8b96a8', dashed=True, width=1)
            arc = np.linspace(0, theta, 100)
            rad = max(.4, .25*rho)
            ax.plot(rad*np.cos(arc), rad*np.sin(arc), 0*arc, color='#c88b26', lw=2)
            ax.text(rad*np.cos(theta/2), rad*np.sin(theta/2), .1,
                    r'$\theta$', color='#c88b26', fontsize=15)
        ax.scatter(*q, color='#526376', s=25)
        ax.text(*q, 'Q (x, y, 0)', fontsize=9)
        self.panel.clear()
        self.panel.axis('off')
        texts = [
            ('THE ANGLE CONVENTION', '#23304a', 12),
            ('theta: around the z-axis in the xy-plane\nphi: DOWN from the positive z-axis', '#526376', 11),
            ('1   Vertical triangle', '#268cb8', 13),
            (r'$z=\rho\cos\phi$'+'\n'+r'$r=\rho\sin\phi$', '#268cb8', 19),
            ('2   Horizontal triangle', '#b67e20', 13),
            (r'$x=r\cos\theta$'+'\n'+r'$y=r\sin\theta$', '#b67e20', 19),
            ('3   Substitute r', '#7356c8', 13),
            (r'$x=\rho\sin\phi\cos\theta$'+'\n'+r'$y=\rho\sin\phi\sin\theta$'+'\n'+r'$z=\rho\cos\phi$', '#7356c8', 18),
        ]
        heights = [1, .925, .81, .75, .59, .53, .37, .31]
        for (label, color, size), height in zip(texts, heights):
            self.panel.text(0, height, label, va='top', color=color, fontsize=size)
        self.readout.set_text(
                        f'P = ({x:.3f}, {y:.3f}, {z:.3f})\n'
                        f'Horizontal distance r = {r:.3f}\n'
                        f'Check: x² + y² + z² = {np.dot(p,p):.3f} = rho²')
        if rho < 1e-9:
            ax.set_title('At the origin: angles do not select a unique direction.', fontsize=10)
        elif abs(np.sin(phi)) < 1e-9:
            ax.set_title('On the z-axis: changing theta does not move P.', fontsize=10)
        else:
            ax.set_title('P is on the sphere of radius rho; Q is its xy-projection.', fontsize=10)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', help='Save a PNG instead of opening a GUI (validation).')
    args = parser.parse_args()
    lesson = SphericalLesson()  # Retain widget references while the window is open.
    if args.snapshot:
        lesson.fig.savefig(args.snapshot, dpi=150, facecolor=lesson.fig.get_facecolor())
    else:
        plt.show()


if __name__ == '__main__':
    main()
