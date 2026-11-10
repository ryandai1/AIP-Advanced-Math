"""AIP Advanced Math: interactive surfaces and exact cross-sections.

Install: python -m pip install numpy matplotlib
Run: python quadric_surfaces.py
Choose a surface, a slicing axis, and move k. Drag the 3D view to rotate.
The pink curve is also shown in true 2D on the right. Surfaces extend beyond
this finite viewing box. a,b,c are positive; the plane uses d=1.
Use --snapshot FILE.png to save the initial view without opening a window.
"""
import argparse
import textwrap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

NAMES = ['Plane', 'Elliptic cylinder', 'Parabolic cylinder', 'Sphere',
         'Ellipsoid', 'Elliptic paraboloid', 'Elliptic cone',
         'Hyperboloid: one sheet', 'Hyperboloid: two sheets',
         'Hyperbolic paraboloid']
FORMULAS = [r'$ax+by+cz=1$', r'$x^2/a^2+y^2/b^2=1$', r'$y=ax^2$',
            r'$x^2+y^2+z^2=a^2$', r'$x^2/a^2+y^2/b^2+z^2/c^2=1$',
            r'$x^2/a^2+y^2/b^2=z/c$', r'$x^2/a^2+y^2/b^2=z^2/c^2$',
            r'$x^2/a^2+y^2/b^2-z^2/c^2=1$',
            r'$x^2/a^2+y^2/b^2-z^2/c^2=-1$',
            r'$y^2/b^2-x^2/a^2=z/c$']
HINTS = ['A normal vector is (a, b, c).',
         'No z term: the ellipse extends along the z-axis.',
         'No z term: the parabola extends along the z-axis.',
         'Every point is distance a from the origin.',
         'Three positive squares: a bounded, stretched sphere.',
         'One unsquared variable: a bowl opening toward +z.',
         'Homogeneous squares: two cones meeting at the origin.',
         'Two + squares and one - square, with RHS +1: one connected sheet.',
         'Two + squares and one - square, with RHS -1: two separate sheets.',
         'Opposite signs of the squared variables produce a saddle.']
LIM = 4.0


def coefficients(index, a, b, c):
    """Return q,l,d for F(X)=sum(q*X**2)+l dot X+d=0."""
    q = np.zeros(3)
    l = np.zeros(3)
    d = 0.
    if index == 0: l[:], d = [a,b,c], -1.
    elif index == 1: q[:], d = [1/a**2,1/b**2,0], -1.
    elif index == 2: q[0], l[1] = a, -1.
    elif index == 3: q[:], d = [1,1,1], -a*a
    elif index == 4: q[:], d = [1/a**2,1/b**2,1/c**2], -1.
    elif index == 5: q[:], l[2] = [1/a**2,1/b**2,0], -1/c
    elif index == 6: q[:] = [1/a**2,1/b**2,-1/c**2]
    elif index == 7: q[:], d = [1/a**2,1/b**2,-1/c**2], -1.
    elif index == 8: q[:], d = [1/a**2,1/b**2,-1/c**2], 1.
    elif index == 9: q[:], l[2] = [-1/a**2,1/b**2,0], -1/c
    return q,l,d


def slice_data(index, a,b,c, axis,k):
    q,l,d = coefficients(index,a,b,c)
    remaining = [j for j in range(3) if j != axis]
    i,j = remaining
    return remaining, (q[i],q[j],l[i],l[j],d+q[axis]*k*k+l[axis]*k)


def conic_paths(A,B,D,E,F):
    """Classify and parameterize the slice, including degenerate cases.

    Equation is A*u²+B*v²+D*u+E*v+F=0. Returned arrays are 2-by-N.
    Geometry is exact up to floating-point tolerance; display is sampled.
    """
    tol = 1e-9
    t = np.linspace(-LIM,LIM,700)
    paths = []
    if abs(A)>tol and abs(B)>tol:
        h,j = -D/(2*A), -E/(2*B)
        K = -F+D*D/(4*A)+E*E/(4*B)
        if A*B>0:
            if abs(K)<tol: return 'Point', [np.array([[h],[j]])]
            if K/A<0: return 'Empty', []
            angle = np.linspace(0,2*np.pi,600)
            ru,rv = np.sqrt(K/A),np.sqrt(K/B)
            kind = 'Circle' if np.isclose(ru,rv) else 'Ellipse'
            return kind,[np.array([h+ru*np.cos(angle),j+rv*np.sin(angle)])]
        if abs(K)<tol:
            return 'Two intersecting lines', [np.array([t+h,j+s*np.sqrt(-A/B)*t]) for s in [-1,1]]
        w = np.linspace(-5,5,1100)
        if K/A>0:
            paths = [np.array([h+s*np.sqrt(K/A)*np.cosh(w),j+np.sqrt(-K/B)*np.sinh(w)]) for s in [-1,1]]
        else:
            paths = [np.array([h+np.sqrt(-K/A)*np.sinh(w),j+s*np.sqrt(K/B)*np.cosh(w)]) for s in [-1,1]]
        return 'Hyperbola',paths
    if abs(A)>tol:
        if abs(E)>tol:
            return 'Parabola',[np.array([t,-(A*t*t+D*t+F)/E])]
        delta = D*D-4*A*F
        if delta < -tol: return 'Empty',[]
        roots = [-D/(2*A)] if abs(delta)<tol else [(-D+s*np.sqrt(delta))/(2*A) for s in [-1,1]]
        return ('One line' if len(roots)==1 else 'Two parallel lines'), [np.array([np.full_like(t,r),t]) for r in roots]
    if abs(B)>tol:
        kind, swapped = conic_paths(B,A,E,D,F)
        return kind,[p[::-1] for p in swapped]
    if abs(E)>tol: return 'Line',[np.array([t,-(D*t+F)/E])]
    if abs(D)>tol: return 'Line',[np.array([np.full_like(t,-F/D),t])]
    return ('Entire slice plane' if abs(F)<tol else 'Empty'),[]


def surface_meshes(index,a,b,c):
    """Parameterizations satisfying the displayed equations."""
    t = np.linspace(0,2*np.pi,75)
    z = np.linspace(-LIM,LIM,65)
    if index in (0,5,9):
        X,Y = np.meshgrid(np.linspace(-LIM,LIM,75),np.linspace(-LIM,LIM,75))
        if index==0: Z=(1-a*X-b*Y)/c
        elif index==5: Z=c*(X*X/a**2+Y*Y/b**2)
        else: Z=c*(Y*Y/b**2-X*X/a**2)
        return [(X,Y,Z)]
    if index==2:
        X,Z=np.meshgrid(np.linspace(-np.sqrt(LIM/a),np.sqrt(LIM/a),75),z)
        return [(X,a*X*X,Z)]
    if index in (3,4):
        T,P=np.meshgrid(t,np.linspace(0,np.pi,55))
        aa,bb,cc=(a,a,a) if index==3 else (a,b,c)
        return [(aa*np.sin(P)*np.cos(T),bb*np.sin(P)*np.sin(T),cc*np.cos(P))]
    if index in (1,6,7):
        T,Z=np.meshgrid(t,z)
        scale = np.ones_like(Z) if index==1 else (np.abs(Z)/c if index==6 else np.sqrt(1+Z*Z/c**2))
        return [(a*scale*np.cos(T),b*scale*np.sin(T),Z)]
    T,V=np.meshgrid(t,np.linspace(0,np.arccosh(LIM/c),65))
    return [(a*np.sinh(V)*np.cos(T),b*np.sinh(V)*np.sin(T),s*c*np.cosh(V)) for s in [-1,1]]


def clipped_path(path):
    p = path.copy()
    outside = np.any(np.abs(p)>LIM+1e-8,axis=0)
    p[:,outside] = np.nan
    return p


class QuadricExplorer:
    def __init__(self):
        self.fig=plt.figure(figsize=(15,9),facecolor='#f5f7fb')
        self.fig.suptitle('QUADRIC SURFACES | shape + cross-section',fontsize=19,fontweight='bold',y=.975)
        self.ax=self.fig.add_axes([.235,.28,.46,.59],projection='3d')
        self.section=self.fig.add_axes([.745,.34,.225,.43])
        self.selector=RadioButtons(self.fig.add_axes([.015,.375,.22,.51]),NAMES,active=4)
        for label in self.selector.labels: label.set_fontsize(10)
        self.axis_selector=RadioButtons(self.fig.add_axes([.03,.235,.17,.115]),('x = k','y = k','z = k'),active=2)
        self.title=self.fig.text(.26,.90,'',fontsize=18,color='#284575')
        self.caption=self.fig.text(.26,.255,'',fontsize=11,color='#34445e')
        self.slice_title=self.fig.text(.745,.81,'',fontsize=13,color='#c13d70',fontweight='bold')
        self.slice_equation=self.fig.text(.735,.285,'',fontsize=10,color='#34445e')
        self.note=self.fig.text(.745,.22,'',fontsize=10,color='#34445e')
        self.sliders=[]
        for i,(label,lo,hi,initial) in enumerate([('a',.5,2.5,1.4),('b',.5,2.5,1.),('c',.5,2.5,1.2),('Slice k',-LIM,LIM,.5)]):
            slider=Slider(self.fig.add_axes([.30,.17-i*.035,.53,.018]),label,lo,hi,valinit=initial,valfmt='%.2f',color='#4c81bc' if i<3 else '#cf4f83')
            slider.on_changed(self.update)
            self.sliders.append(slider)
        self.reset_button=Button(self.fig.add_axes([.89,.10,.08,.045]),'Reset')
        self.reset_button.on_clicked(self.reset)
        self.zero_button=Button(self.fig.add_axes([.89,.05,.08,.04]),'k = 0')
        self.zero_button.on_clicked(lambda event:self.sliders[3].set_val(0))
        self.fig.text(.03,.018,'Drag 3D view to rotate | blue: surface | pink: exact slice | gray: slicing plane | view clipped to -4 through 4',fontsize=10,color='#526376')
        self.selector.on_clicked(self.update)
        self.axis_selector.on_clicked(self.update)
        self.ax.view_init(elev=24,azim=-55)
        self.update()

    def reset(self,event=None):
        for slider in self.sliders: slider.reset()
        self.ax.view_init(elev=24,azim=-55)
        self.update()

    def update(self,event=None):
        index=NAMES.index(self.selector.value_selected)
        axis=('x = k','y = k','z = k').index(self.axis_selector.value_selected)
        a,b,c,k=[s.val for s in self.sliders]
        elev,azim=self.ax.elev,self.ax.azim
        self.ax.clear()
        self.ax.view_init(elev=elev,azim=azim)
        ax=self.ax
        ax.set_facecolor('#f5f7fb')
        ax.set(xlim=(-LIM,LIM),ylim=(-LIM,LIM),zlim=(-LIM,LIM),xlabel='x',ylabel='y',zlabel='z')
        ax.set_box_aspect((1,1,1))
        for mesh in surface_meshes(index,a,b,c):
            X,Y,Z=[v.copy() for v in mesh]
            mask=(np.abs(X)>LIM)|(np.abs(Y)>LIM)|(np.abs(Z)>LIM)
            X[mask]=np.nan;Y[mask]=np.nan;Z[mask]=np.nan
            ax.plot_surface(X,Y,Z,color='#669dc6',alpha=.42,edgecolor='none',rstride=2,cstride=2)
        remaining,co= slice_data(index,a,b,c,axis,k)
        kind,paths=conic_paths(*co)
        U,V=np.meshgrid([-LIM,LIM],[-LIM,LIM])
        plane=[None,None,None]
        plane[axis]=np.full_like(U,k)
        plane[remaining[0]]=U;plane[remaining[1]]=V
        ax.plot_surface(*plane,color='#aab0ba',alpha=.12,shade=False)
        self.section.clear()
        self.section.set_facecolor('#fffafd')
        self.section.set(xlim=(-LIM,LIM),ylim=(-LIM,LIM),xlabel='xyz'[remaining[0]],ylabel='xyz'[remaining[1]])
        self.section.set_aspect('equal',adjustable='box')
        self.section.grid(alpha=.2)
        self.section.axhline(0,color='#a1aaba',lw=.7)
        self.section.axvline(0,color='#a1aaba',lw=.7)
        for p in paths:
            p=clipped_path(p)
            points=np.zeros((3,p.shape[1]));points[axis]=k
            points[remaining]=p
            if p.shape[1]==1:
                ax.scatter(*points,color='#c13d70',s=50,depthshade=False)
                self.section.scatter(*p,color='#c13d70',s=40,zorder=5)
            else:
                ax.plot(*points,color='#c13d70',lw=3,zorder=10)
                self.section.plot(*p,color='#c13d70',lw=2.5)
        if kind=='Empty': self.section.text(.5,.5,'No intersection',ha='center',transform=self.section.transAxes,color='#c13d70')
        if paths and not any(np.isfinite(clipped_path(p)).all(axis=0).any() for p in paths):
            self.section.text(.5,.5,'Slice exists outside\nthe viewing box',ha='center',transform=self.section.transAxes,color='#c13d70')
        self.title.set_text(NAMES[index]+'    '+FORMULAS[index])
        self.caption.set_text(textwrap.fill(HINTS[index], width=66))
        self.slice_title.set_text(f'{"xyz"[axis]} = {k:.2f}  |  {kind}')
        A,B,D,E,F=co
        terms=[]
        for value,term in [(A,'xyz'[remaining[0]]+'²'),(B,'xyz'[remaining[1]]+'²'),(D,'xyz'[remaining[0]]),(E,'xyz'[remaining[1]])]:
            if abs(value)>1e-9: terms.append(f'{value:+.3g}{term}')
        equation=' '.join(terms).lstrip('+')+f' = {-F:.3g}'
        self.slice_equation.set_text('Substitute the fixed coordinate:\n'+equation)
        note='a, b, c control the scale.'
        if index==0: note='a, b, c are normal components; d = 1.'
        elif index==1: note='Only a and b are used.'
        elif index==2: note='Only a is used: y = a x².'
        elif index==3: note='Only a is used: radius = a.'
        self.note.set_text(note+'\nTry k = 0 for a central cross-section.')
        self.fig.canvas.draw_idle()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',metavar='FILE')
    args=parser.parse_args()
    explorer=QuadricExplorer()
    if args.snapshot: explorer.fig.savefig(args.snapshot,dpi=140,facecolor=explorer.fig.get_facecolor())
    else: plt.show()

if __name__=='__main__': main()
