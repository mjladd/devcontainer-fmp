# %% [markdown]
# # Complex Numbers
#
# In this notebook, we review some properties of complex numbers. In particular,
# we need complex numbers in view of a complex-valued formulation of the Fourier
# transform, which significantly simplifies the proof and the understanding of
# certain algebraic properties of this transform.

# %% [markdown]
# ## Basic Definitions
#
# We can write a complex number c = a + ib with real part Re(c) = a,
# imaginary part Im(c) = b, and imaginary unit i = sqrt(-1).
# In Python, the symbol `j` is used to denote the imaginary unit.

# %%
a = 1.5
b = 0.8
c = a + b*1j
print(c)
c2 = complex(a, b)
print(c2)

# %%
import numpy as np

print(np.real(c))
print(np.imag(c))

# %% [markdown]
# A complex number c = a+ib can be plotted as a point (a,b) in the Cartesian
# coordinate system. This point is often visualized by an arrow starting at
# (0,0) and ending at (a,b).

# %%
from matplotlib import pyplot as plt
%matplotlib inline


def generate_figure(figsize=(2, 2), xlim=[0, 1], ylim=[0, 1]):
    """Generate figure for plotting complex numbers"""
    plt.figure(figsize=figsize)
    plt.grid()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$\mathrm{Re}$')
    plt.ylabel(r'$\mathrm{Im}$')


def plot_vector(c, color='k', start=0, linestyle='-'):
    """Plot arrow corresponding to difference of two complex numbers"""
    return plt.arrow(np.real(start), np.imag(start), np.real(c), np.imag(c),
                     linestyle=linestyle, head_width=0.05, fc=color, ec=color,
                     overhang=0.3, length_includes_head=True)


c = 1.5 + 0.8j

generate_figure(figsize=(7.5, 3), xlim=[0, 2.5], ylim=[0, 1])
v = plot_vector(c, color='k')

plt.text(1.5, 0.8, '$c$', size='16')
plt.text(0.8, 0.55, '$|c|$', size='16')
plt.text(0.25, 0.05, '$\gamma$', size='16')
plt.show()

# %% [markdown]
# ## Polar Representation
#
# The **absolute value** (or **modulus**) of a complex number a+ib is defined by
# |c| := sqrt(a^2 + b^2).
#
# The **angle** (given in radians) is given by gamma := atan2(b, a).
# This yields a number in the interval (-pi, pi].

# %%
print('Absolute value:', np.abs(c))
print('Angle (in radians):', np.angle(c))
print('Angle (in degree):', np.rad2deg(np.angle(c)))
print('Angle (in degree):', 180 * np.angle(c) / np.pi)

# %% [markdown]
# The complex number c=a+ib is uniquely defined by the pair (|c|, gamma),
# which is also called the **polar representation** of c. One obtains the
# Cartesian representation (a,b) from the polar representation (|c|,gamma) as:
# a = |c| * cos(gamma)
# b = |c| * sin(gamma)

# %% [markdown]
# ## Operations
#
# For two complex numbers c1=a1+ib1 and c2=a2+ib2, the sum
# c1 + c2 = (a1 + a2) + i(b1 + b2)
# is defined by summing their real and imaginary parts individually.

# %%
c1 = 1.3 - 0.3j
c2 = 0.3 + 0.5j
c = c1 + c2

generate_figure(figsize=(7.5, 3), xlim=[-0.3, 2.2], ylim=[-0.4, 0.6])
v1 = plot_vector(c1, color='k')
v2 = plot_vector(c2, color='b')
plot_vector(c1, start=c2, linestyle=':', color='lightgray')
plot_vector(c2, start=c1, linestyle=':', color='lightgray')
v3 = plot_vector(c, color='r')

plt.legend([v1, v2, v3], ['$c_1$', '$c_2$', '$c_1+c_2$'])
plt.show()

# %% [markdown]
# Complex multiplication of two numbers c1=a1+ib1 and c2=a2+ib2 is defined by:
# c = c1 * c2 = (a1*a2 - b1*b2) + i(a1*b2 + b1*a2)
#
# Geometrically, the product is obtained by adding angles and by multiplying
# the absolute values.

# %%
c1 = 1.0 - 0.5j
c2 = 2.3 + 0.7j
c = c1 * c2

generate_figure(figsize=(7.5, 3), xlim=[-0.5, 4.0], ylim=[-0.75, 0.75])
v1 = plot_vector(c1, color='k')
v2 = plot_vector(c2, color='b')
v3 = plot_vector(c, color='r')
plt.legend([v1, v2, v3], ['$c_1$', '$c_2$', '$c_1 \cdot c_2$'])
plt.show()

# %% [markdown]
# Given a complex number c = a + bi, the **complex conjugation** is defined by
# conj(c) := a - bi. Geometrically, conjugation is reflection on the real axis.

# %%
c = 1.5 + 0.4j
c_conj = np.conj(c)

generate_figure(figsize=(7.5, 3), xlim=[0, 2.5], ylim=[-0.5, 0.5])
v1 = plot_vector(c, color='k')
v2 = plot_vector(c_conj, color='r')

plt.legend([v1, v2], ['$c$', r'$\overline{c}$'])
plt.show()

# %% [markdown]
# For a non-zero complex number c = a + bi, there is an **inverse** complex
# number c^-1 with the property that c * c^-1 = 1.
# The inverse is given by: c^-1 = conj(c) / |c|^2

# %%
c = 1.5 + 0.4j
c_inv = 1 / c
c_prod = c * c_inv

generate_figure(figsize=(7.5, 3), xlim=[-0.3, 2.2], ylim=[-0.5, 0.5])
v1 = plot_vector(c, color='k')
v2 = plot_vector(c_inv, color='r')
v3 = plot_vector(c_prod, color='gray')

plt.legend([v1, v2, v3], ['$c$', '$c^{-1}$', '$c*c^{-1}$'])
plt.show()

# %% [markdown]
# With the inverse, division can be defined:
# c1/c2 = c1 * c2^-1 = (c1 * conj(c2)) / |c2|^2

# %%
c1 = 1.3 + 0.3j
c2 = 0.8 + 0.4j
c = c1 / c2

generate_figure(figsize=(7.5, 3), xlim=[-0.25, 2.25], ylim=[-0.5, 0.5])
v1 = plot_vector(c1, color='k')
v2 = plot_vector(c2, color='b')
v3 = plot_vector(c, color='r')

plt.legend([v1, v2, v3], ['$c_1$', '$c_2$', '$c_1/c_2$'])
plt.show()

# %% [markdown]
# ## Polar Coordinate Plot
#
# Complex vectors can be visualized in a polar coordinate plot.

# %%
def plot_polar_vector(c, label=None, color=None, start=0, linestyle='-'):
    """Plot line in polar plane"""
    line = plt.polar([np.angle(start), np.angle(c)], [np.abs(start), np.abs(c)],
                     label=label, color=color, linestyle=linestyle)
    this_color = line[0].get_color() if color is None else color
    plt.annotate('', xytext=(np.angle(start), np.abs(start)), xy=(np.angle(c), np.abs(c)),
                 arrowprops=dict(facecolor=this_color, edgecolor='none',
                                 headlength=12, headwidth=10, shrink=1, width=0))


c_abs = 1.5
c_angle = 45  # in degree
c_angle_rad = np.deg2rad(c_angle)
a = c_abs * np.cos(c_angle_rad)
b = c_abs * np.sin(c_angle_rad)
c1 = a + b*1j
c2 = -0.5 + 0.75*1j

plt.figure(figsize=(6, 6))
plot_polar_vector(c1, label='$c_1$', color='k')
plot_polar_vector(np.conj(c1), label='$\overline{c}_1$', color='gray')
plot_polar_vector(c2, label='$c_2$', color='b')
plot_polar_vector(c1*c2, label='$c_1\cdot c_2$', color='r')
plot_polar_vector(c1/c2, label='$c_1/c_2$', color='g')

plt.ylim([0, 1.8])
plt.legend(framealpha=1)
plt.show()

# %% [markdown]
# ---
# **Acknowledgment:** This notebook was created by Frank Zalkow and Meinard Muller.
