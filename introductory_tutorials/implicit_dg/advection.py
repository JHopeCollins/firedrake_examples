
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import pi, cos, sin

import firedrake as fd
from firedrake.petsc import PETSc

import argparse

parser = argparse.ArgumentParser(
    description='Scalar advection of a Gaussian bump in a periodic square with DG in space and implicit-theta in time.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--nx', type=int, default=64, help='Number of cells along each square side.')
parser.add_argument('--nt', type=int, default=64, help='Number of timesteps.')
parser.add_argument('--cfl', type=float, default=0.8, help='Convective CFL number.')
parser.add_argument('--angle', type=float, default=pi/6, help='Angle of the convective velocity.')
parser.add_argument('--degree', type=int, default=1, help='Degree of the scalar and velocity spaces.')
parser.add_argument('--theta', type=float, default=0.5, help='Parameter for the implicit theta timestepping method.')
parser.add_argument('--width', type=float, default=0.1, help='Width of the Gaussian bump.')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

umax = 1.
dx = 1./args.nx
dt = args.cfl*dx/umax


# # # === --- domain --- === # # #

mesh = fd.PeriodicUnitSquareMesh(args.nx, args.nx, quadrilateral=True)
x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "DQ", args.degree)
W = fd.VectorFunctionSpace(mesh, "CG", args.degree)

# # # === --- initial conditions --- === # # #

# gaussian bump


def radius(x, y):
    return fd.sqrt(pow(x-0.5, 2) + pow(y-0.5, 2))


def gaussian(x, y):
    return fd.exp(-0.5*pow(radius(x, y)/args.width, 2))


q0 = fd.Function(V, name="scalar_initial")
q0.interpolate(1 + gaussian(x, y))

# angled advection velocity
u = fd.Function(W, name='velocity')
u.interpolate(fd.as_vector((umax*cos(args.angle), umax*sin(args.angle))))


# # # === --- finite element forms --- === # # #


def form_mass(q, phi):
    return phi*q*fd.dx


def form_function(q, phi):
    # upwind switch
    n = fd.FacetNormal(mesh)
    un = 0.5*(fd.dot(u, n) + abs(fd.dot(u, n)))

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over internal facets
    int_facet = (phi('+')-phi('-'))*(un('+')*q('+')-un('-')*q('-'))*fd.dS

    return int_facet - int_cell


# # # === --- timestepping scheme --- === # # #


q = fd.Function(V, name='scalar').assign(q0)
q1 = fd.Function(V, name='scalar_next').assign(q0)
phi = fd.TestFunction(V)

dt1 = fd.Constant(1./dt)
theta = fd.Constant(args.theta)

M = form_mass(q1, phi) - form_mass(q, phi)
K = theta*form_function(q1, phi) + (1 - theta)*form_function(q, phi)

F = dt1*M + K

params = {
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'ksp_type': 'gmres',
    'ksp_rtol': 1e-10,
    'pc_type': 'bjacobi',
}

problem = fd.NonlinearVariationalProblem(F, q1)
solver = fd.NonlinearVariationalSolver(problem, solver_parameters=params)


# # # === --- Timestepping loop --- === # # #


timeseries = [q.copy(deepcopy=True)]

for i in range(args.nt):
    solver.solve()
    q.assign(q1)
    timeseries.append(q.copy(deepcopy=True))


# # # === --- plotting --- === # # #
nsp = 32

fn_plotter = fd.FunctionPlotter(mesh, num_sample_points=nsp)

fig, axes = plt.subplots()
axes.set_aspect('equal')
colors = fd.tripcolor(q, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
fig.colorbar(colors)


def animate(q):
    colors.set_array(fn_plotter(q))


interval = 1e2
animation = FuncAnimation(fig, animate, frames=timeseries, interval=interval)

animation.save("advection.mp4", writer="ffmpeg")
