
import firedrake as fd
from petsc4py import PETSc

import firedrake_utils.mg as mg
import firedrake_utils.planets.earth as earth

import firedrake_utils.shallow_water as swe
import firedrake_utils.shallow_water.williamson1992.case5 as case5

import argparse


# def form_mass(u, h, v, q):
#     return fd.inner(u, v)*fd.dx + h*q*fd.dx
# 
# 
# def form_velocity(mesh, g, b, f, u, h, v):
#     n = fd.FacetNormal(mesh)
#     outward_normals = fd.CellNormal(mesh)
# 
#     def perp(u):
#         return fd.cross(outward_normals, u)
# 
#     def both(u):
#         return 2*fd.avg(u)
# 
#     K = 0.5*fd.inner(u, u)
#     upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
# 
#     return (fd.inner(v, f*perp(u))*fd.dx
#             - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
#             + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
#                        both(upwind*u))*fd.dS
#             - fd.div(v)*(g*(h + b) + K)*fd.dx)
# 
# 
# def form_depth(mesh, g, b, u, h, q):
#     n = fd.FacetNormal(mesh)
#     uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
# 
#     return (- fd.inner(fd.grad(q), u)*h*fd.dx
#             + fd.jump(q)*(uup('+')*h('+')
#                         - uup('-')*h('-'))*fd.dS)
# 
# 
# def form_function(mesh, g, b, f, u, h, v, q):
#     return form_velocity(mesh, g, b, f, u, h, v) \
#            + form_depth(mesh, g, b, u, h, q)


PETSc.Sys.popErrorHandler()

# get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for monolithic fully implicit solver.')  # noqa: E501
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')  # noqa: E501
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 3.')  # noqa: E501
parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')  # noqa: E501
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')  # noqa: E501
parser.add_argument('--filename', type=str, default='output/w5diag')  # noqa: E501
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')  # noqa: E501
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')  # noqa: E501
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')  # noqa: E501

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# # # === --- mesh set up --- === # # #

R0 = earth.radius
nrefinements = args.ref_level - args.base_level
distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

mesh = mg.icosahedral_mesh(R0,
                           args.base_level,
                           args.coords_degree,
                           distribution_parameters,
                           nrefinements)

R0 = fd.Constant(R0)
x, y, z = fd.SpatialCoordinate(mesh)

# # # === --- function spaces --- === # # #

degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

# # # === --- initial conditions --- === # # #

# constants
Omega = earth.Omega  # rotation rate
g = earth.Gravity
H0 = case5.H0  # reference free surface height

# H0: reference free surface height
# H0 + eta: actual free surface height
# b: topology (ground surface height)
# h = (H0+eta)-b: depth

# coriolis expression
f = case5.coriolis_expression(x, y, z)

# initial velocity
u_init = case5.velocity_function(x, y, z, V1, name="Initial Velocity")

# topography
b = case5.topography_function(x, y, z, V2, name="Topography")

# initial surface height perturbation
eta_init = fd.Function(V2, name='Initial Elevation')
eta_init.project(case5.depth_expression(x, y, z) - H0)

# initial depth field
h_init = fd.Function(V2, name='Initial Depth').assign(eta_init + H0 - b)

# write initial data
outfile = fd.File(args.filename+'.pvd')
eta_out = fd.Function(V2, name='Elevation').assign(eta_init)
u_out = fd.Function(V1, name='Velocity').assign(u_init)
outfile.write(u_out, eta_out)

# # # === --- finite element forms --- === # # #

dt = 0.1
dT = fd.Constant(dt)
dT1 = fd.Constant(1/dt)
half = fd.Constant(0.5)

# current and next timestep
un0 = fd.Function(W)
un1 = fd.Function(W)

u0, h0 = fd.split(un0)
u1, h1 = fd.split(un1)

un0.split()[0].assign(u_init)
un0.split()[1].assign(h_init)
un1.split()[0].assign(u_init)
un1.split()[1].assign(h_init)

v, q = fd.TestFunctions(W)

form_mass = dT1*(
    swe.nonlinear.form_mass(mesh, h1, u1, q, v )
    - swe.nonlinear.form_mass(mesh, h0, u0, q, v )
)

form_function = half*(
    swe.nonlinear.form_function(mesh, g, b, f, h0, u0, q, v )
    + swe.nonlinear.form_function(mesh, g, b, f, h1, u1, q, v )
)

eqn = form_mass + form_function

# # # === --- set up nonlinear solver --- === # # #

# parameters for the implicit solve
sparams = {
    "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor": None,
    "ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 5,
    # "mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_codim": 0,
    "mg_levels_patch_pc_patch_construct_type": "vanka",
    "mg_levels_patch_pc_patch_local_type": "additive",
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
    "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
}

nlproblem = fd.NonlinearVariationalProblem(eqn, un1)

nlsolver = fd.NonlinearVariationalSolver(nlproblem, solver_parameters=sparams)

nlsolver.set_transfer_manager(mg.manifold_transfer_manager(W))

# # # === --- timestepping loop --- === # # #

nlsolver.solve()

