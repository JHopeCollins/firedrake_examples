import firedrake as fd

import firedrake_utils.mg as mg
from firedrake_utils.planets import earth
import firedrake_utils.shallow_water.nonlinear as swe
from firedrake_utils.shallow_water.williamson1992 import case5

#get command arguments
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()
import argparse
parser = argparse.ArgumentParser(description='Williamson 5 testcase for augmented Lagrangian solver.')
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')
parser.add_argument('--ref_level', type=int, default=5, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--dmax', type=float, default=1, help='Final time in days. Default 1.')
parser.add_argument('--dumpt', type=float, default=1, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=3, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = earth.radius
H = case5.H0

distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

mesh = mg.icosahedral_mesh(R0=R0,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level)

R0 = fd.Constant(R0)
x,y,z = fd.SpatialCoordinate(mesh)


V1 = fd.FunctionSpace(mesh, "BDM", args.degree+1)
V2 = fd.FunctionSpace(mesh, "DG", args.degree)
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)
W = fd.MixedFunctionSpace((V1, V2))

# U_t + N(U) = 0
#
# TRAPEZOIDAL RULE
# U^{n+1} - U^n + dt*( N(U^{n+1}) + N(U^n) )/2 = 0.
    
# Newton's method
# f(x) = 0, f:R^M -> R^M
# [Df(x)]_{i,j} = df_i/dx_j
# x^0, x^1, ...
# Df(x^k).xp = -f(x^k)
# x^{k+1} = x^k + xp.

f = case5.coriolis_expression(x,y,z)
b = case5.topography_function(x, y, z, V2, name="Topography")
g = earth.Gravity

Un = fd.Function(W)
Unp1 = fd.Function(W)

v, phi = fd.TestFunctions(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)

half = fd.Constant(0.5)
dT = fd.Constant(0.)

equation = (
    swe.form_mass(mesh, h1-h0, u1-u0, phi, v)
    + half*dT*swe.form_function(mesh, g, b, f, h0, u0, phi, v)
    + half*dT*swe.form_function(mesh, g, b, f, h1, u1, phi, v))

# monolithic solver options

sparameters = {
    "snes_monitor": None,
    "snes_converged_reason": None,
    "snes_atol": 1e-0,
    "snes_rtol": 1e-12,
    "snes_stol": 1e-12,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    "ksp_monitor": None,
    # "ksp_converged_reason": None,
    # "ksp_atol": 1e-8,
    # "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 5,
    #"mg_levels_ksp_convergence_test": "skip",
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": True,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_construct_dim": 0,
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
    
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(equation, Unp1)
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters,
                                        appctx={})

transfermanager = mg.manifold_transfer_manager(W)
nsolver.set_transfer_manager(transfermanager)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

un = case5.velocity_function(x, y, z, V1, name="Velocity")
etan = case5.elevation_function(x, y, z, V2, name="Elevation")

u0, h0 = Un.split()
u0.assign(un)
h0.assign(etan + H - b)

qn = fd.Function(V0, name="Relative Vorticity")

outward_normals = fd.CellNormal(mesh)
def perp(u):
    return fd.cross(outward_normals, u)

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

veqn = q*p*fd.dx + fd.inner(perp(fd.grad(p)), un)*fd.dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File('output/'+args.filename+'.pvd')


def write_file():
    etan.assign(h0 - H + b)
    un.assign(u0)
    qsolver.solve()
    file_sw.write(un, etan, qn)


write_file()

Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
linear_itcount = 0
nonlinear_itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)

    if tdump > dumpt - dt*0.5:
        PETSc.Sys.Print('')
        PETSc.Sys.Print('===--- ', stepcount, ' | ', t/(60*60), ' ---===' )
        PETSc.Sys.Print('')
        write_file()
        tdump -= dumpt
    stepcount += 1
    linear_itcount += nsolver.snes.getLinearSolveIterations()
    nonlinear_itcount += nsolver.snes.getIterationNumber()

PETSc.Sys.Print("Linear iterations", linear_itcount)
PETSc.Sys.Print("Nonlinear iterations (linear solves)", nonlinear_itcount)
PETSc.Sys.Print("Linear iterations per timestep", linear_itcount/stepcount)
PETSc.Sys.Print("Nonlinear iterations (linear solves) per timestep", nonlinear_itcount/stepcount)
PETSc.Sys.Print("stepcount", stepcount, "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)
write_file()
