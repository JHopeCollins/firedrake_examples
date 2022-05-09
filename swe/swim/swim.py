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
parser.add_argument('--ref_level', type=int, default=2, help='Refinement level of icosahedral grid. Default 2.')
parser.add_argument('--dmax', type=float, default=0.125, help='Final time in days. Default 0.125.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours. Default 24.')
parser.add_argument('--dt', type=float, default=1, help='Timestep in hours. Default 1.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=3, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')
parser.add_argument('--time_scheme', type=int, default=0, help='Timestepping scheme. 0=Crank-Nicholson. 1=Implicit midpoint rule.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
R0 = earth.radius
H = case5.H0
name = args.filename

distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

mesh = mg.icosahedral_mesh(R0=R0,
                           base_level=args.base_level,
                           degree=args.coords_degree,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level)

R0 = fd.Constant(R0)
x,y,z = fd.SpatialCoordinate(mesh)

outward_normals = fd.CellNormal(mesh)

V1 = fd.FunctionSpace(mesh, "BDM", args.degree+1)
V2 = fd.FunctionSpace(mesh, "DG", args.degree)
V0 = fd.FunctionSpace(mesh, "CG", args.degree+2)
W = fd.MixedFunctionSpace((V1, V2))

f = case5.coriolis_expression(x,y,z)
g = earth.Gravity

# Topography.
b = case5.topography_function(x, y, z, V2, name="Topography")

# D = eta + b


n = fd.FacetNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


def both(u):
    return 2*fd.avg(u)


dT = fd.Constant(0.)


def u_op(v, u, h):
    Upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)
    K = 0.5*fd.inner(u, u)
    return (fd.inner(v, f*perp(u))*fd.dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                          both(Upwind*u))*fd.dS
            - fd.div(v)*(g*(h + b) + K)*fd.dx)


def h_op(phi, u, h):
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))
    return (- fd.inner(fd.grad(phi), u)*h*fd.dx
            + fd.jump(phi)*(uup('+')*h('+')
                            - uup('-')*h('-'))*fd.dS)


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

Un = fd.Function(W)
Unp1 = fd.Function(W)

v, phi = fd.TestFunctions(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)
half = fd.Constant(0.5)

# augmented lagrangian parameter - to be removed
gamma = fd.Constant(0)

testeqn = (
    fd.inner(v, u1 - u0)*fd.dx
    + half*dT*u_op(v, u0, h0)
    + half*dT*u_op(v, u1, h1)
    + phi*(h1 - h0)*fd.dx
    + half*dT*h_op(phi, u0, h0)
    + half*dT*h_op(phi, u1, h1))
# the extra bit
eqn = testeqn \
    + gamma*(fd.div(v)*(h1 - h0)*fd.dx
             + half*dT*h_op(fd.div(v), u0, h0)
             + half*dT*h_op(fd.div(v), u1, h1))
    
# monolithic solver options

sparameters = {
    "snes_monitor": None,
    "mat_type": "matfree",
    "ksp_type": "fgmres",
    #"ksp_monitor_true_residual": None,
    "ksp_converged_reason": None,
    "ksp_atol": 1e-8,
    "ksp_rtol": 1e-8,
    "ksp_max_it": 400,
    "pc_type": "mg",
    "pc_mg_cycle_type": "v",
    "pc_mg_type": "multiplicative",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": 3,
    #"mg_levels_ksp_convergence_test": "skip",
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
    
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
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

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*fd.dx + fd.inner(perp(fd.grad(p)), un)*fd.dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File('output/'+name+'.pvd')


def write_file():
    etan.assign(h0 - H + b)
    un.assign(u0)
    qsolver.solve()
    file_sw.write(un, etan, qn)


write_file()

Unp1.assign(Un)

PETSc.Sys.Print('tmax', tmax, 'dt', dt)
itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print('===--- ', itcount, ' | ', t, ' ---===' )
    t += dt
    tdump += dt

    nsolver.solve()
    res = fd.assemble(testeqn)
    PETSc.Sys.Print(res.dat.data[0].max(), res.dat.data[0].min(),
          res.dat.data[1].max(), res.dat.data[1].min())
    Un.assign(Unp1)
    res = fd.assemble(testeqn)
    PETSc.Sys.Print(res.dat.data[0].max(), res.dat.data[0].min(),
          res.dat.data[1].max(), res.dat.data[1].min())
    
    if tdump > dumpt - dt*0.5:
        write_file()
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()
PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount,
                "dt", dt, "ref_level", args.ref_level, "dmax", args.dmax)
write_file()
