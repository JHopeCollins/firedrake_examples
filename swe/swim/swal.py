import firedrake as fd
#get command arguments
from petsc4py import PETSc
PETSc.Sys.popErrorHandler()

from functools import partial

# asQ utils
from utils import mg
from utils.shallow_water.williamson1992 import case5 as case
from utils.planets import earth
from utils.shallow_water import nonlinear as swe

import argparse
parser = argparse.ArgumentParser(
    description='Williamson 5 testcase for augmented Lagrangian solver.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve.')
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid.')
parser.add_argument('--dmax', type=float, default=0.01, help='Final time in days.05.')
parser.add_argument('--dumpt', type=float, default=24, help='Dump time in hours.')
parser.add_argument('--gamma', type=float, default=1.0e4, help='Augmented Lagrangian scaling parameter.')
parser.add_argument('--dt', type=float, default=0.5, help='Timestep in hours.')
parser.add_argument('--filename', type=str, default='w5aug')
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')
parser.add_argument('--kspschur', type=int, default=40, help='Max number of KSP iterations on the Schur complement.')
parser.add_argument('--kspmg', type=int, default=3, help='Max number of KSP iterations in the MG levels.')
parser.add_argument('--tlblock', type=str, default='mg', help='Solver for the velocity-velocity block. mg==Multigrid with patchPC, lu==direct solver with MUMPS, patch==just do a patch smoother.')
parser.add_argument('--schurpc', type=str, default='mass', help='Preconditioner for the Schur complement. mass==mass inverse, helmholtz==helmholtz inverse * laplace * mass inverse.')
parser.add_argument('--show_args', action='store_true', default=True, help='Output all the arguments.')

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# some domain, parameters and FS setup
H = case.H0
distribution_parameters = {"partition": True, "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)}

mesh = mg.icosahedral_mesh(earth.radius,
                           base_level=args.base_level,
                           degree=1,
                           distribution_parameters=distribution_parameters,
                           nrefs=args.ref_level-args.base_level)
x = fd.SpatialCoordinate(mesh)
outward_normals = fd.CellNormal(mesh)


def perp(u):
    return fd.cross(outward_normals, u)


degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

f = case.coriolis_expression(*x)
g = earth.Gravity  # Gravitational constant

b = fd.Function(V2, name="Topography")
b.interpolate(case.topography_expression(*x))
gamma = fd.Constant(args.gamma)

# D = eta + b

One = fd.Function(V2).assign(1.0)

u, eta = fd.TrialFunctions(W)
v, phi = fd.TestFunctions(W)

dx = fd.dx

Un = fd.Function(W)
Unp1 = fd.Function(W)

u0, h0 = fd.split(Un)
u1, h1 = fd.split(Unp1)
n = fd.FacetNormal(mesh)


def both(u):
    return 2*fd.avg(u)


dT = fd.Constant(0.)
dS = fd.dS


half = fd.Constant(0.5)

def ufunc(v, u, h):
    return swe.form_function_velocity(mesh, g, b, f, u, h, v)

def hfunc(phi, u, h):
    return swe.form_function_depth(mesh, u, h, phi)

def umass(v, u):
    return swe.form_mass_u(mesh, u, v)

def hmass(phi, h):
    return swe.form_mass_h(mesh, h, phi)

def ueq(v, u0, u1, h0, h1):
    form = (
    umass(v, u1 - u0)
    + half*dT*ufunc(v, u0, h0)
    + half*dT*ufunc(v, u1, h1)
    )
    return form

def heq(phi, u0, u1, h0, h1):
    form =(
    hmass(phi, h1 - h0)
    + half*dT*hfunc(phi, u0, h0)
    + half*dT*hfunc(phi, u1, h1)
    )
    return form

testeqn = (
    ueq(v, u0, u1, h0, h1)
    + heq(phi, u0, u1, h0, h1)
)

aleqn = gamma*heq(fd.div(v), u0, u1, h0, h1)

# the extra bit
eqn = testeqn + aleqn

class HelmholtzPC(fd.PCBase):
    def initialize(self, pc):
        if pc.getType() != "python":
            raise ValueError("Expecting PC type python")
        prefix = pc.getOptionsPrefix() + "helmholtz_"

        mm_solve_parameters = {
            'ksp_type':'preonly',
            'pc_type':'bjacobi',
            'sub_pc_type':'lu',
        }

        # we assume P has things stuffed inside of it
        _, P = pc.getOperators()
        context = P.getPythonContext()
        appctx = context.appctx
        self.appctx = appctx

        # FunctionSpace checks
        u, v = context.a.arguments()
        if u.function_space() != v.function_space():
            raise ValueError("Pressure space test and trial space differ")

        # the mass solve
        a = u*v*fd.dx
        self.Msolver = fd.LinearSolver(fd.assemble(a),
                                       solver_parameters=
                                       mm_solve_parameters)
        # the Helmholtz solve
        eta0 = appctx.get("helmholtz_eta", 20)
        def get_laplace(q,phi):
            h = fd.avg(fd.CellVolume(mesh))/fd.FacetArea(mesh)
            mu = eta0/h
            n = fd.FacetNormal(mesh)
            ad = (- fd.jump(phi)*fd.jump(fd.inner(fd.grad(q),n))
                  - fd.jump(q)*fd.jump(fd.inner(fd.grad(phi),n)))*fd.dS
            ad +=  mu * fd.jump(phi)*fd.jump(q)*fd.dS
            ad += fd.inner(fd.grad(q), fd.grad(phi)) * fd.dx
            return ad

        a = (fd.Constant(2)/dT/H)*u*v*fd.dx + fd.Constant(0.5)*g*dT*get_laplace(u, v)
        #input and output functions
        V = u.function_space()
        self.xfstar = fd.Function(V) # the input residual
        self.xf = fd.Function(V) # the output function from Riesz map
        self.yf = fd.Function(V) # the preconditioned residual

        L = get_laplace(u, self.xf*gamma)
        hh_prob = fd.LinearVariationalProblem(a, L, self.yf)
        self.hh_solver = fd.LinearVariationalSolver(
            hh_prob,
            options_prefix=prefix)

    def update(self, pc):
        pass

    def applyTranspose(self, pc, x, y):
        raise NotImplementedError

    def apply(self, pc, x, y):

        # copy petsc vec into Function
        with self.xfstar.dat.vec_wo as v:
            x.copy(v)
        
        #do the mass solver, solve(x, b)
        self.Msolver.solve(self.xf, self.xfstar)

        # get the mean
        xbar = fd.assemble(self.xf*fd.dx)/fd.assemble(One*fd.dx)
        self.xf -= xbar
        
        #do the Helmholtz solver
        self.hh_solver.solve()

        # add the mean
        self.yf += xbar/2*dT*gamma*H
        
        # copy petsc vec into Function
        with self.yf.dat.vec_ro as v:
            v.copy(y)


sparameters = {
    "mat_type":"matfree",
    'snes_monitor': None,
    'snes_converged_reason': None,
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    #"pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
}

bottomright_helm = {
    "ksp_type": "fgmres",
    "ksp_monitor": None,
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_max_it": args.kspschur,
    "pc_type": "python",
    "pc_python_type": "__main__.HelmholtzPC",
    "helmholtz" :
    {"ksp_type":"preonly",
     "pc_type": "lu"}
}

bottomright_mass = {
    "ksp_type": "preonly",
    #"ksp_monitor":None,
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_max_it": args.kspschur,
    #"ksp_monitor":None,
    "pc_type": "python",
    "pc_python_type": "firedrake.MassInvPC",
    "Mp_pc_type": "bjacobi",
    "Mp_sub_pc_type": "ilu"
}

if args.schurpc == "mass":
    sparameters["fieldsplit_1"] = bottomright_mass
elif args.schurpc == "helmholtz":
    sparameters["fieldsplit_1"] = bottomright_helm
else:
    raise KeyError('Unknown Schur PC option.')

topleft_LU = {
    "ksp_type": "preonly",
    "pc_type": "python",
    "pc_python_type": "firedrake.AssembledPC",
    "assembled_pc_type": "lu",
    "assembled_pc_factor_mat_solver_type": "mumps"
}

topleft_MG = {
    "ksp_type": "preonly",
    "pc_type": "mg",
    #"pc_mg_type": "full",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": args.kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.PatchPC",
    "mg_levels_patch_pc_patch_save_operators": True,
    "mg_levels_patch_pc_patch_partition_of_unity": False,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqaij",
    "mg_levels_patch_pc_patch_construct_type": "star",
    "mg_levels_patch_pc_patch_multiplicative": False,
    "mg_levels_patch_pc_patch_symmetrise_sweep": False,
    "mg_levels_patch_pc_patch_construct_dim": 0,
    "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
    "mg_levels_patch_pc_patch_dense_inverse": True,
    "mg_levels_patch_pc_patch_precompute_element_tensors": True,
    "mg_levels_patch_sub_pc_factor_mat_solver_type": "petsc",
    "mg_levels_patch_sub_ksp_type": "preonly",
    "mg_levels_patch_sub_pc_type": "lu",
}

topleft_MGs = {
    "ksp_type": "preonly",
    "ksp_max_it": 3,
    "pc_type": "mg",
    "mg_coarse_ksp_type": "preonly",
    "mg_coarse_pc_type": "python",
    "mg_coarse_pc_python_type": "firedrake.AssembledPC",
    "mg_coarse_assembled_pc_type": "lu",
    "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
    "mg_levels_ksp_type": "gmres",
    "mg_levels_ksp_max_it": args.kspmg,
    "mg_levels_pc_type": "python",
    "mg_levels_pc_python_type": "firedrake.AssembledPC",
    "mg_levels_assembled_pc_type": "python",
    "mg_levels_assembled_pc_python_type": "firedrake.ASMStarPC",
    "mg_levels_assembled_pc_star_backend": "tinyasm",
    "mg_levels_assmbled_pc_star_construct_dim": 0
}

topleft_smoother = {
    "ksp_type": "gmres",
    "ksp_max_it": 3,
    "ksp_monitor": None,
    "pc_type": "python",
    "pc_python_type": "firedrake.PatchPC",
    "patch_pc_patch_save_operators": True,
    "patch_pc_patch_partition_of_unity": False,
    "patch_pc_patch_sub_mat_type": "seqaij",
    "patch_pc_patch_construct_type": "star",
    "patch_pc_patch_multiplicative": False,
    "patch_pc_patch_symmetrise_sweep": False,
    "patch_pc_patch_construct_dim": 0,
    "patch_sub_ksp_type": "preonly",
    "patch_sub_pc_type": "lu",
}

if args.tlblock == "mg":
    sparameters["fieldsplit_0"] = topleft_MG
elif args.tlblock == "patch":
    sparameters["fieldsplit_0"] = topleft_smoother
else:
    assert(args.tlblock=="lu")
    sparameters["fieldsplit_0"] = topleft_LU
dt = 60*60*args.dt
dT.assign(dt)
t = 0.

nprob = fd.NonlinearVariationalProblem(eqn, Unp1)
ctx = {"mu": gamma*2/g/dt}
nsolver = fd.NonlinearVariationalSolver(nprob,
                                        solver_parameters=sparameters,
                                        appctx=ctx)
vtransfer = mg.ManifoldTransfer()
tm = fd.TransferManager()
transfers = {
    V1.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject),
    V2.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
                       vtransfer.inject)
}
transfermanager = fd.TransferManager(native_transfers=transfers)
nsolver.set_transfer_manager(transfermanager)

dmax = args.dmax
hmax = 24*dmax
tmax = 60.*60.*hmax
hdump = args.dumpt
dumpt = hdump*60.*60.
tdump = 0.

un = fd.Function(V1, name="Velocity")
un.project(case.velocity_expression(*x))
etan = fd.Function(V2, name="Elevation")
etan.project(case.elevation_expression(*x))

u0 = Un.subfunctions[0]
h0 = Un.subfunctions[1]
u0.assign(un)
h0.assign(etan + H - b)

q = fd.TrialFunction(V0)
p = fd.TestFunction(V0)

qn = fd.Function(V0, name="Relative Vorticity")
veqn = q*p*dx + fd.inner(perp(fd.grad(p)), un)*dx
vprob = fd.LinearVariationalProblem(fd.lhs(veqn), fd.rhs(veqn), qn)
qparams = {'ksp_type':'cg'}
qsolver = fd.LinearVariationalSolver(vprob,
                                     solver_parameters=qparams)

file_sw = fd.File(args.filename+'.pvd')
etan.assign(h0 - H + b)
un.assign(u0)
qsolver.solve()
file_sw.write(un, etan, qn)
Unp1.assign(Un)

itcount = 0
stepcount = 0
while t < tmax + 0.5*dt:
    PETSc.Sys.Print(f"\nTime: {t/3600} | Iteration: {stepcount}\n")
    t += dt
    tdump += dt

    nsolver.solve()
    Un.assign(Unp1)
    
    if tdump > dumpt - dt*0.5:
        etan.assign(h0 - H + b)
        un.assign(u0)
        qsolver.solve()
        file_sw.write(un, etan, qn)
        tdump -= dumpt
    stepcount += 1
    itcount += nsolver.snes.getLinearSolveIterations()

PETSc.Sys.Print("Iterations", itcount, "its per step", itcount/stepcount)
