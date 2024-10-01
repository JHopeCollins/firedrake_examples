import firedrake as fd
from firedrake.petsc import PETSc
import numpy as np

Print = PETSc.Sys.Print

def build_problem(mesh_size, parameters, aP=None, mat_type='aij'):
    np.random.seed(54321)
    mesh = fd.UnitSquareMesh(2 ** mesh_size, 2 ** mesh_size)
    #mesh = fd.MeshHierarchy(mesh, 2)[-1]

    Sigma = fd.FunctionSpace(mesh, "RT", 1)
    V = fd.FunctionSpace(mesh, "DG", 0)
    W = Sigma * V

    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)

    f = fd.Function(V)
    fvector = f.vector()
    fvector.set_local(np.random.uniform(size=fvector.local_size()))

    a = (fd.dot(sigma, tau) + fd.div(tau)*u + fd.div(sigma)*v)*fd.dx
    L = -f*v*fd.dx

    if aP is not None:
        aP = aP(W)
        P = fd.assemble(aP, mat_type=mat_type)

    w = fd.Function(W)
    problem = fd.LinearVariationalProblem(a, L, w, aP=aP)
    solver =  fd.LinearVariationalSolver(problem, solver_parameters=parameters)

    return solver, w


# a naive approach
parameters = {
    'ksp_type': 'gmres',
    'ksp_gmres_restart': 100,
    'ksp_rtol': 1e-8,
    #'pc_type': 'ilu',
    'pc_type': 'bjacobi',
    'pc_sub_pc_type': 'ilu',
}

# an exact schur complement
parameters = {
    #'ksp_monitor': None,
    #'ksp_converged_reason': None,
    'ksp_type': 'fgmres',
    'ksp_rtol': 1e-8,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'fieldsplit_0': {
        #'ksp_monitor': None,
        #'ksp_converged_reason': None,
        'ksp_type': 'cg',
        'pc_type': 'bjacobi',
        'pc_sub_pc_type': 'icc',
        'ksp_rtol': 1e-12,
    },
    'fieldsplit_1': {
        #'ksp_monitor': None,
        #'ksp_converged_reason': None,
        'ksp_type': 'cg',
        'pc_type': 'none',
        'ksp_rtol': 1e-12,
    }
}

# mass-lump preconditioning the schur complement
parameters = {
    #'ksp_monitor': None,
    #'ksp_converged_reason': None,
    'ksp_type': 'gmres',
    'ksp_rtol': 1e-8,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'pc_fieldsplit_schur_precondition': 'selfp',
    'fieldsplit_0': {
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'pc_sub_pc_type': 'icc',
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_type': 'hypre',
    }
}

# auxillary schur complement approximation
class DGLaplacian(fd.AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        W = u.function_space()
        n = fd.FacetNormal(W.mesh())
        alpha = fd.Constant(4.0)
        gamma = fd.Constant(8.0)
        h = fd.CellSize(W.mesh())
        h_avg = (h('+') + h('-'))/2

        a_dg = \
            -(fd.inner(fd.grad(u), fd.grad(v))*fd.dx \
            - fd.inner(fd.jump(u, n), fd.avg(fd.grad(v)))*fd.dS \
            - fd.inner(fd.avg(fd.grad(u)), fd.jump(v, n), )*fd.dS \
            + alpha/h_avg * fd.inner(fd.jump(u, n), fd.jump(v, n))*fd.dS \
            - fd.inner(u*n, fd.grad(v))*fd.ds \
            - fd.inner(fd.grad(u), v*n)*fd.ds \
            + (gamma/h)*fd.inner(u, v)*fd.ds)

        bcs = None
        return (a_dg, bcs)

parameters = {
    #'ksp_monitor': None,
    #'ksp_converged_reason': None,
    'ksp_type': 'gmres',
    'ksp_rtol': 1e-8,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'schur',
    'pc_fieldsplit_schur_fact_type': 'full',
    'fieldsplit_0': {
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'pc_sub_pc_type': 'icc',
    },
    'fieldsplit_1': {
        'ksp_type': 'preonly',
        'pc_type': 'python',
        'pc_python_type': __name__ + '.DGLaplacian',
        'aux_pc_type': 'hypre',
    }
}


# block diagonal riesz map preconditioner

def riesz(W):
    sigma, u = fd.TrialFunctions(W)
    tau, v = fd.TestFunctions(W)
    return (fd.dot(sigma, tau) + fd.div(sigma)*fd.div(tau) + u*v)*fd.dx

parameters = {
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'ksp_type': 'fgmres',
    'ksp_rtol': 1e-8,
    'pc_type': 'fieldsplit',
    'pc_fieldsplit_type': 'additive',
    'fieldsplit_0': {
        #'ksp_monitor': None,
        #'ksp_converged_reason': None,
        # ~ ~
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
    },
    'fieldsplit_1': {
        #'ksp_converged_reason': None,
        'ksp_type': 'preonly',
        'pc_type': 'bjacobi',
        'pc_sub_pc_type': 'icc',
    },
}

Print(f"cell count  |  gmres its")
for n in range(8):
    solver, w = build_problem(n, parameters, aP=riesz, mat_type='nest')
    solver.solve()
    num_cells = 2*(2**(2*n))
    iterations = solver.snes.ksp.getIterationNumber()
    Print(f"{str(num_cells).rjust(10,' ')}  |  {str(iterations).rjust(4, ' ')}")
