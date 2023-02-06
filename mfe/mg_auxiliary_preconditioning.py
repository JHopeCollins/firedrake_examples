import firedrake as fd

base_mesh = fd.UnitSquareMesh(8, 8)
mesh = fd.MeshHierarchy(base_mesh, 4)[-1]  # breaks

x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

bcs = fd.DirichletBC(V, fd.zero(), (1, 2, 3, 4))

f = -0.5*fd.pi*fd.pi*(4*fd.cos(fd.pi*x) - 5*fd.cos(fd.pi*x*0.5) + 2)*fd.sin(fd.pi*y)
L = fd.inner(f, v)*fd.dx

a = fd.dot(fd.grad(u), fd.grad(v))*fd.dx
aP = fd.Constant(1.5)*a

class MGAuxPC(fd.AuxiliaryOperatorPC):
    def form(self, pc, u, v):
        return (aP, bcs)

parameters = {
    'ksp_monitor': None,
    'ksp_converged_reason': None,
    'ksp_type': 'fgmres',
    'pc_type': 'python',
    'pc_python_type': __name__+'.MGAuxPC',
    'aux_pc_type': 'ksp',
    'aux_ksp': {
        'ksp_type': 'cg',
        'ksp': {
            'monitor': None,
            'converged_reason': None,
            'rtol': 1e-5
        },
        'pc_type': 'mg',
        'pc_mg_type': 'full',
        'pc_mg_cycle_type': 'w',
        'mg': {
            'levels': {
                'ksp_type': 'richardson',
                'ksp_max_it': 5,
                'pc_type': 'jacobi',
                'ksp_richardson_scale': 2/3,
            },
            'coarse': {
                'pc_type': 'python',
                'pc_python_type': 'firedrake.AssembledPC',
                'assembled_pc_type': 'lu',
                'assembled_pc_factor_mat_solver_type': 'mumps',
            }
        }
    }
}

w = fd.Function(V)
fd.solve(a == L, w, bcs=bcs, solver_parameters=parameters)
