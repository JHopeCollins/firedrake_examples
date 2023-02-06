import firedrake as fd

mg = True

if mg is True:
    base_mesh = fd.UnitSquareMesh(8, 8)
    mesh = fd.MeshHierarchy(base_mesh, 4)[-1]  # breaks
else:
    mesh = fd.UnitSquareMesh(128, 128)  # works

x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

bcs = fd.DirichletBC(V, fd.zero(), (1, 2, 3, 4))

f = -0.5*fd.pi*fd.pi*(4*fd.cos(fd.pi*x) - 5*fd.cos(fd.pi*x*0.5) + 2)*fd.sin(fd.pi*y)
L = fd.inner(f, v)*fd.dx

a = fd.dot(fd.grad(u), fd.grad(v))*fd.dx
aP = fd.Constant(0.5)*a

parameters = {
    'ksp_type': 'cg',
    'pc_type': 'mg',
}

w = fd.Function(V)
fd.solve(a == L, w, Jp=aP, bcs=bcs, solver_parameters=parameters)
