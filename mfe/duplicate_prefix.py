import firedrake as fd

mesh = fd.UnitSquareMesh(128, 128)  # works

x, y = fd.SpatialCoordinate(mesh)

V = fd.FunctionSpace(mesh, "CG", 1)

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

bcs = fd.DirichletBC(V, fd.zero(), (1, 2, 3, 4))

f = -0.5*fd.pi*fd.pi*(4*fd.cos(fd.pi*x) - 5*fd.cos(fd.pi*x*0.5) + 2)*fd.sin(fd.pi*y)

a = fd.dot(fd.grad(u), fd.grad(v))*fd.dx

L0 = fd.inner(f, v)*fd.dx
w0 = fd.Function(V)

params = {'ksp_type': 'cg', 'pc_type': 'mg'}

problem0 = fd.LinearVariationalProblem(a, L0, w0)
solver0 = fd.LinearVariationalSolver(problem0, options_prefix='dup', solver_parameters=params)

L1 = fd.inner(f, v)*fd.dx
w1 = fd.Function(V)

problem1 = fd.LinearVariationalProblem(a, L1, w1)
solver1 = fd.LinearVariationalSolver(problem1, options_prefix='dup')

solver0.solve()
solver1.solve()
