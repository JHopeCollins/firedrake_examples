import firedrake as fd

mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)

V = fd.FunctionSpace(mesh, "Q", 1)

v = fd.TestFunction(V)
u = fd.Function(V)

F = u*v*fd.dx

sparams1 = { 'ksp_rtol': 1e-10 }
sparams2 = { 'ksp': { 'rtol': 1e-10 } }

nlprob = fd.NonlinearVariationalProblem(F, u)

print("Constructing NonlinearVariationalSolver with solver_parameters = { 'ksp_rtol': 1e-10 }")
nlsolv1 = fd.NonlinearVariationalSolver(nlprob, solver_parameters=sparams1)

print("Constructing NonlinearVariationalSolver with solver_parameters = { 'ksp': { 'rtol': 1e-10 } }")
nlsolv2 = fd.NonlinearVariationalSolver(nlprob, solver_parameters=sparams2)
