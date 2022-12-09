import firedrake as fd
import numpy as np
import matplotlib.pyplot as plt

mesh = fd.UnitSquareMesh(10, 10, quadrilateral=True)

V = fd.FunctionSpace(mesh, "Q", 1)
zero = fd.Constant(0)

x,y = fd.SpatialCoordinate(mesh)
u_exact = fd.sin(fd.pi*x)*fd.sin(fd.pi*y)
#f = (2*fd.pi**2)*u_exact
f = zero

u = fd.TrialFunction(V)
v = fd.TestFunction(V)

a = fd.dot(fd.grad(u), fd.grad(v))*fd.dx
L = f*v*fd.dx

#boundary_ids = (1,2,3,4)
#bcs = fd.DirichletBC(V, 0, boundary_ids)
bcs = [fd.DirichletBC(V, 1, 3),
       fd.DirichletBC(V, 2, 4)]

solver_parameters = {
    'ksp_type': 'cg',
    'pc_type': 'none'
}

uh = fd.Function(V)
fd.solve( a==L, uh, bcs=bcs, solver_parameters=solver_parameters)

fig, axes = plt.subplots()
collection = fd.tripcolor(uh, axes=axes)
fig.colorbar(collection)
plt.show()
