from ufl import (Cell, CellNormal, Coefficient, FunctionSpace, Mesh,
                 VectorElement, derivative, dx, grad, inner)
from ufl.algorithms import compute_form_data
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering

mesh = Mesh(VectorElement("P", Cell("triangle", 3), 2))

try_workaround = False
if try_workaround:
    n = apply_geometry_lowering(CellNormal(mesh))
else:
    n = CellNormal(mesh)

V = FunctionSpace(mesh, VectorElement("P", mesh.ufl_cell(), 2))

u = Coefficient(V)

phi = grad(inner(u, n))[0]*dx

fd = compute_form_data(
    derivative(phi, u),
    do_apply_function_pullbacks=True,
    do_apply_geometry_lowering=True,
)

