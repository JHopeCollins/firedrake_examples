import firedrake as fd
import ufl

mesh = fd.UnitSquareMesh(4, 4)
V = fd.FunctionSpace(mesh, "CG", 1)
W = V*V

u = fd.Function(V)
v = fd.TestFunction(V)
w = fd.TrialFunction(V)

uu = fd.Function(W)
vv = fd.TestFunction(W)
ww = fd.TrialFunction(W)

form = fd.inner(u, v)*fd.dx
mixed_form = fd.inner(uu, vv)*fd.dx

vnew = fd.TestFunction(V)

form_new = ufl.replace(form, {v: vnew})

vvs = fd.split(mixed_form.arguments()[0])

mixed_form_new = ufl.replace(form, {vvs[0]: v})

form_new2 = ufl.replace(form, {form.arguments()[0]: vvs[1]})
