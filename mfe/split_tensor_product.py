import firedrake as fd

mesh2D = fd.UnitSquareMesh(4, 4, quadrilateral=True)

mesh1D = fd.UnitIntervalMesh(4)
meshEx = fd.ExtrudedMesh(mesh1D, 4, layer_height=0.25)

V = fd.FunctionSpace(mesh2D, "CG", 1)
Ve = fd.FunctionSpace(meshEx, "CG", 1)

v = fd.Function(V)
ve = fd.Function(Ve)

fd.split(v)   # works
fd.split(ve)  # breaks

print(type(v.ufl_element()))
print(type(ve.ufl_element()))

print(fd.split(v))
print(fd.split(ve))
