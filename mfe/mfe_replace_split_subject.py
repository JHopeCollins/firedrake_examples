import firedrake as fd
from gusto.labels import replace_subject, replace_test_function, subject

# base function space

mesh = fd.UnitIntervalMesh(4)
V = fd.FunctionSpace(mesh, "CG", 1)

# form on single space

u = fd.Function(V)
v = fd.TestFunction(V)

orig_form = fd.inner(u, v)*fd.dx

# mixed space (for multiple timesteps)

W = V*V

w = fd.Function(W)
ys = fd.TestFunction(W)

# label the form

term = subject(orig_form, u).terms[0]

# replace base test/functions with test/functions from
# the mixed space (i.e. transfer time-serial form onto
# various timesteps in the all-at-once system)

# this works
replace_test_function(ys[0])(term)

# this works, but isn't what I need
replace_subject(w.split()[0])(term)

# this breaks, but is what I need (fd.split(w) knows
# that it is part of the all-at-once mixed space)
#replace_subject(fd.split(w)[0])(term)

replace_test_function(ys, idx=0)(term)

replace_subject(w, idx=0)(term)
replace_subject(w.split(), idx=0)(term)
replace_subject(w.split()[0])(term)

vv = fd.Function(V)
replace_subject(fd.split(vv)[0])(term)

replace_subject(fd.split(w), idx=0)(term)
replace_subject(fd.split(w)[0])(term)

mterm = subject(fd.inner(w, ys)*fd.dx, w).terms[0]

replace_subject(fd.split(w)[0], idx=0)(mterm)
replace_subject(fd.split(w), idx=0)(mterm)
