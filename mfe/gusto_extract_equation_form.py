import firedrake as fd
import gusto
from sys import exit

# set up gusto equation

mesh = fd.IcosahedralSphereMesh(radius=1, refinement_level=2, degree=1)
x, y, z = fd.SpatialCoordinate(mesh)

swe_parameters = gusto.ShallowWaterParameters(H=1, g=1, Omega=1)

dt = 1
domain = gusto.Domain(mesh, dt, 'BDM', degree=1)

eqn = gusto.ShallowWaterEquations(domain, swe_parameters, fexpr=z)

# get mass and stiffness form of serial form

from gusto.labels import (replace_subject, replace_test_function, time_derivative)
from gusto.fml.form_manipulation_labelling import all_terms, drop

M = eqn.residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_false=drop)

K = eqn.residual.label_map(lambda t: t.has_label(time_derivative),
                           map_if_true=drop)

# test replacing with elements from the serial space

W = eqn.function_space

w = fd.Function(W)
u, h = fd.split(w)
v, q = fd.TestFunctions(W)

R = eqn.residual
F = R.label_map(all_terms, replace_test_function(v, idx=0)) # works
F = R.label_map(all_terms, replace_test_function(q, idx=1)) # works
F = R.label_map(all_terms, replace_test_function((v, q))) # works

F = R.label_map(all_terms, replace_subject((u, h))) # works
F = R.label_map(all_terms, replace_subject(w)) # works
F = R.label_map(all_terms, replace_subject(u, idx=0)) # breaks 1
F = R.label_map(all_terms, replace_subject(h, idx=1)) # breaks 2

#print(K.form)
exit()

F = R.label_map(all_terms, replace_subject(h, idx=1)) # breaks 2
F = R.label_map(all_terms, replace_subject(u, idx=0)) # breaks 1

# generate parallel form by replacing functions in serial form
# with functions from all-at-once space

# all-at-once function space with two timesteps

WW = W*W

ww = fd.Function(WW)

u0, h0, u1, h1 = fd.split(ww)

v0, q0, v1, q1 = fd.TestFunctions(ww)
vt0, qt0, vt1, qt1 = fd.TrialFunctions(ww)

# mass form for first and second timesteps

# test functions

M0 = M.label_map(all_terms, replace_test_function(v0, idx=0)) \
      .label_map(all_terms, replace_test_function(q0, idx=1))

M1 = M.label_map(all_terms, replace_test_function(v1, idx=0)) \
      .label_map(all_terms, replace_test_function(q1, idx=1))

# trial functions

M0 = M0.label_map(all_terms, replace_subject(vt0, idx=0)) \
       .label_map(all_terms, replace_subject(qt0, idx=1))

M1 = M1.label_map(all_terms, replace_subject(vt1, idx=0)) \
       .label_map(all_terms, replace_subject(qt1, idx=1))

MM = M0 + M1

# stiffness form for first and second timesteps

K0 = K.label_map(all_terms, replace_test_function(v0, idx=0)) \
      .label_map(all_terms, replace_test_function(q0, idx=1))

K1 = K.label_map(all_terms, replace_test_function(v1, idx=0)) \
      .label_map(all_terms, replace_test_function(q1, idx=1))

# replace subject for stiffness

K0 = K0.label_map(all_terms, replace_subject(u0, idx=0)) \
       .label_map(all_terms, replace_subject(h0, idx=1))

K1 = K1.label_map(all_terms, replace_subject(u1, idx=0)) \
       .label_map(all_terms, replace_subject(h1, idx=1))

KK = K0 + K1

# try to do the same with the 'complex' space

W0 = W.split()[0]
W1 = W.split()[1]

Wc0 = fd.FunctionSpace(mesh, fd.VectorElement(W0.ufl_element(), dim=2))
Wc1 = fd.FunctionSpace(mesh, fd.VectorElement(W1.ufl_element(), dim=2))

Wc = Wc0*Wc1
