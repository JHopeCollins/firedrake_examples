import firedrake as fd
from ufl import replace
from gusto.labels import replace_subject, replace_test_function, subject

mesh = fd.UnitIntervalMesh(4)
V = fd.FunctionSpace(mesh, "CG", 1)


### form on single space

u = fd.Function(V)
v = fd.TestFunction(V)

orig_form = fd.inner(u, v)*fd.dx

### mixed space for multiple timesteps

W = V*V

w = fd.Function(W)
ws = fd.split(w)
ys = fd.TestFunctions(W)

### Do it by hand with ufl

# I need to be able to get a form with the test/trials replaced
# with the function/test function from the mixed space. It needs
# to be from `fd.split(w)` not `w.split()` because we need to
# build a form over all components of the mixed space

# diagonal terms
mixed_form_0 = replace(orig_form, {u: ws[0]})
mixed_form_0 = replace(mixed_form_0, {v: ys[0]})

mixed_form_1 = replace(orig_form, {u: ws[1]})
mixed_form_1 = replace(mixed_form_1, {v: ys[1]})

# off-diagonal coupling term
# (e.g. time-derivative involves current and previous timestep)
mixed_form_2 = replace(orig_form, {u: ws[0]})
mixed_form_2 = replace(mixed_form_2, {v: ys[1]})

mixed_form = mixed_form_0 + mixed_form_1 + mixed_form_2

### try gusto

# I make a term and tell it what the function part is

term = subject(orig_form, u).terms[0]

# I make two new terms with the test functions replaced
# with components of the mixed space

mixed_term_0 = replace_test_function(ys[0])(term)
mixed_term_1 = replace_test_function(ys[1])(term)
mixed_term_2 = replace_test_function(ys[1])(term)

# but replacing the subject doesn't work!
# What am I getting wrong?

mixed_term_0 = replace_subject(ws[0])(mixed_term_0)
mixed_term_1 = replace_subject(ws[1])(mixed_term_1)
mixed_term_2 = replace_subject(ws[0])(mixed_term_1)

# this works (as in it doesn't throw) but I don't think
# that this is what we need
mixed_term_0 = replace_subject(w.split()[0])(mixed_term_0)
mixed_term_1 = replace_subject(w.split()[1])(mixed_term_1)
mixed_term_2 = replace_subject(w.split()[0])(mixed_term_1)

# this is what I need at the end
mixed_labelled_form = mixed_term_0 + mixed_term_1 + mixed_term_2

### I need to be able to do things like:
# w.assign(expr)
# fd.assemble(mixed_labelled_form, tensor=another_w)
# jac = fd.derivative(mixed_labelled_form, w)
