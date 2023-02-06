from firedrake import *
from gusto import replace_subject, subject, all_terms, Term
from gusto.labels import perp as perp_label
from ufl import replace

mesh = UnitSquareMesh(10, 10)
V = VectorFunctionSpace(mesh, "CG", 1)
W = V*V
p, q = TestFunctions(W)
v, w = TrialFunctions(W)

f = Function(W)
g = Function(W)

print("f: ", f, "g: ", g)

f0, _ = split(f)
g0, _ = split(g)

def perpify(t):
    perp_function = t.get(perp_label)
    replace_dict = {split(g)[0]: perp_function(split(g)[0])}
    for k, v in replace_dict.items(): print(k, " : ", v)
    new_form = replace(t.form, replace_dict)
    return Term(new_form, t.labels)

b1 = perp_label(subject(inner(f0, p)*dx, f), perp)
b2 = b1.label_map(all_terms, replace_subject(g))
b3 = b2.label_map(lambda t: t.has_label(perp_label), replace_subject(perp(split(g)[0]), idx=0))
b4 = b2.label_map(lambda t: t.has_label(perp_label), perpify)

print("b1: ", b1.form)
print("b2: ", b2.form)
print("b3: ", b3.form)
print("b4: ", b4.form)
print("rq: ", inner(perp(g0), p)*dx)

#b3 = b2.label_map(lambda t: t.has_label(perp_label), replace_subject(perp(g0), idx=0))

# b = subject(inner(perp(f0), p)*dx, f)
# print(b.form)
# print("\n")
# b2 = b.label_map(all_terms, replace_subject(g)).label_map(all_terms, replace_subject(perp(g0), 0))
# print(b2.form)
# print("\n")
# #b3 = b2.label_map(all_terms, replace_subject(perp(g0), 0))
# b3 = Term(replace(b2.terms[0].form, {f0: perp(f0)}), b2.terms[0].labels)
# print(b3.form)
# print("\n")
# 
# c1 = subject(inner(f0, p)*dx, f)
# c2 = c1.label_map(all_terms, replace_subject(g))
# c3 = c2.label_map(all_terms, replace_subject(f))
# print(c1.form)
# print(c2.form)
# print(c3.form)
# print()
# 
# d1 = perp_label(subject(inner(f0, p)*dx, f), perp)
# d2 = d1.label_map(all_terms, replace_subject(g))
# d3 = d2.label_map(all_terms, replace_subject(f))
# print(d1.form)
# print(d2.form)
# print(d3.form)
