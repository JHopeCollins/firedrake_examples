
from math import log
import numpy as np
import firedrake as fd

# set up domain on unit square
nx=10
ny=10

mesh = fd.UnitSquareMesh( nx,ny, quadrilateral=True )
x,y = fd.SpatialCoordinate( mesh )

# function space for solution
# CG: continuous galerkin
Vs = [fd.FunctionSpace(mesh, "Q", i) for i in range(1,8)]

# test and trial functions
us = [fd.TrialFunction(W) for W in Vs]
vs = [fd.TestFunction(W) for W in Vs]

# right hand side function
f = (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2)

# bilinear and linear forms for left and right hand sides

As = [( fd.inner( fd.grad(u), fd.grad(v) ) + fd.inner( u,v ) )*fd.dx for u,v in zip(us,vs)]
Ls = [fd.inner( f, v )*fd.dx for v in vs]

# redefine u as function in V to hold the solution
us = [fd.Function(W) for W in Vs]

# solve with PETSc arguments:
#
# 'ksp_type' : krylov subspace solution method
#   'cg' -> conjugate gradient
#
# 'pc_type' : preconditioning
#   'none' -> no preconditioning

for A, L, u in zip(As,Ls,us):
    fd.solve( A==L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'} )

# check solution against analytic solution
uexacts = [fd.Function(W).interpolate( fd.cos(2*fd.pi*x)*fd.cos(2*fd.pi*y) ) for W in Vs]

# error function
uerrors = [u - uexact for u,uexact in zip(us,uexacts)]

# L2 residual
residuals = [fd.errornorm(uexact,u) for u,uexact in zip(us,uexacts)]

for res in residuals:
    print( res ) 

## write paraview file to visualise the solution
#fd.File("helmholtz.pvd").write(u)

# visualise with matplotlib
#import matplotlib.pyplot as plt
#fig, axes = plt.subplots()
#colors = fd.tripcolor( u, axes=axes )
#fig.colorbar( colors )
#plt.show()

