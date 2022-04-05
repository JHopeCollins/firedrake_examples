
import numpy as np
import firedrake as fd

# set up domain on unit square
nx=10
ny=10

mesh = fd.UnitSquareMesh( nx,ny )
x,y = fd.SpatialCoordinate( mesh )

# function space for solution
# CG: continuous galerkin
V = fd.FunctionSpace( mesh, "CG", 1 )

# test and trial functions
u = fd.TrialFunction(V)
v = fd.TestFunction(V)

# right hand side function
f = fd.Function(V)

f.interpolate( (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2) )

# bilinear and linear forms for left and right hand sides

a = ( fd.inner( fd.grad(u), fd.grad(v) ) + fd.inner( u,v ) )*fd.dx
L = fd.inner( f, v )*fd.dx

# redifine u as function in V to hold the solution
u = fd.Function(V)

# solve with PETSc arguments:
#
# 'ksp_type' : krylov subspace solution method
#   'cg' -> conjugate gradient
#
# 'pc_type' : preconditioning
#   'none' -> no preconditioning

fd.solve( a==L, u, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'} )

# check solution against analytic solution
uexact = fd.Function(V)
uexact.interpolate( fd.cos(2*fd.pi*x)*fd.cos(2*fd.pi*y) )

# error function
uerror = u - uexact

# L2 residual
residual = fd.assemble( fd.dot(u-uexact,u-uexact)*fd.dx )

print( np.sqrt( residual ) ) 

# write paraview file to visualise the solution
fd.File("helmholtz.pvd").write(u)

# visualise with matplotlib
#import matplotlib.pyplot as plt
#fig, axes = plt.subplots()
#colors = fd.tripcolor( u, axes=axes )
#fig.colorbar( colors )
#plt.show()

