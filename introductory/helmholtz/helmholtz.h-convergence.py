
import numpy as np
import matplotlib.pyplot as plt

import firedrake as fd

def l2error( u : fd.Function,
             v : fd.Function ):
    return fd.assemble( fd.dot( u-v,u-v )*fd.dx )

def helmholtz_solve( V    : fd.FunctionSpace,
                     rhs  : fd.Function ):

    if rhs.function_space() != V :
        raise ValueError( "fd.Function 'rhs' should belong to the fd.FunctionSpace 'V'" )

    # test and trial functions
    u = fd.TrialFunction(V)
    v = fd.TestFunction(V)

    # bilinear and linear forms for left and right hand sides
    a = ( fd.inner( fd.grad(u), fd.grad(v) ) + fd.inner( u,v ) )*fd.dx
    L = fd.inner( rhs, v )*fd.dx

    # solution
    uresult = fd.Function(V)

    # solve with PETSc arguments:
    #
    # 'ksp_type' : krylov subspace solution method
    #   'cg' -> conjugate gradient
    #
    # 'pc_type' : preconditioning
    #   'none' -> no preconditioning
    
    fd.solve( a==L, uresult, solver_parameters={'ksp_type': 'cg', 'pc_type': 'none'} )

    return uresult

def helmholtz_error( V      : fd.FunctionSpace,
                     rhs    : fd.Function,
                     uexact : fd.Function ):

    if rhs.function_space() != V :
        raise ValueError( "fd.Function 'rhs' should belong to the fd.FunctionSpace 'V'" )

    if uexact.function_space() != V :
        raise ValueError( "fd.Function 'uexact' should belong to the fd.FunctionSpace 'V'" )

    return l2error( helmholtz_solve(V,rhs), uexact )

def simple_helmholtz( nx, order ):
    # set up domain on unit square
    mesh = fd.UnitSquareMesh( nx,nx )
    x,y = fd.SpatialCoordinate( mesh )

    # function space for solution
    # CG: continuous galerkin
    V = fd.FunctionSpace( mesh, "CG", order )

    # right hand side function
    f = fd.Function(V)
    f.interpolate( (1+8*fd.pi*fd.pi)*fd.cos(x*fd.pi*2)*fd.cos(y*fd.pi*2) )

    # analytic solution
    uexact = fd.Function(V)
    uexact.interpolate( fd.cos(2*fd.pi*x)*fd.cos(2*fd.pi*y) )

    return np.sqrt( helmholtz_error(V,f,uexact) )

order=1
mesh_sizes = np.array([8,16,32,64,128])
solution_errors = np.zeros_like(mesh_sizes,dtype=float)

for i in range(0,mesh_sizes.size):
    n = mesh_sizes[i]
    solution_errors[i] = simple_helmholtz(n,order)
    e = simple_helmholtz(n,order)
    print( i, " | ", mesh_sizes[i], " | ", solution_errors[i] )

plt.plot(mesh_sizes[0]/mesh_sizes,solution_errors/solution_errors[0])
plt.xscale('log')
plt.yscale('log')
plt.show()

