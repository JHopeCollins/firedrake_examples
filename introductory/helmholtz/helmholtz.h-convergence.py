
import numpy as np
import matplotlib.pyplot as plt

import firedrake as fd

def l2error( u : fd.Function,
             v : fd.Function ):
    if u.function_space() != v.function_space() :
        raise ValueError( "fd.Function 'u' and fd.Function 'v' must belong to the same fd.FunctionSpace" )
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

def simple_helmholtz( nx, order ):
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

    return np.sqrt( l2error( helmholtz_solve(V,f), uexact ) )

### === --- main --- === ###

order=1
mesh_sizes = np.array([8,16,32,64,128])
solution_errors = np.zeros_like(mesh_sizes,dtype=float)

print( "element order: ", order )
print( "nx   |  error       |  convergence rate" )
for i in range( 0, len(mesh_sizes) ):
    error = simple_helmholtz( mesh_sizes[i], order )
    solution_errors[i] = error

    error0 = solution_errors[max(0,i-1)]
    convergence_rate = np.sqrt(  error0 / error )

    print( str(mesh_sizes[i]).ljust(4),
           "| ",
           np.format_float_scientific( error, precision=4 ),
           " | ",
           np.format_float_scientific( convergence_rate, precision=2 ) )

relative_sizes  = mesh_sizes[0]/mesh_sizes
relative_errors = solution_errors/solution_errors[0]

# write paraview file to visualise the solution
fd.File("helmholtz.pvd").write(u)

#plt.plot( relative_sizes, relative_errors )
#plt.xscale('log')
#plt.yscale('log')
#plt.show()

