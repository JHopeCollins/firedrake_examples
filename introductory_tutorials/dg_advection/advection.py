
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import firedrake as fd

# euclidean distance between (x0,y0) and (x1,y1)
def euclid( x0,y0, x1,y1 ):
    return fd.sqrt( pow(x1-x0,2) + pow(y1-y0,2) )

### === --- initial profile functions --- === ###
# profiles from Leveque1996

# cosine-bell profile
def bell( x,y, x0=0.25, y0=0.50, r0=0.15 ):
    return 0.5*(1+fd.cos(math.pi*fd.min_value(euclid(x0,y0,x,y)/r0, 1.0)))

# cone profile
def cone( x,y, x0=0.50, y0=0.25, r0=0.15 ):
    return 1.0 - fd.min_value(euclid(x0,y0,x,y)/r0, 1.0)

# slotted cylinder
def slot( x,y, x0=0.50, y0=0.75, r0=0.15, left=0.475, right=0.525, top=0.85 ):
    return fd.conditional(euclid(x0,y0,x,y) < r0,
             fd.conditional(fd.And(fd.And(x > left, x < right), y < top),
             0.0, 1.0), 0.0)

### === --- advection equation terms --- === ###

# mass matrix for time-derivative
def advection_mass( phi : fd.TestFunction ): # test function for solution
    V = phi.function_space()
    dq = fd.TrialFunction(V)
    return phi*dq*fd.dx

# integrations for right hand side spatial terms over cells and facets
def advection_rhs( u   : fd.Function,       # velocity field
                   phi : fd.TestFunction,   # test function for solution
                   q   : fd.Function,       # solution function
                   qin : fd.Constant ):     # inflow value

    if( u.function_space().rank !=1 ):
        raise ValueError( "'u' must be a vector valued function" )

    if( q.function_space() != phi.function_space() ):
        raise ValueError( "fd.Function 'q' and fd.TestFunction 'phi' must be defined on same fd.FunctionSpace" )

    mesh = q.function_space().mesh()

    if( mesh != u.function_space().mesh() ):
        raise ValueError( "fd.Function 'u' and 'q' must be defined over the same mesh" )

    # upwind switch
    n = fd.FacetNormal(mesh)
    un = 0.5*( fd.dot(u,n) + abs(fd.dot(u,n)) )

    # integration over element volume
    int_cell = q*fd.div(phi*u)*fd.dx

    # integration over boundary facets
    int_inflow  = fd.conditional( fd.dot(u,n) < 0, phi*fd.dot(u,n)*qin, 0 )*fd.ds
    int_outflow = fd.conditional( fd.dot(u,n) > 0, phi*fd.dot(u,n)*q,   0 )*fd.ds

    # integration over internal facets
    int_facet = ( phi('+') - phi('-') )*( un('+')*q('+') - un('-')*q('-') )*fd.dS

    return ( int_cell - ( int_inflow + int_outflow + int_facet ) )

# return mass matrix and right hand side forms
def advection_forms( u   : fd.Function,     # velocity field
                     q   : fd.Function,     # solution function
                     qin : fd.Constant,     # inflow value
                     dtc : fd.Constant ):   # timestep

    if( u.function_space().rank !=1 ):
        raise ValueError( "'u' must be a vector valued function" )

    if( u.function_space().mesh() != q.function_space().mesh() ):
        raise ValueError( "fd.Function 'u' and 'q' must be defined over the same mesh" )

    phi = fd.TestFunction( q.function_space() )

    return advection_mass( phi ), dtc*advection_rhs( u, phi, q, qin )


### === --- case parameters --- === ###

# time steps
nt = 600

# proportion of period to run for
end = 1.0

T = 2.0*math.pi
dt = T/nt
dtc = fd.Constant(dt)

# mesh size
nx = 40

# snapshot frequency
output_freq = 20

# background scalar value
qref = 1.0

# solver parameters
params = {
    'ksp_type'    : 'preonly',
    'pc_type'     : 'bjacobi',
    'sub_pc_type' : 'ilu'
    }

### === --- domain and function spaces --- === ###

mesh = fd.UnitSquareMesh( nx,nx, quadrilateral=True )
x,y = fd.SpatialCoordinate( mesh )

# function spaces for passive scalar and convecting velocity
V = fd.FunctionSpace(       mesh, "DQ", 1 )
W = fd.VectorFunctionSpace( mesh, "CG", 1 )

### === --- initial conditions --- === ###

# cosine-bell-cone-slotted-cylinder

qinitial = fd.Function(V) \
             .interpolate( qref + bell( x,y )
                                + cone( x,y )
                                + slot( x,y ) )

q = fd.Function(V).assign(qinitial)

# rotating velocity field

u = fd.Function(W).interpolate(fd.as_vector((0.5-y,x-0.5)))

### === --- set up finite element scheme --- === ###

# boundary condition
q_inflow = fd.Constant(qref)

# variational forms
a, L1 = advection_forms( u, q, q_inflow, dtc )

# stage solutions
q1 = fd.Function(V)
q2 = fd.Function(V)

L2 = fd.replace( L1, {q:q1})
L3 = fd.replace( L1, {q:q2})

# stage increment
dq = fd.Function(V)

# precompute assembled problems
prob1 = fd.LinearVariationalProblem( a, L1, dq )
prob2 = fd.LinearVariationalProblem( a, L2, dq )
prob3 = fd.LinearVariationalProblem( a, L3, dq )

solv1 = fd.LinearVariationalSolver( prob1, solver_parameters=params )
solv2 = fd.LinearVariationalSolver( prob2, solver_parameters=params )
solv3 = fd.LinearVariationalSolver( prob3, solver_parameters=params )

### === --- timestepping loop --- === ###

t = 0.0
step = 0
third = 1./3.

# list of solution snapshots
qs = []

while t < ( end*T - 0.5*dt ):
    solv1.solve()
    q1.assign( q + dq )

    solv2.solve()
    q2.assign( 0.75*q + 0.25*( q1 + dq ) )

    solv3.solve()
    q.assign( third*q + 2*third*( q2 + dq ) )

    if step % output_freq == 0:
        qs.append( q.copy( deepcopy=True ) )
        #print( "step: ", step, " | t = ", t )
    step += 1
    t += dt


l2_error = fd.sqrt( fd.assemble( ( q-qinitial )*( q-qinitial )*fd.dx ) )
l2_init  = fd.sqrt( fd.assemble( (   qinitial )*(   qinitial )*fd.dx ) )

print( "L2 error: ", l2_error/l2_init )

### === --- plotting --- === ###
nsp = 32

fn_plotter = fd.FunctionPlotter( mesh, num_sample_points=nsp )

fig, axes = plt.subplots()
axes.set_aspect('equal')
colors = fd.tripcolor( q, num_sample_points=nsp, vmin=qref, vmax=qref+1, axes=axes)
fig.colorbar(colors)

def animate( q ):
    colors.set_array(fn_plotter(q))

interval = 1e3*output_freq*dt
animation = FuncAnimation( fig, animate, frames=qs, interval=interval )

animation.save( "advection.mp4", writer="ffmpeg" )

