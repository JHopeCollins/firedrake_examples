
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import firedrake as fd

### === --- initial profile functions --- === ###

# euclidean distance between (x0,y0) and (x1,y1)
def euclid( x0,y0, x1,y1 ):
    return fd.sqrt( pow(x1-x0,2) + pow(y1-y0,2) )

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

### === --- set up --- === ###

# time steps
T = 2.0*math.pi
dt = T/600.0
dtc = fd.Constant(dt)

# domain
nx = 40

mesh = fd.UnitSquareMesh( nx,nx, quadrilateral=True )
x,y = fd.SpatialCoordinate( mesh )

# function spaces for passive scalar and convecting velocity
V = fd.FunctionSpace(       mesh, "DQ", 1 )
W = fd.VectorFunctionSpace( mesh, "CG", 1 )

### === --- initial conditions --- === ###

# cosine-bell-cone-slotted-cylinder

qinitial = fd.Function(V).interpolate( 1.0 + bell( x,y )
                                           + cone( x,y )
                                           + slot( x,y ) )

q = fd.Function(V, name="q").assign(qinitial)

# rotating velocity field
u = fd.Function(W).interpolate(fd.as_vector((0.5-y,x-0.5)))

### === --- set up finite element scheme --- === ###

# boundary condition
q_inflow = fd.Constant(1.0)

# variational forms

# mass matrix
p = fd.TrialFunction(V)
phi = fd.TestFunction(V)
a = phi*p*fd.dx

# right hand side

# upwind switch
n = fd.FacetNormal(mesh)
un = 0.5*( fd.dot(u,n) + abs(fd.dot(u,n)) )

# cell volume integration
int_cell = q*fd.div(phi*u)*fd.dx

# boundary integration
int_inflow  = fd.conditional( fd.dot(u,n) < 0, phi*fd.dot(u,n)*q_inflow, 0 )*fd.ds
int_outflow = fd.conditional( fd.dot(u,n) > 0, phi*fd.dot(u,n)*q       , 0 )*fd.ds

# internal face integration
int_facet = ( phi('+') - phi('-'))*( un('+')*q('+') - un('-')*q('-') )*fd.dS

# residual calculation for first stage
L1 = dtc*( int_cell - ( int_inflow + int_outflow + int_facet ) )

# stage solutions
q1 = fd.Function(V)
q2 = fd.Function(V)

L2 = fd.replace( L1, {q:q1})
L3 = fd.replace( L1, {q:q2})

# stage increment
dq = fd.Function(V)

# precompute assembled problem
params = {
    'ksp_type'    : 'preonly',
    'pc_type'     : 'bjacobi',
    'sub_pc_type' : 'ilu'
    }

prob1 = fd.LinearVariationalProblem( a, L1, dq )
solv1 = fd.LinearVariationalSolver( prob1, solver_parameters=params )

prob2 = fd.LinearVariationalProblem( a, L2, dq )
solv2 = fd.LinearVariationalSolver( prob2, solver_parameters=params )

prob3 = fd.LinearVariationalProblem( a, L3, dq )
solv3 = fd.LinearVariationalSolver( prob3, solver_parameters=params )

### === --- timestepping loop --- === ###

t = 0.0
step = 0
output_freq = 20
third = 1./3.
end = 1.0

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
        print( f"step {step} | t {t}" )
    step += 1
    t += dt

with fd.CheckpointFile('advection.h5', 'w') as cpfile:
    cpfile.save_mesh(mesh)
    cpfile.save_function(q)

q.assign(0)

with fd.CheckpointFile('advection.h5', 'r') as cpfile:
    mesh = cpfile.load_mesh()
    q.assign(cpfile.load_function(mesh, name="q"))

t = 0.0
while t < ( end*T - 0.5*dt ):
    solv1.solve()
    q1.assign( q + dq )

    solv2.solve()
    q2.assign( 0.75*q + 0.25*( q1 + dq ) )

    solv3.solve()
    q.assign( third*q + 2*third*( q2 + dq ) )

    if step % output_freq == 0:
        qs.append( q.copy( deepcopy=True ) )
        print( f"step {step} | t {t}" )
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
colors = fd.tripcolor( q, num_sample_points=nsp, vmin=1, vmax=2, axes=axes)
fig.colorbar(colors)

def animate( q ):
    colors.set_array(fn_plotter(q))

interval = 1e3*output_freq*dt
animation = FuncAnimation( fig, animate, frames=qs, interval=interval )

animation.save( "advection.mp4", writer="ffmpeg" )

