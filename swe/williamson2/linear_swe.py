
import firedrake as fd

from firedrake_utils import units
from firedrake_utils.planets import earth

import firedrake_utils.shallow_water.linear as swe
from firedrake_utils.shallow_water.williamson1992 import case2


### === --- inputs --- === ###

# constants from Williamson1992
gh0 = case2.gh0

# time discretisation
delta_t = 20*units.minute  # implicit timestep size
tmax = 1*units.hour        # simulation time
theta = 0.0                # implicit theta-method ( 0: backward Euler, 0.5: trapezium )
out_freq = 2               # number of delta_ts between snapshots

# mesh refinement
refinement_level = 3

# function space degrees
element_degree = 2


### === --- process inputs --- === ###

g = fd.Constant(earth.gravity)
H = fd.Constant(case2.h0)    # mean depth
Omega = fd.Constant(earth.omega)
u0 = fd.Constant(case2.u0)    # max velocity
dt = fd.Constant(delta_t)
R0 = fd.Constant(earth.radius)

velocity_degree = element_degree
height_degree = element_degree-1
mesh_degree = element_degree+1


### === --- set up mesh --- === ###

globe = earth.IcosahedralMesh(
            refinement_level = refinement_level,
            degree = mesh_degree )

coords = fd.SpatialCoordinate( globe )

### === --- function spaces --- === ###

# solution function spaces
V1 = fd.FunctionSpace( globe, "BDM", velocity_degree )
V2 = fd.FunctionSpace( globe, "DG",    height_degree )
W  = fd.MixedFunctionSpace( (V1,V2) )

# function space for coriolis parameter
Vf = fd.FunctionSpace( globe, "CG", mesh_degree )


### === --- analytical solution --- === ###

# steady-state geostrophic balance solution to nonlinear SWE

f = case2.coriolis_function(      *coords, Vf )
uexact = case2.velocity_function( *coords, V1 )
hexact = case2.depth_function(    *coords, V2 )

fd.File( "vtk/williamson2.exact.pvd" ).write( uexact, hexact )


### === --- full equations --- === ###

# solution at current timestep
wn = fd.Function(W)
un, hn = wn.split()

# use exact profiles as initial conditions
un.assign(uexact)
hn.assign(hexact)

# residual weights for theta method
imp_weight = fd.Constant( (1-theta)*dt )
exp_weight = fd.Constant( (  theta)*dt )

u,h = fd.TrialFunctions( W )
w,p = fd.TestFunctions(  W )

# forms for next (lhs) and current (rhs) timesteps
lhs = swe.form_mass( globe, h, u,  p,w ) + imp_weight*swe.form_function( globe, g,H,f, h, u,  p,w )
rhs = swe.form_mass( globe, hn,un, p,w ) - exp_weight*swe.form_function( globe, g,H,f, hn,un, p,w )

equation = lhs - rhs


### === --- linear solver --- === ###

# solution at next step
wn1 = fd.Function(W)
un1, hn1 = wn1.split()

problem = fd.LinearVariationalProblem( fd.lhs(equation),
                                       fd.rhs(equation),
                                       wn1 )

# petsc parameters
params = { 'mat_type' : 'aij',
           'ksp_type' : 'preonly',
           'pc_type'  : 'lu',
           'pc_factor_mat_solver_type' : 'mumps'
         }

solver = fd.LinearVariationalSolver( problem, solver_parameters = params )


### === --- output files --- === ###

# set up output
u_out = fd.Function( V1, name="velocity" ).assign(un)
h_out = fd.Function( V2, name="depth"    ).assign(hn)

outfile = fd.File( "vtk/williamson2.pvd" )
outfile.write( u_out, h_out )


### === --- timestepping loop --- === ###

t=0.0
timestep=0

while t<tmax:
    t+=delta_t
    solver.solve()

    un.assign( un1 )
    hn.assign( hn1 )

    timestep+=1

    if timestep%out_freq==0 :
        print( "saving solution at timestep ", timestep, " and time: ", t/units.hour, " hours" )
        u_out.assign(un)
        h_out.assign(hn)
        outfile.write( u_out, h_out )


### === --- finish up --- === ###

# write final timestep if not already done
if timestep%out_freq!=0:
    u_out.assign(un)
    h_out.assign(hn)
    outfile.write( u_out, h_out )

