
import firedrake as fd
from math import pi

# time durations in seconds
second = 1.
minute = 60*second
hour = 60*minute
day = 24*hour


### === --- inputs --- === ###

# constants from Williamson1992
R0 = 6371220.           # earth's radius
Omega = 7.292e-5        # earth's rotation
g = 9.80616             # earth's gravity
gh0 = 2.94e4            # gravity * mean height
period = 12.0           # days taken for velocity to travel circumference

# time discretisation
delta_t = 20*minute     # implicit timestep size
tmax = 2*day            # simulation time
theta = 0.0             # implicit theta-method ( 0: backward Euler, 0.5: trapezium )
out_freq = 3            # number of delta_ts between snapshots

# mesh refinement
refinement_level = 3

# function space degrees
element_degree = 3


### === --- process inputs --- === ###

H = fd.Constant(gh0/g)    # mean depth
Omega = fd.Constant(Omega)
g = fd.Constant(g)
u0 = fd.Constant(2*pi*R0/(period*day))    # max velocity
dt = fd.Constant(delta_t)

velocity_degree = element_degree
height_degree = element_degree-1
mesh_degree = element_degree+1


### === --- set up mesh --- === ###

earth = fd.IcosahedralSphereMesh(
            radius = R0,
            refinement_level = refinement_level,
            degree = mesh_degree )

R0 = fd.Constant(R0)

earth.init_cell_orientations( fd.SpatialCoordinate( earth ) )
x,y,z = fd.SpatialCoordinate( earth )


### === --- function spaces --- === ###

# solution function spaces
V1 = fd.FunctionSpace( earth, "BDM", velocity_degree )
V2 = fd.FunctionSpace( earth, "DG",    height_degree )
W  = fd.MixedFunctionSpace( (V1,V2) )

# function space for coriolis parameter
Vf = fd.FunctionSpace( earth, "CG", mesh_degree )


### === --- analytical solution --- === ###

# steady-state analytical solution to nonlinear SWE

# coriolis parameter
f_exp = 2*Omega*z/R0
f = fd.Function(Vf).interpolate( f_exp )

# exact profiles for geostrophic balance
u_exp = fd.as_vector([ -u0*y/R0, u0*x/R0, 0.0 ])
h_exp = H - ( R0*Omega*u0 + 0.5*u0*u0 )*(z*z/(R0*R0))/g

uexact = fd.Function( V1, name="velocity" ).project(u_exp)
hexact = fd.Function( V2, name="depth" ).interpolate(h_exp)

fd.File( "williamson2.exact.pvd" ).write(uexact,hexact)


### === --- finite element forms --- === ###

# linear shallow water equation forms

outward_normals = fd.CellNormal(earth)
perp = lambda u: fd.cross( outward_normals, u )

def form_function_h( H, h, u, p ):
    return ( H*p*fd.div(u) )*fd.dx

def form_function_u( g, f, h, u, w ):
    return ( fd.inner( w, f*perp(u) ) - g*h*fd.div(w) )*fd.dx

def form_mass_h( h, p ):
    return ( p*h )*fd.dx

def form_mass_u( u, w ):
    return fd.inner( u, w )*fd.dx


### === --- full equations --- === ###

# use exact profiles as initial conditions
un = fd.Function(V1).assign(uexact)
hn = fd.Function(V2).assign(hexact)

imp_weight = fd.Constant( (1-theta)*dt )
exp_weight = fd.Constant( (  theta)*dt )

u,h = fd.TrialFunctions( W )
w,p = fd.TestFunctions(  W )

lhs_h = form_mass_h( h,  p ) + imp_weight*form_function_h( H, h,  u,  p )
rhs_h = form_mass_h( hn, p ) - exp_weight*form_function_h( H, hn, un, p )

lhs_u = form_mass_u( u,  w ) + imp_weight*form_function_u( g, f, h,  u,  w )
rhs_u = form_mass_u( un, w ) - exp_weight*form_function_u( g, f, hn, un, w )

lhs = lhs_h + lhs_u
rhs = rhs_h + rhs_u

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

outfile = fd.File( "williamson2.pvd" )
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
        print( "saving solution at timestep ", timestep, " and time: ", t/hour, " hours" )
        u_out.assign(un)
        h_out.assign(hn)
        outfile.write( u_out, h_out )


### === --- finish up --- === ###

u_out.assign(un)
h_out.assign(hn)
outfile.write( u_out, h_out )

