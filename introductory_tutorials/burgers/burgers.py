
import firedrake as fd

def write_velocity_file(
        name : str,
        func : fd.Function,
        space : fd.VectorFunctionSpace ):
    fd.File(name+".pvd").write(fd.project(func,space,name="velocity"))

def burgers_spatial_residual( u  : fd.Function,
                              v  : fd.Function,
                              nu : float ):
    if u.function_space() != v.function_space() :
        raise ValueError( "function 'u' and testfunction 'v' should belong to the same fd.FunctionSpace" )

    return (  fd.inner( fd.dot(u, fd.nabla_grad(u) ), v )
            + nu*fd.inner( fd.grad(u), fd.grad(v) )
           )*fd.dx

def BDF1_integration( u0  : fd.Function,
                      u1  : fd.Function,
                      v   : fd.Function,
                      dt  : float ):
    if u0.function_space() != u1.function_space() :
        raise ValueError( "functions 'u0' and 'u1' should belong to the same fd.FunctionSpace" )
    if u0.function_space() !=  v.function_space() :
        raise ValueError( "function 'u0' and testfunction 'v' should belong to the same fd.FunctionSpace" )
    return fd.inner((u1-u0)/dt, v )*fd.dx
    
def BDF2_integration( u0  : fd.Function,
                      u1  : fd.Function,
                      u2  : fd.Function,
                      v   : fd.Function,
                      dt  : float ):
    if ( u0.function_space() != u1.function_space() ) and \
       ( u0.function_space() != u2.function_space() ) :
        raise ValueError( "functions 'u0' and 'u1' and 'u2' should belong to the same fd.FunctionSpace" )
    if u0.function_space() !=  v.function_space() :
        raise ValueError( "function 'u0' and testfunction 'v' should belong to the same fd.FunctionSpace" )
    return fd.inner((3*u2 - 4*u1 + u0)/(2*dt), v )*fd.dx

### === --- inputs --- === ###

# number of mesh points in each direction
nx = 30

# approximate cfl number
cfl = 1.0

# order of finite elements
order = 3

# final time
T = 10

# reynolds number
reynolds = 10000.

# background velocity and variation
uref = 0.2
du = 0.8

# frequency to save at
save_freq=4

### === --- set up --- === ###

# parameters
dx = 1./nx
nu = 1./reynolds

dt_c = cfl*dx/(du+uref)
dt_v = cfl*dx*dx/nu
pe = (du+uref)*dx/nu

dt =min(dt_c,dt_v)

print("cfl_c = ", dt_c )
print("cfl_v = ", dt_v )
print("pec#  = ", pe )
print("dt = ", dt)

# domain
mesh = fd.UnitSquareMesh( nx,nx, quadrilateral=True )

# vector spaces for solution and output
V = fd.VectorFunctionSpace( mesh, "CG", order )
Vout = fd.VectorFunctionSpace( mesh, "CG", 1 )

# solutions at current and next timestep
ulast = fd.Function( V, name="ulast" )
ucurr = fd.Function( V, name="ucurr" )
unext = fd.Function( V, name="unext" )

v = fd.TestFunction(V)

# initial conditions
x = fd.SpatialCoordinate( mesh )
u0 = fd.project( fd.as_vector([ uref + du*fd.sin( fd.pi*x[0] ), 0 ]), V )

write_velocity_file( "solution.initial", u0, Vout )

ulast.assign(u0)
ucurr.assign(u0)
unext.assign(u0)

# set up residual

# spatial residual
R = burgers_spatial_residual( unext, v, nu )

# timestepping
#I = BDF1_integration(      ucurr,unext,v,dt)
I = BDF2_integration(ulast,ucurr,unext,v,dt)

# full residual
F=I+R

# inflow boundary condition
inflow_bc = fd.DirichletBC( V, uref, 1 )

### === --- time integration --- === ###

outfile = fd.File("solution.pvd")
outfile.write( fd.project( unext, Vout, name="velocity" ))

t=0
i=0
while( t <= T ):
    fd.solve( F==0, unext, bcs=[inflow_bc] )
    ulast.assign(ucurr)
    ucurr.assign(unext)
    t += dt
    i += 1
    if( i%save_freq == 0 ):
        outfile.write( fd.project( unext, Vout, name="velocity" ))

