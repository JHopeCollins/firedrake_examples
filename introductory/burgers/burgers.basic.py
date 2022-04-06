
import firedrake as fd

def write_velocity_file(
        name : str,
        func : fd.Function,
        space : fd.VectorFunctionSpace ):
    fd.File(name+".pvd").write(fd.project(func,space,name="velocity"))

### === --- inputs --- === ###

# number of mesh points in each direction
nx = 30

# approximate cfl number
cfl = 1.

# order of finite elements
order = 2

# final time
T = 1

# reynolds number
reynolds = 1000.

### === --- set up --- === ###

# parameters
nu = 1./reynolds
dt = cfl/nx

# domain
mesh = fd.UnitSquareMesh( nx,nx )

# vector spaces for solution and output
V = fd.VectorFunctionSpace( mesh, "CG", order )
Vout = fd.VectorFunctionSpace( mesh, "CG", 1 )

# solutions at current and next timestep
ucurr = fd.Function( V, name="ucurr" )
unext = fd.Function( V, name="unext" )

v = fd.TestFunction(V)

# initial conditions
x = fd.SpatialCoordinate( mesh )
u0 = fd.project( fd.as_vector([ fd.sin(fd.pi*x[0]), 0 ]), V )

write_velocity_file( "solution.initial", u0, Vout )

ucurr.assign(u0)
unext.assign(u0)

# set up residual
F = (     fd.inner((unext-ucurr)/dt, v )
     +    fd.inner( fd.dot(unext, fd.nabla_grad(unext) ), v )
     + nu*fd.inner( fd.grad(unext), fd.grad(v) )
    )*fd.dx

### === --- time integration --- === ###

outfile = fd.File("solution.pvd")
outfile.write( fd.project( unext, Vout, name="velocity" ))

t=0
while( t <= T ):
    fd.solve( F==0, unext )
    ucurr.assign(unext)
    t += dt
    if out : outfile.write( fd.project( unext, Vout, name="velocity" ))

