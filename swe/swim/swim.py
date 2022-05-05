
import firedrake as fd
from petsc4py import PETSc

import firedrake_utils.mg as mg
import firedrake_utils.planets.earth as earth
import firedrake_utils.shallow_water.williamson1992.case5 as case5

import argparse


# set up mesh levels for multigrid scheme
def high_order_mesh_hierarchy(mh, degree, R0):
    meshes = []
    for m in mh:
        X = fd.VectorFunctionSpace(m, "Lagrange", degree)
        new_coords = fd.interpolate(m.coordinates, X)
        x, y, z = new_coords
        r = (x**2 + y**2 + z**2)**0.5
        new_coords.assign(R0*new_coords/r)
        new_mesh = fd.Mesh(new_coords)
        meshes.append(new_mesh)

    return fd.HierarchyBase(meshes, mh.coarse_to_fine_cells,
                            mh.fine_to_coarse_cells,
                            mh.refinements_per_level, mh.nested)


def mg_mesh(R0, base_level, degree, distribution_parameters, nrefs):
    basemesh = fd.IcosahedralSphereMesh(
                    radius=R0,
                    refinement_level=base_level,
                    degree=degree,
                    distribution_parameters=distribution_parameters)
    del basemesh._radius
    mh = fd.MeshHierarchy(basemesh, nrefs)
    mh = high_order_mesh_hierarchy(mh, degree, R0)
    for mesh in mh:
        xf = mesh.coordinates
        mesh.transfer_coordinates = fd.Function(xf)
        x = fd.SpatialCoordinate(mesh)
        r = (x[0]**2 + x[1]**2 + x[2]**2)**0.5
        xf.interpolate(R0*xf/r)
        mesh.init_cell_orientations(x)
    mesh = mh[-1]
    return mesh


def form_mass(u, h, v, q):
    return fd.inner(u, v)*fd.dx + h*q*fd.dx


def form_velocity(mesh, g, b, f, u, h, v):
    n = fd.FacetNormal(mesh)
    outward_normals = fd.CellNormal(mesh)

    def perp(u):
        return fd.cross(outward_normals, u)

    def both(u):
        return 2*fd.avg(u)

    K = 0.5*fd.inner(u, u)
    upwind = 0.5 * (fd.sign(fd.dot(u, n)) + 1)

    return (fd.inner(v, f*perp(u))*fd.dx
            - fd.inner(perp(fd.grad(fd.inner(v, perp(u)))), u)*fd.dx
            + fd.inner(both(perp(n)*fd.inner(v, perp(u))),
                       both(upwind*u))*fd.dS
            - fd.div(v)*(g*(h + b) + K)*fd.dx)


def form_depth(mesh, g, b, u, h, q):
    n = fd.FacetNormal(mesh)
    uup = 0.5 * (fd.dot(u, n) + abs(fd.dot(u, n)))

    return (- fd.inner(fd.grad(q), u)*h*fd.dx
            + fd.jump(q)*(uup('+')*h('+')
                        - uup('-')*h('-'))*fd.dS)


def form_function(mesh, g, b, f, u, h, v, q):
    return form_velocity(mesh, g, b, f, u, h, v) \
           + form_depth(mesh, g, b, u, h, q)


PETSc.Sys.popErrorHandler()

# get command arguments
parser = argparse.ArgumentParser(description='Williamson 5 testcase for monolithic fully implicit solver.')  # noqa: E501
parser.add_argument('--base_level', type=int, default=1, help='Base refinement level of icosahedral grid for MG solve. Default 1.')  # noqa: E501
parser.add_argument('--ref_level', type=int, default=3, help='Refinement level of icosahedral grid. Default 3.')  # noqa: E501
parser.add_argument('--nsteps', type=int, default=10, help='Number of timesteps. Default 10.')  # noqa: E501
parser.add_argument('--dt', type=float, default=0.05, help='Timestep in hours. Default 0.05.')  # noqa: E501
parser.add_argument('--filename', type=str, default='output/w5diag')  # noqa: E501
parser.add_argument('--coords_degree', type=int, default=1, help='Degree of polynomials for sphere mesh approximation.')  # noqa: E501
parser.add_argument('--degree', type=int, default=1, help='Degree of finite element space (the DG space).')  # noqa: E501
parser.add_argument('--show_args', action='store_true', help='Output all the arguments.')  # noqa: E501

args = parser.parse_known_args()
args = args[0]

if args.show_args:
    PETSc.Sys.Print(args)

# # # === --- mesh set up --- === # # #

R0 = earth.radius
nrefinements = args.ref_level - args.base_level
distribution_parameters = {
    "partition": True,
    "overlap_type": (fd.DistributedMeshOverlapType.VERTEX, 2)
}

mesh = mg_mesh(R0,
               args.base_level,
               args.coords_degree,
               distribution_parameters,
               nrefinements)

R0 = fd.Constant(R0)
x, y, z = fd.SpatialCoordinate(mesh)

# # # === --- function spaces --- === # # #

degree = args.degree
V1 = fd.FunctionSpace(mesh, "BDM", degree+1)
V2 = fd.FunctionSpace(mesh, "DG", degree)
V0 = fd.FunctionSpace(mesh, "CG", degree+2)
W = fd.MixedFunctionSpace((V1, V2))

# # # === --- initial conditions --- === # # #

# constants
Omega = earth.Omega  # rotation rate
g = earth.Gravity
H0 = case5.H0  # reference free surface height

# H0: reference free surface height
# H0 + eta: actual free surface height
# b: topology (ground surface height)
# h = (H0+eta)-b: depth

# coriolis expression
f = case5.coriolis_expression(x, y, z)

# initial velocity
u0 = case5.velocity_function(x, y, z, V1, name="Initial Velocity")

# topography
b = case5.topography_function(x, y, z, V2, name="Topography")

# initial surface height perturbation
eta0 = fd.Function(V2, name='Initial Elevation')
eta0.project(case5.depth_expression(x, y, z) - H0)

# initial depth field
h0 = fd.Function(V2, name='Initial Depth').assign(eta0 + H0 - b)

# write initial data
outfile = fd.File(args.filename+'.pvd')
eta_out = fd.Function(V2, name='Elevation').assign(eta0)
u_out = fd.Function(V1, name='Velocity').assign(u0)
outfile.write(u_out, eta_out)

# parameters for the implicit solve

# sparameters = {
#     # "snes_monitor": None,
#     "mat_type": "matfree",
#     "ksp_type": "fgmres",
#     # "ksp_monitor": None,
#     # "ksp_monitor_true_residual": None,
#     # "ksp_converged_reason": None,
#     "ksp_atol": 1e-8,
#     "ksp_rtol": 1e-8,
#     "ksp_max_it": 400,
#     "pc_type": "mg",
#     "pc_mg_cycle_type": "v",
#     "pc_mg_type": "multiplicative",
#     "mg_levels_ksp_type": "gmres",
#     "mg_levels_ksp_max_it": 5,
#     # "mg_levels_ksp_convergence_test": "skip",
#     "mg_levels_pc_type": "python",
#     "mg_levels_pc_python_type": "firedrake.PatchPC",
#     "mg_levels_patch_pc_patch_save_operators": True,
#     "mg_levels_patch_pc_patch_partition_of_unity": True,
#     "mg_levels_patch_pc_patch_sub_mat_type": "seqdense",
#     "mg_levels_patch_pc_patch_construct_codim": 0,
#     "mg_levels_patch_pc_patch_construct_type": "vanka",
#     "mg_levels_patch_pc_patch_local_type": "additive",
#     "mg_levels_patch_pc_patch_precompute_element_tensors": True,
#     "mg_levels_patch_pc_patch_symmetrise_sweep": False,
#     "mg_levels_patch_sub_ksp_type": "preonly",
#     "mg_levels_patch_sub_pc_type": "lu",
#     "mg_levels_patch_sub_pc_factor_shift_type": "nonzero",
#     "mg_coarse_pc_type": "python",
#     "mg_coarse_pc_python_type": "firedrake.AssembledPC",
#     "mg_coarse_assembled_pc_type": "lu",
#     "mg_coarse_assembled_pc_factor_mat_solver_type": "mumps",
# }
#
# theta = 0.5
#
# # mesh transfer operators
# vtransfer = mg.ManifoldTransfer()
# transfers = {
#     V1.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
#                        vtransfer.inject),
#     V2.ufl_element(): (vtransfer.prolong, vtransfer.restrict,
#                        vtransfer.inject)
# }
# transfer_manager = fd.TransferManager(native_transfers=transfers)
#
