# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: pg
#     language: python
#     name: python3
# ---

# %%
import sys
sys.path.append("../..")
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pygimli as pg
import nsg26.CrossGradient_Alex.jointtools.meshinfo as mI
import modelinfo as modI
import nsg26.CrossGradient_Alex.jointtools.spatialgradient.spatialgradient as sG

# %%
circle = pg.meshtools.createCircle(
pos=[0, 0], radius=1, area=0.1, nSegments=100)

mesh = pg.meshtools.createMesh(circle, area=0.01)
# mesh = pg.meshtools.createMesh(circle, area=0.001) # to investigate meshing effects
# mesh = pg.meshtools.createMesh(circle, area=0.0005) # to investigate meshing effects
fig, ax = plt.subplots(1)
pg.show(mesh, ax=ax)

# Create a model
cell_centers = np.array([np.array(cell.center())[0:2] for cell in mesh.cells()])

# Calculate the spatial gradients
# distance for which quadratic works but non linear not - dist = .2
dist = .05
# dist = .032
neighbour_function = partial(mI.meshinfo.distance_to_neighbour_list_for_cell, dist=dist, mesh=mesh)
neighbour_function = partial(mI.meshinfo.get_n_closest_neighbours, n=15, mesh=mesh)
# neighbour_function = None
mi = mI.MeshInfo(
    mesh=mesh,
    initialise_gn2=True,
    neighbour_function=neighbour_function
)

# %%
no_of_neighbour_cells = [len(cni.neighbour_cells) for cni in mi.cell_neighbour_info]
print(f"Minimum number of neighbour cells: {np.min(no_of_neighbour_cells)} - Maximum number of neighbour cells: {np.max(no_of_neighbour_cells)}")

# %%
alpha = 10
a = 0
beta = 10
b = 0
# The non linear function is given as a jump function with a superimposed gradient
# For x<= we have f(x) = 100 and for x>0 we have f(x) = 200
# On this we add a highly non linear function
# Gradients and Hessian are only calculated numerically

# The hard edge is a good example for which the laplacian is not working 
# since the edge is not smooth and to sharp for the mesh/laplacian to cover
# model_for_hessian_test = np.where(cell_centers[:,0] <= 0, 100, 200)

# alternatively model the step function as a smooth logistic function
# these edges can now be resolved by the laplacian
# double edged
model_for_hessian_test = 100 + 100/(1+np.exp(-alpha*(cell_centers[:,0]-a))) + 100/(1+np.exp(-beta*(cell_centers[:,1]-b)))

# single edged
# model_for_hessian_test = 100 + 100/(1+np.exp(-alpha*(cell_centers[:,0]-a))) + 100/(1+np.exp(-beta*(cell_centers[:,0]-b)))

# Multiplier 10 is a good examples for which the laplacian is not working
# i.e. introducing "wrong" edges -> this phenomenon is present in the undisturbed
# model as well
add_on = 0*np.sin(10*cell_centers[:,0])*np.sin(10*cell_centers[:,1])
model_for_hessian_test = model_for_hessian_test + add_on

fig, axs = plt.subplots(1, layout="constrained", figsize=(20,5), sharex=True, sharey=True)
ax = axs
_ = pg.show(mesh, model_for_hessian_test, ax=ax), ax.set_title("Model")


# %%
hessian_matrix_numeric = sG.calculate_hessian_matrix(
    model = model_for_hessian_test,
    mesh_info=mi
)
gradient_1 = sG.calculate_spatial_gradient(
    model=model_for_hessian_test,
    mesh_info=mi
)

gradient_2 = sG.calculate_spatial_gradient(
    model=model_for_hessian_test,
    mesh_info=mi,
    taylor_order=2
)

norm_of_hessin_matrix_numeric = np.array(
    [np.linalg.norm(hessian_matrix_numeric_cell)
    for hessian_matrix_numeric_cell in hessian_matrix_numeric]
    )
laplacian_numeric = sG.calculate_laplacian_from_hessian_matrix_model(hessian_matrix_numeric)

gradient_difference_12 = gradient_1-gradient_2
gradient_difference_12_norm = np.linalg.norm(gradient_difference_12, axis=1) / (np.linalg.norm(gradient_1, axis=1)+1e-5)

# Eigenvalue and -vector analysis
evev_numeric = [np.linalg.eig(hess_matrix_cell) for hess_matrix_cell in hessian_matrix_numeric]
eigenvalues_numeric = np.array([np.array([np.max(ev[0]), np.min(ev[0])]) for ev in evev_numeric])
eigenvectors_numeric = np.array([ev[1][:,0]for ev in evev_numeric])
eigenvalues_max_numeric = eigenvalues_numeric[:,0]
eigenvalues_min_numeric = eigenvalues_numeric[:,1]

fig, ax = sG.plot_hessian_matrix_overview(
    hessian_matrix_list = hessian_matrix_numeric,
    mesh=mesh
)


# %%
# Plot gradient analytic, gradient 1, gradient 2
# Plot difference gradient 1 to analytic, gradient 2 to analytic, gradient 1 to gradient 

vector=False
fig, axs = plt.subplots(1, 3, layout="tight", figsize=(15, 5))

# Absolute plots
cMin = np.min([
    np.min(np.linalg.norm(gradient_1, axis=1)),
    np.min(np.linalg.norm(gradient_2, axis=1))
    ])
cMax = np.max([
    np.max(np.linalg.norm(gradient_1, axis=1)),
    np.max(np.linalg.norm(gradient_2, axis=1))
    ])
cMap="turbo"

if vector:
    _=pg.show(mesh, gradient_1, ax=axs[0]), axs[0].set_title("Gradient Taylor order 1")
    _=pg.show(mesh, gradient_2, ax=axs[1]), axs[1].set_title("Gradient Taylor order 2")
else:
    _=pg.show(mesh, np.linalg.norm(gradient_1, axis=1), ax=axs[0], cMin=cMin, cMax=cMax, cMap=cMap), axs[0].set_title("Gradient Taylor order 1")
    _=pg.show(mesh, np.linalg.norm(gradient_2, axis=1), ax=axs[1], cMin=cMin, cMax=cMax, cMap=cMap), axs[1].set_title("Gradient Taylor order 2")

# Difference plots
# Absolute plots

cMap="seismic"
if vector:
    # Difference of vector fields should idealy look "random"
    _=pg.show(mesh, gradient_difference_12, ax=axs[2]), axs[2].set_title("Difference of Taylor order 1/2")
else:
    cMin = np.min([
    np.min(gradient_difference_12_norm),
    ])
    cMax = np.max([
        np.max(gradient_difference_12_norm),
        ])
    cMax = np.max([np.abs(cMin), np.abs(cMax)]) * 1e-1
    cMin = -cMax

    _=pg.show(mesh, gradient_difference_12_norm, ax=axs[2], cMin=cMin, cMax=cMax, cMap=cMap), axs[2].set_title("Difference of Taylor order 1/2")

# %%
# Plot norm of hessian analytic, norm of hessian_numeric
# Plot laplacian, laplacian_analytic
# Plot difference norm hessian to analytic, laplacian to analytic
# Plot laplacian_analytic, difference norm hessian to analytic, laplacian to analytic

fig, axs = plt.subplots(1, 2, layout="tight", figsize=(10, 15))

# Hessian norm plots
cMin = np.min(
    [
    np.min(norm_of_hessin_matrix_numeric),
    ]
)
cMax = np.max(
    [
    np.max(norm_of_hessin_matrix_numeric),
    ]
)
cMap="turbo"
ax = axs[0]
_=pg.show(mesh, norm_of_hessin_matrix_numeric, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Norm of Hessian numeric")


# Lapliacian norm plots
c = np.max(np.abs(laplacian_numeric))
cMap="seismic"
ax = axs[1]
_=pg.show(mesh, laplacian_numeric, ax=ax, cMin=-c, cMax=c, cMap=cMap), ax.set_title("Laplacian numeric")

# %%
# Eigenvector plots
# Plot Eigenvectors and their angular difference
# Plot Maximum eigenvalue and their difference
# Plot Minumum eigenvalue and their difference

# Eigenvectors
fig, axs = plt.subplots(1, 3, layout="tight", figsize=(15, 5))
ax = axs[0]
_ = pg.show(mesh, eigenvectors_numeric, ax=ax), ax.set_title("Eigenvector max eigenval numeric")
# Maximum eigenvalue
cmin = np.min(
    [
        np.min(eigenvalues_max_numeric)
    ]
)
cmax = np.max(
    [
        np.max(eigenvalues_max_numeric)
    ]
)
ax = axs[1]
_ = pg.show(mesh, eigenvalues_max_numeric, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Maximum eigenvalue numeric")

# Minimum eigenvalue
cmin = np.min(
    [
        np.min(eigenvalues_min_numeric)
    ]
)
cmax = np.max(
    [
        np.max(eigenvalues_min_numeric)
    ]
)
ax = axs[2]
_ = pg.show(mesh, eigenvalues_min_numeric, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Minimum eigenvalue numeric")

# %%
# Edge detection - gradient vs laplacian

fig, axs = plt.subplots(1, 4, layout="constrained", figsize = (20,5))

ax = axs[0]
_=pg.show(mesh, model_for_hessian_test, ax=ax)

ax = axs[1]
_=pg.show(mesh, np.linalg.norm(gradient_1, axis=1), ax=ax, cMin=0, cMap="turbo")

ax = axs[2]
_=pg.show(mesh, np.linalg.norm(gradient_2, axis=1), ax=ax, cMin=0, cMap="turbo")

ax = axs[3]
c = np.max(np.abs(laplacian_numeric))
_=pg.show(mesh, laplacian_numeric, ax=ax, cMin=-c, cMax=c, cMap="seismic")
