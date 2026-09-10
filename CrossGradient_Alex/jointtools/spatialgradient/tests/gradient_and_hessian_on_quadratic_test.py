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

# mesh = pg.meshtools.createMesh(circle, area=0.01)
mesh = pg.meshtools.createMesh(circle, area=0.001) # to investigate meshing effects
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
neighbour_function = partial(mI.meshinfo.get_n_closest_neighbours, n=8, mesh=mesh)
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
logarithmic_model = np.log(1+cell_centers[:,0]) * np.log(1+cell_centers[:,1])
pg.show(mesh, logarithmic_model, label="Logarithmic model")

# %%
Q_alpha = 1
Q_full = Q_alpha * np.array([[1,0], [0,42]])

G_alpha = 20
G_full = G_alpha*np.array([1,1])

N_alpha = 3*3
# nonlinear term is sin(4*(x_0 + x_1))
# derivative is 4*cos(4*(x_0+x_1) in both components
# 2nd derivative thus is the same aswell as 16 
# derivative thus is 4 * cos([x_0 x_1])

model_for_hessian_test = np.array(
    [
    cell_center @ Q_full @cell_center +
    G_full@cell_center +
    N_alpha*np.sin(4*np.sum(cell_center))
    for cell_center in cell_centers
    ])

gradient_analytic = 2 * cell_centers @ Q_full + G_full + N_alpha * 4 * np.tile(np.cos(4*np.sum(cell_centers, axis=1)), reps=(2,1)).T

hessian_matrix_analytic = np.array([
    Q_full - N_alpha * 16 * np.sin(4*np.sum(cell_center))
    for cell_center in cell_centers
    ])

norm_of_hessian_matrix_analytic = np.array([
    np.linalg.norm(hess_matrix_cell)
    for hess_matrix_cell in hessian_matrix_analytic
]
)

laplacian_analytic = np.array(
    [np.sum(np.diag(hess_matrix_cell))
    for hess_matrix_cell in hessian_matrix_analytic
    ]
)

evev_analytic = [np.linalg.eig(hess_matrix_cell) for hess_matrix_cell in hessian_matrix_analytic]
eigenvalues_analytic = np.array([np.array([np.max(ev[0]), np.min(ev[0])]) for ev in evev_analytic])
eigenvectors_analytic = np.array([ev[1][:,0]for ev in evev_analytic])
eigenvalues_max_analytic = eigenvalues_analytic[:,0]
eigenvalues_min_analytic = eigenvalues_analytic[:,1]

fig, axs = plt.subplots(1,4, layout="constrained", figsize=(20,5), sharex=True, sharey=True)
ax = axs[0]
_ = pg.show(mesh, model_for_hessian_test, ax=ax), ax.set_title("Model")
ax = axs[1]
_ = pg.show(mesh, gradient_analytic, ax=ax), ax.set_title("Gradient")
ax = axs[2]
_ = pg.show(mesh, norm_of_hessian_matrix_analytic, ax=ax), ax.set_title("Norm of Hessian")
ax = axs[3]
_ = pg.show(mesh, laplacian_analytic, ax=ax), ax.set_title("Laplacian")

fig, axs = plt.subplots(1,4, layout="constrained", figsize=(20,5), sharex=True, sharey=True)
ax = axs[0]
_ = pg.show(mesh, model_for_hessian_test, ax=ax), ax.set_title("Model")
ax = axs[1]
_ = pg.show(mesh, eigenvectors_analytic, ax=ax), ax.set_title("Eigenvectors for Hessian big")
ax = axs[2]
_ = pg.show(mesh, eigenvalues_max_analytic, ax=ax), ax.set_title("Maximum eigenvalue of Hessian")
ax = axs[3]
_ = pg.show(mesh, eigenvalues_min_analytic, ax=ax), ax.set_title("Minimum eigenvalue of Hessian")


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
gradient_difference_12_norm = np.linalg.norm(gradient_difference_12, axis=1) / np.linalg.norm(gradient_1, axis=1)

gradient_difference_1_to_analytic = gradient_1 - gradient_analytic
gradient_difference_1_to_analytic_norm = np.linalg.norm(gradient_difference_1_to_analytic, axis=1)/np.linalg.norm(gradient_analytic)

gradient_difference_2_to_analytic = gradient_2 - gradient_analytic
gradient_difference_2_to_analytic_norm = np.linalg.norm(gradient_difference_2_to_analytic, axis=1)/np.linalg.norm(gradient_analytic)

hessian_difference_to_analytic = hessian_matrix_numeric - hessian_matrix_analytic
hessian_difference_to_analytic_norm = np.array([np.linalg.norm(hess_diff)/np.linalg.norm(hessian_matrix_analytic) for hess_diff in hessian_difference_to_analytic])

laplacian_difference_to_analytic = laplacian_numeric - laplacian_analytic
laplacian_difference_to_analytic_norm = laplacian_difference_to_analytic / np.abs(laplacian_analytic)

# Eigenvalue and -vector analysis
evev_numeric = [np.linalg.eig(hess_matrix_cell) for hess_matrix_cell in hessian_matrix_numeric]
eigenvalues_numeric = np.array([np.array([np.max(ev[0]), np.min(ev[0])]) for ev in evev_numeric])
eigenvectors_numeric = np.array([ev[1][:,0]for ev in evev_numeric])
eigenvalues_max_numeric = eigenvalues_numeric[:,0]
eigenvalues_min_numeric = eigenvalues_numeric[:,1]

angular_difference_eigenvector = np.array([np.arccos(np.dot(tup[0],tup[1])) for tup in zip(eigenvectors_analytic, eigenvectors_numeric)])


# %%
# Plot gradient analytic, gradient 1, gradient 2
# Plot difference gradient 1 to analytic, gradient 2 to analytic, gradient 1 to gradient 

vector=False
fig, axs = plt.subplots(2, 3, layout="tight", figsize=(10, 15))

# Absolute plots
cMin = np.min([
    np.min(np.linalg.norm(gradient_analytic, axis=1)),
    np.min(np.linalg.norm(gradient_1, axis=1)),
    np.min(np.linalg.norm(gradient_2, axis=1))
    ])
cMax = np.max([
    np.max(np.linalg.norm(gradient_analytic, axis=1)),
    np.max(np.linalg.norm(gradient_1, axis=1)),
    np.max(np.linalg.norm(gradient_2, axis=1))
    ])
cMap="turbo"

if vector:
    _=pg.show(mesh, gradient_analytic, ax=axs[0,0]), axs[0,0].set_title("Gradient analytic")
    _=pg.show(mesh, gradient_1, ax=axs[0,1]), axs[0,1].set_title("Gradient Taylor order 1")
    _=pg.show(mesh, gradient_2, ax=axs[0,2]), axs[0,2].set_title("Gradient Taylor order 2")
else:
    _=pg.show(mesh, np.linalg.norm(gradient_analytic, axis=1), ax=axs[0,0], cMin=cMin, cMax=cMax, cMap=cMap), axs[0,0].set_title("Gradient analytic")
    _=pg.show(mesh, np.linalg.norm(gradient_1, axis=1), ax=axs[0,1], cMin=cMin, cMax=cMax, cMap=cMap), axs[0,1].set_title("Gradient Taylor order 1")
    _=pg.show(mesh, np.linalg.norm(gradient_2, axis=1), ax=axs[0,2], cMin=cMin, cMax=cMax, cMap=cMap), axs[0,2].set_title("Gradient Taylor order 2")

# Difference plots
# Absolute plots

cMap="seismic"

if vector:
    # Difference of vector fields should idealy look "random"
    _=pg.show(mesh, gradient_difference_12, ax=axs[1,0]), axs[1,0].set_title("Difference of Taylor order 1/2")
    _=pg.show(mesh, gradient_difference_1_to_analytic, ax=axs[1,1]), axs[1,1].set_title("Difference Taylor 1 to analytic")
    _=pg.show(mesh, gradient_difference_2_to_analytic, ax=axs[1,2]), axs[1,2].set_title("Difference Taylor 2 to analytic")
else:
    cMin = np.min([
    np.min(gradient_difference_12_norm),
    ])
    cMax = np.max([
        np.max(gradient_difference_12_norm),
        ])
    cMax = np.max([np.abs(cMin), np.abs(cMax)]) * 1e-1
    cMin = -cMax

    _=pg.show(mesh, gradient_difference_12_norm, ax=axs[1,0], cMin=cMin, cMax=cMax, cMap=cMap), axs[1,0].set_title("Difference of Taylor order 1/2")

    cMin = np.min([
    np.min(gradient_difference_1_to_analytic_norm),
    np.min(gradient_difference_2_to_analytic_norm)
    ])
    cMax = np.max([
        np.max(gradient_difference_1_to_analytic_norm),
        np.max(gradient_difference_2_to_analytic_norm)
    ])
    cMax = np.max([np.abs(cMin), np.abs(cMax)])
    cMin = -cMax

    _=pg.show(mesh, gradient_difference_1_to_analytic_norm, ax=axs[1,1], cMin=cMin, cMax=cMax, cMap=cMap), axs[1,1].set_title("Difference Taylor 1 to analytic")
    _=pg.show(mesh, gradient_difference_2_to_analytic_norm, ax=axs[1,2], cMin=cMin, cMax=cMax, cMap=cMap), axs[1,2].set_title("Difference Taylor 2 to analytic")

# %%
# Plot norm of hessian analytic, norm of hessian_numeric
# Plot laplacian, laplacian_analytic
# Plot difference norm hessian to analytic, laplacian to analytic
# Plot laplacian_analytic, difference norm hessian to analytic, laplacian to analytic

fig, axs = plt.subplots(2, 3, layout="tight", figsize=(10, 15))

# Hessian norm plots
cMin = np.min(
    [
    np.min(norm_of_hessian_matrix_analytic),
    np.min(norm_of_hessin_matrix_numeric),
    ]
)
cMax = np.max(
    [
    np.max(norm_of_hessian_matrix_analytic),
    np.max(norm_of_hessin_matrix_numeric),
    ]
)
cMap="turbo"
ax = axs[0,0]
_=pg.show(mesh, norm_of_hessian_matrix_analytic, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Norm of Hessian analytic")
ax = axs[0,1]
_=pg.show(mesh, norm_of_hessin_matrix_numeric, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Norm of Hessian numeric")


# Lapliacian norm plots
cMin = np.min(
    [
    np.min(laplacian_analytic),
    np.min(laplacian_numeric),
    ]
)
cMax = np.max(
    [
    np.max(laplacian_analytic),
    np.max(laplacian_numeric),
    ]
)
cMap="turbo"
ax = axs[1,0]
_=pg.show(mesh, laplacian_analytic, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Laplacian analytic")
ax = axs[1,1]
_=pg.show(mesh, laplacian_numeric, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Laplacian numeric")

# Difference plots
cMap="seismic"

cMin = np.min([
    np.min(hessian_difference_to_analytic_norm),
    ])
cMax = np.max([
    np.max(hessian_difference_to_analytic_norm),
    ])
cMax = np.max([np.abs(cMin), np.abs(cMax)]) * 1e0
cMin = -cMax
ax = axs[0,2]
_=pg.show(mesh, hessian_difference_to_analytic_norm, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Difference Norm Hessian")

cMin = np.min([
    np.min(laplacian_difference_to_analytic_norm),
    ])
cMax = np.max([
    np.max(laplacian_difference_to_analytic_norm),
    ])
cMax = np.max([np.abs(cMin), np.abs(cMax)]) * 1e-2
cMin = -cMax
ax = axs[1,2]
_=pg.show(mesh, laplacian_difference_to_analytic_norm, ax=ax, cMin=cMin, cMax=cMax, cMap=cMap), ax.set_title("Difference Laplacian")

# %%
# Eigenvector plots
# Plot Eigenvectors and their angular difference
# Plot Maximum eigenvalue and their difference
# Plot Minumum eigenvalue and their difference

# Eigenvectors
fig, axs = plt.subplots(3, 3, layout="tight", figsize=(15, 15))
ax = axs[0,0]
_ = pg.show(mesh, eigenvectors_analytic, ax=ax), ax.set_title("Eigenvector max eigenval analytic")

ax = axs[1,0]
_ = pg.show(mesh, eigenvectors_numeric, ax=ax), ax.set_title("Eigenvector max eigenval numeric")

ax = axs[2,0]
c = np.max(np.abs(angular_difference_eigenvector))
_ = pg.show(mesh, angular_difference_eigenvector%np.pi, ax=ax, cMin=-c, cMax=c, cMap="seismic"), ax.set_title("Angle between max eigenvector in radians")

# Maximum eigenvalue
cmin = np.min(
    [
        np.min(eigenvalues_max_analytic),
        np.min(eigenvalues_max_numeric)
    ]
)
cmax = np.max(
    [
        np.max(eigenvalues_max_analytic),
        np.max(eigenvalues_max_numeric)
    ]
)
ax = axs[0,1]
_ = pg.show(mesh, eigenvalues_max_analytic, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Maximum eigenvalue analytic")

ax = axs[1,1]
_ = pg.show(mesh, eigenvalues_max_numeric, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Maximum eigenvalue numeric")

ax = axs[2,1]
relative_difference_max_eigenvalue = (eigenvalues_max_analytic-eigenvalues_max_numeric)/np.abs(eigenvalues_max_analytic)
c = np.max(np.abs(relative_difference_max_eigenvalue))
_ = pg.show(mesh, relative_difference_max_eigenvalue, ax=ax, cMin=-c, cMax=c, cMap="seismic"), ax.set_title("Rel. difference maximum eigenvalue analytic")

# Minimum eigenvalue
cmin = np.min(
    [
        np.min(eigenvalues_min_analytic),
        np.min(eigenvalues_min_numeric)
    ]
)
cmax = np.max(
    [
        np.max(eigenvalues_min_analytic),
        np.max(eigenvalues_min_numeric)
    ]
)
ax = axs[0,2]
_ = pg.show(mesh, eigenvalues_min_analytic, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Minimum eigenvalue analytic")

ax = axs[1,2]
_ = pg.show(mesh, eigenvalues_min_numeric, ax=ax, cMin=cmin, cMax=cmax, cMap="turbo"), ax.set_title("Minimum eigenvalue numeric")

ax = axs[2,2]
relative_difference_min_eigenvalue = (eigenvalues_min_analytic-eigenvalues_min_numeric)/np.abs(eigenvalues_min_analytic)
_ = pg.show(mesh, relative_difference_min_eigenvalue, cMin=-c, cMax=c, cMap="seismic", ax=ax), ax.set_title("Rel. difference minimum eigenvalue analytic")

# %%
# Edge detection - gradient vs laplacian

fig, axs = plt.subplots(1, 3, layout="constrained", figsize = (15,5))

ax = axs[0]
pg.show(mesh, model_for_hessian_test, ax=ax)

ax = axs[1]
pg.show(mesh, np.linalg.norm(gradient_analytic, axis=1), ax=ax, cMin=0, cMap="turbo")

ax = axs[2]
c = np.max(np.abs(laplacian_analytic))
pg.show(mesh, laplacian_analytic, ax=ax, cMin=-c, cMax=c, cMap="seismic")
