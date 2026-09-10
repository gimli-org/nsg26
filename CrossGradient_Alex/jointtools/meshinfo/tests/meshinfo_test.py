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
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import pygimli as pg
import nsg26.CrossGradient_Alex.jointtools.meshinfo.meshinfo as mI

# %%
circle = pg.meshtools.createCircle(
    pos=[0, 0], radius=1, area=0.1, nSegments=100)

mesh = pg.meshtools.createMesh(circle, area=0.01)
pg.show(mesh)
cell = mesh.cells()[3]
nodes = cell.nodes()

# %%
neighbour_function = None
neighbour_function = partial(mI.distance_to_neighbour_list_for_cell, mesh = mesh, dist=0.2)
mi = mI.MeshInfo(mesh, neighbour_function=neighbour_function, initialise_gn2=True)
# mi = mI.MeshInfo(mesh, neighbour_function=neighbour_function, initialise_gn2=True)
min_cell_neighbours = np.min([len(cni.neighbour_cells) for cni in mi.cell_neighbour_info])
print(f"The minimum number of neighbour cells is: {min_cell_neighbours}")

# %%
a = mi.cell_neighbour_info[0]
b = mi.cell_neighbour_info[1]

# %%
print(a.cell_center)
print(b.cell_center)

# %%
markersize = 4
px = 0.5
py = 0.1
cell_id_to_show_neighbours = mesh.findCell([px, py]).id()
cell = mesh.cells()[cell_id_to_show_neighbours]
fig, ax = plt.subplots(1, 1)
pg.show(mesh, ax=ax)
ax.plot(cell.center()[0], cell.center()[1], 'ro', markersize=markersize)
ax.hlines(py, -1, 1, linewidth=0.5, color='orange')
ax.vlines(px, -1, 1, linewidth=0.5, color='orange')
for n in mi.cell_neighbour_info[cell_id_to_show_neighbours].neighbour_cells:
    n_cell = mesh.cells()[n]
    ax.plot(n_cell.center()[0], n_cell.center()[1], 'bo', markersize=markersize)

# %%
index_vector_gn1, sensitivity_matrix_gn1 = mi.cell_neighbour_info[cell_id_to_show_neighbours].get_gradient_mesh_sensitivities()
index_vector_gn2, sensitivity_matrix_gn2 = mi.cell_neighbour_info[cell_id_to_show_neighbours].get_gradient_mesh_sensitivities(order=2)
# 
full_sensitivity_matrix_gn1 = np.zeros((2, len(mesh.cells())))
full_sensitivity_matrix_gn1[:, index_vector_gn1] = sensitivity_matrix_gn1

full_sensitivity_matrix_gn2 = np.zeros((2, len(mesh.cells())))
full_sensitivity_matrix_gn2[:, index_vector_gn2] = sensitivity_matrix_gn2

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

cmax = np.max(np.abs(full_sensitivity_matrix_gn1))
cmin = -cmax
ax = axs[0,0]
_ = pg.show(mesh, data=full_sensitivity_matrix_gn1[0], label="x", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("GN1 x")
ax = axs[0,1]
_ = pg.show(mesh, data=full_sensitivity_matrix_gn1[1], label="y", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("GN1 y")
ax = axs[1,0]
_ = pg.show(mesh, data=full_sensitivity_matrix_gn2[0], label="x", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("GN2 x")
ax = axs[1,1]
_ = pg.show(mesh, data=full_sensitivity_matrix_gn2[1], label="y", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("GN2 y")


# %%
# Plot entries for hessian matrix
index_vector_hessian, hessian_matrix = mi.cell_neighbour_info[cell_id_to_show_neighbours].get_hessian_mesh_sensitivities()

full_hessian_matrix = np.zeros((3, len(mesh.cells())))
full_hessian_matrix[:, index_vector_hessian] = hessian_matrix

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

cmax = np.max(np.abs(full_hessian_matrix[0]))
cmin = -cmax
ax = axs[0]
_ = pg.show(mesh, data=full_hessian_matrix[0], label="xx", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("Hessian xx")
ax = axs[1]
_ = pg.show(mesh, data=full_hessian_matrix[1], label="xy", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("Hessian xy")
ax = axs[2]
_ = pg.show(mesh, data=full_hessian_matrix[2], label="yy", ax=ax, cMin=cmin, cMax=cmax, cMap="seismic"), ax.set_title("Hessian yy")

# %%
