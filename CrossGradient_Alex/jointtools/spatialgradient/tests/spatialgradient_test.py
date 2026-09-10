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
import nsg26.CrossGradient_Alex.jointtools.spatialgradient.spatialgradient as sG

# %%
circle = pg.meshtools.createCircle(
pos=[0, 0], radius=1, area=0.1, nSegments=100)

mesh = pg.meshtools.createMesh(circle)
fig, ax = plt.subplots(1)
pg.show(mesh, ax=ax)

# Create a model
cell_centers = np.array([np.array(cell.center())[0:2] for cell in mesh.cells()])
gradient_direction = np.array([-1, 1])/(np.sqrt(2))
variation = 100
offset = 700
model = variation * (cell_centers @ gradient_direction) + offset
pg.show(mesh, data=model, label="Model", ax=ax)

# %%
# Calculate the spatial gradients
mi = mI.MeshInfo(
    mesh=mesh
)

# %%
gradient = sG.calculate_spatial_gradient(
    model=model,
    mesh_info=mi,
)

# %%
fig, ax = sG.plot_gradient_field(
    spatial_gradient=gradient,
    mesh=mesh,
    scale=1e3,
    show_mesh=True,
    figsize=(5,5)
)

# %%
fig, ax = sG.plot_absolute_value_of_gradient_field_from_vectors(
    spatial_gradient=gradient,
    mesh=mesh,
    show_mesh=True,
    figsize=(5,5)
)
#%%
pg.show(mesh,markers=True)
# %%
