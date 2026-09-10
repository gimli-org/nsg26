"""
This module contains the classes to calculate and visualise spatial gradients on 
unstructured meshes.

The module contains the following classes:


The module contains the following functions:
    
    * calculate_spatial_gradient: Calculate the spatial gradient of a model on a mesh.
    * plot_gradient_field: Plot a gradient field on a mesh.
    * plot_absolute_value_of_gradient_field: Plot the absolute value of a gradient field on a mesh.
    * plot_absolute_value_of_gradient_field_from_vectors: Plot the absolute value of a gradient 
    *                                                     field on a mesh.
    
Author: Hagen Söding
Affiliation: ETH Zürich
Email: hagen.soeding@eaps.ethz.ch

"""

import numpy as np
import matplotlib.pyplot as plt
import pygimli as pg

### Functions for calculating the spatial gradient of a model on a mesh
def calculate_spatial_gradient(
    model,
    mesh_info,
    taylor_order=1,
):
    """
    Calculate the spatial gradient of a model on a mesh.
    
    Parameters
    ----------
    model : np.ndarray
    The model for which to calculate the spatial gradient.

    mesh_info : MeshInfo
    The mesh information object containing the mesh and the cell neighbour
    information.

    taylor_order : int, optional
    The order of the Taylor expansion to use for the gradient calculation.
    Can be 1 or 2. The default is 1.
    
    Returns
    -------
    spatial_gradient : np.ndarray
    The spatial gradient of the model.
    
    """
    # Initialize the spatial gradient
    spatial_gradient = np.zeros(
        (len(model), mesh_info.dimension)
    )

    # Calculate the spatial gradient in each cell
    for num, model_value in enumerate(model):
        neighbour_cells = mesh_info.cell_neighbour_info[num].neighbour_cells

        if taylor_order == 1:
            system_matrix = mesh_info.cell_neighbour_info[num].distance_matrix_gn_taylor1
        elif taylor_order == 2:
            system_matrix = mesh_info.cell_neighbour_info[num].distance_matrix_gn_taylor2
        else:
            raise ValueError("taylor_order must be 1 or 2")

        rhs = np.array(model[neighbour_cells]) - model_value

        solution_of_system = system_matrix @ rhs
        spatial_gradient[num] = solution_of_system[:mesh_info.dimension]

    return spatial_gradient

def quadratic_indices_to_vector_indices(mesh_info, indices):
    """
    Function to convert the indices of the quadratic matrix to the indices of the vector.

    Parameters:
    indices (tuple): The indices of the quadratic matrix.

    Returns:
    tuple: The indices of the vector.

    """
    # Get the indices of the vector
    assert len(indices) == 2, "Indices must be a tuple of length 2"
    gradient_offset = mesh_info.dimension
    if indices[0] > indices[1]:
        row, col = indices[1], indices[0]
    else:
        row, col = indices[0], indices[1]
    
    indices_triu = np.triu_indices(mesh_info.dimension)
    triangular_offset = np.where((indices_triu[0] == row) & (indices_triu[1] == col))[0][0]
    return gradient_offset + triangular_offset

def vector_indices_to_quadratic_indices(mesh_info, indices):
    """
    Function to convert the indices of the vector to the indices of the quadratic matrix.

    Parameters:
    indices (int): The indices of the vector.

    Returns:
    tuple: The indices of the quadratic matrix.

    """
    gradient_offset = mesh_info.dimension
    if indices < gradient_offset:
        raise ValueError("Index not in Hessian matrix")
    else:
        indices_triu = np.triu_indices(mesh_info.dimension)
        triangular_offset = indices - gradient_offset
        return (indices_triu[0][triangular_offset], indices_triu[1][triangular_offset])

def calculate_hessian_matrix(model, mesh_info, taylor_order=2):
    assert taylor_order >= 2, "Taylor order must be at least 2 to calculate the hessian matrix"

    # Initialise the Hessian matrix
    hessian_matrix = np.zeros(
        (len(model), mesh_info.dimension, mesh_info.dimension)
    )

    # Calculate the hessian matrix in each cell
    for num, model_value in enumerate(model):
        neighbour_cells = mesh_info.cell_neighbour_info[num].neighbour_cells

        if taylor_order == 2:
            system_matrix = mesh_info.cell_neighbour_info[num].distance_matrix_gn_taylor2
        else:
            raise ValueError("taylor_order must be 1 or 2")

        rhs = np.array(model[neighbour_cells]) - model_value

        solution_of_system = system_matrix @ rhs

        for i in range(mesh_info.dimension):
            for j in range(mesh_info.dimension):
                index = quadratic_indices_to_vector_indices(mesh_info=mesh_info, indices=(i,j))
                hessian_matrix[num, i, j] = solution_of_system[index]

    return hessian_matrix

def calculate_laplacian_from_hessian_matrix_model(hessian_matrix_model):
    # Initialise the laplacian vector
    laplacian = np.zeros(hessian_matrix_model.shape[0])

    for num, hessian_matrix_cell in enumerate(hessian_matrix_model):
        laplacian[num] = np.sum(np.diag(hessian_matrix_cell))

    return laplacian

### Functions to visualise gradient fields on a mesh
def plot_gradient_field(
    spatial_gradient,
    mesh,
    dimension=2,
    scale = 2e6,
    show_mesh=False,
    ax=None,
    figsize=(10, 10),
    ):
    """
    Plot a gradient field on a mesh.

    Parameters
    ----------
    spatial_gradient : np.ndarray
        The gradient field to plot.
    mesh : pygimli.core._pygimli_.Mesh
        The mesh on which the gradient field is defined.
    dimension : int, optional
        The dimension of the mesh. Can be 2 or 3. The default is 2.
    scale : float, optional
        The scale of the arrows in the plot. The default is 2e6.
    show_mesh : bool, optional
        Whether to show the mesh. The default is False.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the gradient field. If not provided, a new
        figure and axis are created. The default is None.
    figsize : tuple, optional
        The size of the figure. The default is (10, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the gradient field is plotted.
    ax : matplotlib.axes.Axes
        The axis on which the gradient field is plotted.

    """
    # Calculate x,y,z vectors
    cell_centers = [cell.center() for cell in mesh.cells()]

    x = np.array([cell_center[0] for cell_center in cell_centers])
    y = np.array([cell_center[1] for cell_center in cell_centers])
    if dimension == 3:
        z = np.array([cell_center[2] for cell_center in cell_centers])

    # Calculate u,v,w vectors
    u = spatial_gradient[:,0]
    v = spatial_gradient[:,1]
    if dimension == 3:
        w = spatial_gradient[:,2]

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    else:
        fig = ax.get_figure()

    # Plot the gradient field
    if dimension == 2:
        ax.quiver(x, y, u, v, angles="xy", scale=scale)
    elif dimension == 3:
        ax.quiver(x, y, z, u, v, w, scale=scale)

    # Show mesh if requested
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
    return fig, ax

def plot_absolute_value_of_gradient_field(
    spatial_gradient_absolute,
    mesh,
    cmin=None,
    cmax=None,
    label="Norm of structure gradient",
    cmap=None,
    show_mesh=False,
    log=False,
    sensor_positions=None,
    ax=None,
    figsize=(10, 10),
    ):
    """
    Plot the absolute value of a gradient field on a mesh.

    Parameters
    ----------
    spatial_gradient_absolute : np.ndarray
        The norm of the gradient field to plot.
    mesh : pygimli.core._pygimli_.Mesh
        The mesh on which the gradient field is defined.
    cmin : float, optional
        The minimum value of the color scale. If not provided, the minimum
        value of the gradient field is used. The default is None.
    cmax : float, optional
        The maximum value of the color scale. If not provided, the maximum
        value of the gradient field is used. The default is None.
    label : str, optional
        The label of the color scale. The default is "Norm of structure gradient".
    cmap : str, optional
        The colormap to use. The default is None.
    show_mesh : bool, optional
        Whether to show the mesh. The default is False.
    log : bool, optional
        Whether to use a logarithmic scale. The default is False.
    sensor_positions : np.ndarray, optional
        The positions of the sensors. The default is None.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the gradient field. If not provided, a new
        figure and axis are created. The default is None.
    figsize : tuple, optional
        The size of the figure. The default is (10, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the gradient field is plotted.
    ax : matplotlib.axes.Axes
        The axis on which the gradient field is plotted.
            
    """

    # Define default colormap options
    if cmin is None:
        cmin = np.min(spatial_gradient_absolute)
    if cmax is None:
        cmax = np.max(spatial_gradient_absolute)
    if cmap is None:
        cmap = 'copper'

    # Create figure and axis if not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    else:
        fig = ax.get_figure()


    # Plot the gradient field
    pg.show(
        mesh,
        spatial_gradient_absolute,
        label=label,
        logScale=log,
        cMin=cmin,
        cMax=cmax,
        cMap=cmap,
        ax=ax
    )
    # ticks = (np.array([-.2, -.1, 0, .1, .2]) * 1).tolist()
    # ax.xaxis.set_ticks(ticks)
    # ax.yaxis.set_ticks(ticks)

    if not sensor_positions is None:
        pg.viewer.mpl.drawSensors(ax, sensor_positions, diam=0.005)

    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
    return fig, ax

def plot_absolute_value_of_gradient_field_from_vectors(
    spatial_gradient,
    mesh,
    cmin=None,
    cmax=None,
    label="Norm of structure gradient",
    cmap=None,
    show_mesh=False,
    log=False,
    sensor_positions=None,
    ax=None,
    figsize=(10, 10),
    ):
    """
    Plot the absolute value of a gradient field on a mesh.

    Parameters
    ----------
    spatial_gradient : np.ndarray
        The gradient field to plot.
    mesh : pygimli.core._pygimli_.Mesh
        The mesh on which the gradient field is defined.
    cmin : float, optional
        The minimum value of the color scale. If not provided, the minimum
        value of the gradient field is used. The default is None.
    cmax : float, optional
        The maximum value of the color scale. If not provided, the maximum
        value of the gradient field is used. The default is None.
    label : str, optional
        The label of the color scale. The default is "Norm of structure gradient".
    cmap : str, optional
        The colormap to use. The default is None.
    show_mesh : bool, optional
        Whether to show the mesh. The default is False.
    log : bool, optional
        Whether to use a logarithmic scale. The default is False.
    sensor_positions : np.ndarray, optional
        The positions of the sensors. The default is None.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the gradient field. If not provided, a new
        figure and axis are created. The default is None.
    figsize : tuple, optional
        The size of the figure. The default is (10, 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure on which the gradient field is plotted.
    ax : matplotlib.axes.Axes
        The axis on which the gradient field is plotted.
            
    """

    # Calculate the absolute value of the gradient field
    grad_abs = np.linalg.norm(spatial_gradient, axis=1)

    # Plot the absolute value of the gradient field
    fig, ax = plot_absolute_value_of_gradient_field(
        spatial_gradient_absolute=grad_abs,
        mesh=mesh,
        cmin=cmin,
        cmax=cmax,
        label=label,
        cmap=cmap,
        show_mesh=show_mesh,
        log=log,
        sensor_positions=sensor_positions,
        ax=ax,
        figsize=figsize,
    )
    return fig, ax

def plot_hessian_matrix_overview(
    hessian_matrix_list,
    mesh,
    cmap=None,
    show_mesh=False,
    log=False,
    sensor_positions=None,
    figsize=(15, 10),
    ):
    """
    Function to plot an overview of the hessian matrix. The overview consists of
    the norm of the hessian matrix and the diagonal of the hessian matrix (laplacian).
    As well as the Eigenvectors and Eigenvalues of the hessian matrix.

    Parameters:
    hessian_matrix_list (list): The list of hessian matrices.

    """
    # Calculate the norm of the hessian matrix
    hessian_matrix_norm_list = np.array([np.linalg.norm(hessian_matrix) for hessian_matrix in hessian_matrix_list])

    # Calculate the diagonal of the hessian matrix
    laplacian_list = np.array([np.sum(np.diag(hessian_matrix)) for hessian_matrix in hessian_matrix_list])

    # Calculate the eigenvectors and eigenvalues of the hessian matrix
    evtuple = [np.linalg.eig(hessian_matrix) for hessian_matrix in hessian_matrix_list]
    reorder_indices = np.array([np.argsort(ev[0])[::-1] for ev in evtuple])

    eigenvalues_ordered = np.array([ev[0][reorder_indices[num]] for num, ev in enumerate(evtuple)])
    eigenvectors_ordered = np.array([ev[1][:, reorder_indices[num]] for num, ev in enumerate(evtuple)])

    eigenvalues_max = np.array([np.max(ev) for ev in eigenvalues_ordered])
    eigenvalues_min = np.array([np.min(ev) for ev in eigenvalues_ordered])

    eigenvectors_max = np.array([ev[:,0] for ev in eigenvectors_ordered])
    eigenvectors_min = np.array([ev[:,-1] for ev in eigenvectors_ordered])

    # Plot the norm of the hessian matrix
    fig, axs = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)

    ax = axs[0, 0]
    cmin=0
    cmax=np.max(hessian_matrix_norm_list)
    pg.show(mesh, hessian_matrix_norm_list,
            label="Norm of hessian matrix",
            logScale=log,
            ax=ax,
            cMin=cmin,
            cMax=cmax,
            cMap=cmap)
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    # Plot the Laplacian
    ax = axs[1, 0]
    cmax = np.max(np.abs(laplacian_list))
    cmin = -cmax
    pg.show(mesh, laplacian_list,
            label="Laplacian",
            logScale=log,
            ax=ax,
            cMin=cmin,
            cMax=cmax,
            cMap="seismic")
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    # Plot the eigenvectors of the hessian matrix
    ax = axs[0, 1]
    pg.show(mesh, eigenvectors_max,
            label="Eigenvector for biggest eigenvalue",
            ax=ax)
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    # Plot the eigenvectors of the hessian matrix
    ax = axs[1, 1]
    pg.show(mesh, eigenvectors_min,
            label="Eigenvector for smallest eigenvalue",
            ax=ax)
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    # Plot the eigenvalues of the hessian matrix
    ax = axs[0, 2]
    cmin = np.min(eigenvectors_max)
    cmax = np.max(eigenvectors_max)
    pg.show(mesh, eigenvalues_max,
            label="Max Eigenvalue",
            ax=ax,
            cMin=cmin,
            cMax=cmax,
            cMap=cmap)
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    # Plot the eigenvalues of the hessian matrix
    ax = axs[1, 2]
    cmin = np.min(eigenvectors_min)
    cmax = np.max(eigenvectors_min)
    pg.show(mesh, eigenvalues_min,
            label="Min Eigenvalue",
            ax=ax,
            cMin=cmin,
            cMax=cmax,
            cMap=cmap)
    if show_mesh:
        pg.viewer.mpl.drawSelectedMeshBoundaries(ax, mesh.boundaries(),
                                                 linewidth=0.3, color="0.2")
        
    if not sensor_positions is None:
        for ax in axs.flatten():
            pg.viewer.mpl.drawSensors(ax, sensor_positions, diam=0.005)

    return fig, axs
