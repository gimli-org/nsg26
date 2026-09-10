"""
This module contains the classes to store the information about the mesh.

The module contains the following classes:

    * CellNeighbourInfo: Class to store the information about the neighborhood of a node.
    * MeshInfo: Class to store the information about the mesh.

The module contains the following functions:
    
    * cell_area_triangle: Function to calculate the area from pyGimli cell object.
    * distance_to_neighbour_list: Function to calculate the neighbours of a cell.

Author: Hagen Söding
Affiliation: ETH Zürich
Email: hagen.soeding@eaps.ethz.ch

"""


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Function to calculate area from pyGimli cell object
def cell_area_triangle(cell):
    """
    Calculate the area of a triangle cell.

    Parameters:
    cell (object): The cell object representing a triangle.

    Returns:
    float: The area of the triangle.

    Raises:
    ValueError: If the cell does not have exactly 3 nodes.

    """
    # Get the nodes of the cell
    nodes = cell.nodes()
    # Get the number of nodes
    n_nodes = len(nodes)
    # Initialize the area
    area = 0
    # Abort if the cell has less/more than 3 nodes
    if n_nodes < 3 or n_nodes > 3:
        raise ValueError("Only triangles are supported")
    # Get the coordinates of the nodes
    x1, y1 = nodes[0].x(), nodes[0].y()
    x2, y2 = nodes[1].x(), nodes[1].y()
    x3, y3 = nodes[2].x(), nodes[2].y()
    # Add the area of the triangle formed by the origin and the two nodes
    # This is Heron's formula bzw. crossproduct
    area = 0.5 * (x1 * y2 - x2 * y1 + x2 * y3 - x3 * y2 + x3 * y1 - x1 * y3)
    return area

# Functions to calculate the neighbours of a cell
def distance_to_neighbour_list_for_cell(cell, mesh, dist = 1.0, dimension=2):
    """
    Function to calculate the neighbours of a cell.

    Parameters:
    cell (object): The cell object.
    mesh (object): The mesh object.
    dist (float): The distance to consider as a neighbour.
    dimension (int): The dimension of the mesh.

    Returns:
    
    function: The function to calculate the neighbours of a cell.

    """
    # Get the centers of the cells
    centers = np.zeros((len(mesh.cells()),dimension))
    for num, cell_temp in enumerate(mesh.cells()):
        centers[num] = np.array(cell_temp.center())[0:dimension]

    # Calculate the distances
    centers = centers - np.array(cell.center())[0:dimension]

    # Get the neighbours of the cells
    neigh_list = list(np.where(np.linalg.norm(centers, axis=1)<=dist)[0])
    return neigh_list

def distance_to_neighbour_list_for_mesh(mesh, dist = 1.0, dimension=2):
    """
    Function to calculate the neighbours of a cell for the whole mesh.

    Parameters:
    mesh (object): The mesh object.
    dist (float): The distance to consider as a neighbour.
    dimension (int): The dimension of the mesh.

    Returns:
    
    function: The function to calculate the neighbours of a cell.

    """
    # Get the centers of the cells
    centers = np.zeros((len(mesh.cells()),dimension))
    for num, cell in enumerate(mesh.cells()):
        centers[num] = np.array(cell.center())[0:dimension]

    # Calculate the distance matrix
    dist_mat = sp.spatial.distance_matrix(centers, centers)
    dist_mat_bool = dist_mat<=dist

    # Get the neighbours of the cells
    neigh_list_list = []
    for j in range(dist_mat_bool.shape[0]):
        temp_list = list(np.where(dist_mat_bool[j] is True)[0])
        temp_list.remove(j)
        neigh_list_list.append(temp_list)
    return neigh_list_list

def get_n_closest_neighbours(cell, mesh, n=3):
    """
    Function to get the n closest neighbours of a cell.

    Parameters:
    cell (object): The cell object.
    mesh (object): The mesh object.
    n (int): The number of neighbours to get.

    Returns:

    np.array: The list of the cell numbers of the n closest neighbours.
    """
    assert n > 0, "n must be greater than 0"
    cell_centers = np.array([np.array(ce.center()) for ce in mesh.cells()])
    distances = sp.spatial.distance_matrix(cell_centers, [cell_centers[cell.id()]])
    closest_neighbours = np.argsort(distances, axis=0)[1:n+1]
    return closest_neighbours[:,0]

def get_n_closest_neighbours_function_for_mesh(mesh, n=3):
    """
    Function to get the n closest neighbours of a cell for the whole mesh.

    Parameters:
    mesh (object): The mesh object.
    n (int): The number of neighbours to get.

    Returns:

    function: The function to get the n closest neighbours of a cell.
    """
    cell_centers = np.array([np.array(ce.center()) for ce in mesh.cells()])
    distance_matrix = sp.spatial.distance_matrix(cell_centers, cell_centers)
    def get_closest_neighbour_preset_function(cell):
        distances = distance_matrix[:, cell.id()]
        closest_neighbours = np.argsort(distances)[1:n+1]
        return closest_neighbours
    return get_closest_neighbour_preset_function

# Class for storing the information about the neighborhood of a node
class CellNeighbourInfo:
    """
    Class to store the information about the neighborhood of a cell.

    Attributes:
    cell_number (int): The number of the cell.
    cell_area (float): The area of the cell.
    cell_center (np.array): The center of the cell.
    dimension (int): The dimension of the mesh.
    neighbour_cells (list): The list of the cell numbers of the neighbours.
    distance_matrix (np.array): The distance matrix of the cell.
    distance_matrix_gn_taylor1 (np.array): The distance matrix of the cell used for the 
                                           Gauss-Newton algorithm.
    distance_matrix_gn_taylor2 (np.array): The distance matrix of the cell used for the 
                                           Gauss-Newton algorithm from a second degree 
                                           taylor polynomial approximation.

    """

    def __init__(
            self,
            cell,
            dimension=2,
            cell_area_function=cell_area_triangle,
            neighbour_function=None,
            verbose=False):
        """
        Initialize the NeighbourAttribute object.

        Parameters:
        cell (object): The cell object.
        dimension (int): The dimension of the mesh.
        cell_area_function (function): The function to calculate the area of the cell.
        neighbour_function (function): The function to calculate the neighbours of the cell.
        verbose (bool): If True, print the progress.
        """
        # Get the cell number
        self._cell_number = cell.id()

        # Get the area of the cell
        self._cell_area = cell_area_function(cell)

        # Get the dimension of the cell
        assert isinstance(dimension, int) and dimension > 0, "Dimension must be a positive integer"
        self._dimension = dimension

        # Get the center of the cell
        self._cell_center = np.array(cell.center())[0:self.dimension]

        # Get the neighbours of the cell
        if neighbour_function is not None:
            self._neighbour_cells = neighbour_function(cell)
        else:
            self._neighbour_cells = []
            for j in range(cell.neighborCellCount()):
                try:
                    self._neighbour_cells.append(cell.neighborCell(j).id())
                except AttributeError:
                    if verbose:
                        print(f"Cell {cell.id()} has no neighbour at position {j}")
            # self.neighbour_cells = [cell.neighborCell(j).id() for
            #                         j in range(cell.neighborCellCount())]

        # Get the distance matrix of the cell
        self._distance_matrix = None

        # Get the inverse of the distance matrix of the cell
        self._distance_matrix_gn_taylor1 = None

        # Get the inverse of the distance matrix for a 2nd order taylor approximation
        self._distance_matrix_gn_taylor2 = None

    @property
    def cell_number(self):
        """ The number of the cell. """
        return self._cell_number

    @property
    def cell_area(self):
        """ The area of the cell. """
        return self._cell_area

    @property
    def cell_center(self):
        """ The center of the cell. """
        return self._cell_center

    @property
    def dimension(self):
        """ The dimension of the mesh. """
        return self._dimension

    @property
    def neighbour_cells(self):
        """ The list of the cell numbers of the neighbours. """
        return self._neighbour_cells

    @property
    def distance_matrix(self):
        """ The distance matrix of the cell. """
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, mesh):
        distance_matrix = np.zeros((len(self.neighbour_cells), self.dimension))

        for i, cell_id in enumerate(self.neighbour_cells):
            cell = mesh.cell(cell_id)
            distance_matrix[i] = np.array(cell.center())[0:self.dimension] - self.cell_center
        self._distance_matrix = distance_matrix.copy()

    @property
    def distance_matrix_gn_taylor1(self):
        """ The distance matrix of the cell used for the Gauss-Newton algorithm. 
        With the distance matrix as D, this is given as (D.T@D)^-1@D. """
        return self._distance_matrix_gn_taylor1

    @distance_matrix_gn_taylor1.setter
    def distance_matrix_gn_taylor1(self, mesh):
        if not isinstance(self._distance_matrix, np.ndarray):
            self.distance_matrix = mesh
        else:
            try:
                self._distance_matrix_gn_taylor1 = np.linalg.inv(\
                    self.distance_matrix.T @ self.distance_matrix)\
                    @ self.distance_matrix.T
            except np.linalg.LinAlgError:
                self._distance_matrix_gn_taylor1 = None
                print("Singular matrix for cell", self.cell_number)
                print("GN Taylor 1 matrix not calculated - try increasing the number of neighbours")

    @property
    def distance_matrix_gn_taylor2(self):
        """ The distance matrix of the cell used for the Gauss-Newton algorithm from a
        second degree taylor polynomial approximation. The second degree taylor polynomial
        approximation is given as f(x) - f(x0) = (x-x0).T nabla{f(x0)} + 0.5*(x-x0).T H(x0)(x-x0).
        Writing the gradient as [a b] and the Hessian as [[c d] [e f]], the equation reads
        f(x) - f(x0) = (x-x0).T [a b] + 0.5*(x-x0).T [[c d] [e f]] (x-x0). Combining the vector of unknowns
        as [a b c d e f] and the vector of model differences as [f(x) - f(x0)] and the difference matrix as D,
        the equation reads [f(x) - f(x0)] = [D (x-x0)_0 * D (x-x0)_1 * D ...] [a b c d e f ...], with the 
        Hessian matrix "stacked" columnwise. The components of the gradient and Hessian are then given as
        the solutiong of the linear system.
        """
        return self._distance_matrix_gn_taylor2

    @distance_matrix_gn_taylor2.setter
    def distance_matrix_gn_taylor2(self, mesh):
        if not isinstance(self._distance_matrix, np.ndarray):
            self.distance_matrix = mesh
        else:
            try:
                """ For theory see property description. """
                # first matrix is with respect to the gradient
                # follow up matrices is with respect to the rows/cols of the Hessian
                dist_mat_sq = np.tile(self._distance_matrix, reps=(1,1+self.dimension)).copy()
                # multiply the Hessian distance matrices with the corrseponding distances
                for dim in range(self.dimension):
                    dist_mat_sq[:,(dim+1)*self.dimension:(dim+2)*self.dimension] *= np.tile(dist_mat_sq[:,dim], reps=(self.dimension,1)).T
                # Add symmetric rows - calculate running indices
                triu_indices = []
                tril_indices = []
                for i in range(self.dimension):
                    for j in range(self.dimension):
                        running_index = self.dimension + i * self.dimension + j
                        if i<j:
                            triu_indices.append(running_index)
                        if j<i:
                            tril_indices.append(running_index)
                dist_mat_sq[:, triu_indices] = dist_mat_sq[:, tril_indices]
                dist_mat_sq = np.delete(dist_mat_sq, tril_indices, axis=1)
                self._distance_matrix_gn_taylor2 = np.dot(
                    np.linalg.inv(np.dot(dist_mat_sq.T, dist_mat_sq)), dist_mat_sq.T
                    )
            except np.linalg.LinAlgError:
                self._distance_matrix_gn_taylor2 = None
                print("Singular matrix for cell", self.cell_number)
                print("GN Taylor 2 matrix not calculated - try increasing the number of neighbours")

    def get_gradient_mesh_sensitivities(self, order=1):
        """
        Function to return the sensitivities of the gradient with respect to the mesh. That means,
        that the gradient is given as gradient = output @ model. This functions returns an column index vector
        as well as the entries of the columns.
        
        Parameters:
        order (int): The order of the taylor polynomial approximation.

        Returns:
        np.array: The column index vector.
        np.array: The entries of the columns.
        """
        if order == 1:
            assert self._distance_matrix_gn_taylor1 is not None, "GN Taylor 1 matrix not calculated"
            original_matrix = self._distance_matrix_gn_taylor1
        elif order == 2:
            assert self._distance_matrix_gn_taylor2 is not None, "GN Taylor 2 matrix not calculated"
            original_matrix = self._distance_matrix_gn_taylor2[:self.dimension, :]
        else:
            raise ValueError("Only orders 1 and 2 are supported")
        # Get the column index vector and append the cell id
        # column_index_vector = self.neighbour_cells
        # column_index_vector.append(self.cell_number)
        column_index_vector = np.array([*self.neighbour_cells, self.cell_number])

        # Vector concerning the cell itself is given as negative sum of all columns
        center_cell_it_sensitivity = -np.sum(original_matrix, axis=1)
        # Append the sensitivity of the cell itself
        original_matrix = np.hstack((original_matrix, center_cell_it_sensitivity[:,np.newaxis]))
        return column_index_vector, original_matrix
    
    def get_hessian_mesh_sensitivities(self):
        """
        Function to return the sensitivities of the Hessian with respect to the mesh. That means,
        that the Hessian is given as Hessian = output @ model. This functions returns an column index vector.

        Parameters:
            None

        Returns:
            np.array: The column index vector.
            np.array: The entries of the columns.
        """
        assert self._distance_matrix_gn_taylor2 is not None, "GN Taylor 2 matrix not calculated"
        original_matrix = self._distance_matrix_gn_taylor2[self.dimension:, :]
        # Get the column index vector and append the cell id
        column_index_vector = np.array([*self.neighbour_cells, self.cell_number])



        # Vector concerning the cell itself is given as negative sum of all columns
        center_cell_it_sensitivity = -np.sum(original_matrix, axis=1)
        # Append the sensitivity of the cell itself
        original_matrix = np.hstack((original_matrix, center_cell_it_sensitivity[:,np.newaxis]))
        return column_index_vector, original_matrix

class MeshInfo:
    """
    Class to store the information about the mesh.

    Attributes:
    mesh (object): The mesh object.
    dimension (int): The dimension of the mesh.
    cell_area_function (function): The function to calculate the area of the cell.
    neighbour_function (function): The function to calculate the neighbours of the cell.
    cell_neighbour_info (list): The list of the CellNeighbourInfo objects.

    """

    def __init__(
            self,
            mesh,
            dimension=2,
            cell_area_function=cell_area_triangle,
            neighbour_function=None,
            initialise_gn1=True,
            initialise_gn2=False):
        """
        Initialize the MeshInfo object.

        Parameters:
        mesh (object): The mesh object.
        dimension (int): The dimension of the mesh.
        cell_area_function (function): The function to calculate the area of the cell.
        neighbour_function (function): The function to calculate the neighbours of the cell.
        initialise_matrices (bool): If True, initialise the matrices for gradient calculation.

        """
        # Set the mesh
        self._mesh = mesh

        # Set the dimension of the mesh
        self._dimension = dimension

        # Set the function to calculate the area of the cell
        self._cell_area_function = cell_area_function

        # Set the function to calculate the neighbours of the cell
        self._neighbour_function = neighbour_function

        # Set the region of interest
        cell_markers = np.array(mesh.cellMarkers())
        smallest_cell_marker= np.min(cell_markers)
        if np.all(cell_markers == smallest_cell_marker):
            print("All cells have the same marker - region of interssssst is the whole mesh")
            self._region_of_interest = np.array([True]*len(mesh.cells()))
        else:
            print("Cells have different markers - region of interest are cells with non minimum marker")
            self._region_of_interest = np.array(cell_markers != smallest_cell_marker)

        # Set the cell neighbour info
        cell_neighbour_info_list = []
        no_of_cells = len(self._mesh.cells())
        five_percent_cells = no_of_cells // 20
        counter = 0

        for num, cell in enumerate(self._mesh.cells()):
            if num % five_percent_cells == 0:
                #print(f"Progress at {5*counter}% - Calculating cell {num}/{no_of_cells}")
                counter += 1
            cell_neighbour_info = CellNeighbourInfo(
                cell,
                self._dimension,
                self._cell_area_function,
                self._neighbour_function)

            # Initialise the matrices for the Gauss-Newton algorithm
            cell_neighbour_info.distance_matrix = self._mesh

            if initialise_gn1:
                cell_neighbour_info.distance_matrix_gn_taylor1 = self._mesh
            if initialise_gn2:
                cell_neighbour_info.distance_matrix_gn_taylor2 = self._mesh

            # Append the cell neighbour info to the list
            cell_neighbour_info_list.append(cell_neighbour_info)
        self._cell_neighbour_info = cell_neighbour_info_list
        self._gn_taylor_1_set_successfully = np.all([cni.distance_matrix_gn_taylor1 is not None for cni in self._cell_neighbour_info])
        self._gn_taylor_2_set_successfully = np.all([cni.distance_matrix_gn_taylor2 is not None for cni in self._cell_neighbour_info])
        
    @property
    def mesh(self):
        """ The mesh object. """
        return self._mesh

    @property
    def dimension(self):
        """ The dimension of the mesh. """
        return self._dimension

    @property
    def cell_area_function(self):
        """ The function to calculate the area of the cell. """
        return self._cell_area_function
    
    @property
    def region_of_interest(self):
        """ The region of interest. """
        return self._region_of_interest

    @property
    def neighbour_function(self):
        """ The function to calculate the neighbours of the cell. """
        return self._neighbour_function

    @property
    def cell_neighbour_info(self):
        """ The list of the CellNeighbourInfo objects. """
        return self._cell_neighbour_info
    
    def show_region_of_interest(self, markersize=1, marker="o", mode="triang", ax=None):
        """ Shows the region of interest."""
        mesh = self.mesh
        cell_centers = mesh.cellCenters()
        region_of_interest = self.region_of_interest

        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        else:
            fig = ax.get_figure()
        ax.set_title("Region of Interest")

        if mode == "scatter":
            x, y = cell_centers[:, 0], cell_centers[:, 1]
            ax.scatter(x[region_of_interest], y[region_of_interest], marker=marker, color="red", s=markersize)
            ax.scatter(x[~region_of_interest], y[~region_of_interest], marker=marker, color="blue", s=markersize)
            # Create a legend
            ax.plot(x[0], y[0], marker=marker, markersize=markersize, label="Region of Interest")
            ax.plot(x[0], y[0], marker=marker, markersize=markersize, label="Background")
            ax.legend(loc="upper right")
        elif mode == "triang":
            # Get the coordinates and connectivity of the mesh
            x = mesh.positions()[:, 0]
            y = mesh.positions()[:, 1]
            nodes = [
                np.array([node.id() for node in cell.nodes()]) for cell in mesh.cells()
            ]

            # Create the matplotlib triangulation object
            triang = tri.Triangulation(x, y, nodes)

            # Plot the mesh
            im = ax.tripcolor(triang, facecolors=self.region_of_interest, shading="flat", cmap="coolwarm", vmin=0, vmax=1)
            fig.colorbar(im, ax=ax, label="Region of Interest", ticks=[0, 1], location="bottom")
            ax.triplot(triang, color='black', linewidth=0.2)
        return fig, ax
