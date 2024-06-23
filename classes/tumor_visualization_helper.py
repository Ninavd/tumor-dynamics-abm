import numpy as np
import matplotlib.pyplot as plt
import copy

class TumorVisualizationHelper():
    def __init__(self, model):
        self.model = model

    def calculate_average_distance(self, mask, center):
        """
        Calculate the average distance from the center of the mask to the edge.

        Args:
            mask (np.ndarray): Binary mask matrix.

        Returns:
            float: Average distance to the edge.
        """
        distances = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    cell = np.array([i, j])
                    distance = np.linalg.norm(cell - center)
                    distances.append(distance)
        if distances == []:
            return 0
        return np.mean(distances)
    
    def find_geographical_center(self, mask):
        """
        Find the geographical center of the mask.

        Args:
            mask (np.ndarray): Binary mask matrix.

        Returns:
            tuple: Coordinates of the geographical center.
        """
        total = np.sum(mask)
        weighted_sum = np.array([0, 0])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    cell = np.array([i, j])
                    weighted_sum += cell
        if total == 0: # if total is 0, return the center of the mask, prevent division by 0
            return (mask.shape[0] // 2, mask.shape[1] // 2)
        return tuple(weighted_sum // total)

    def get_edges_of_a_mask(self, mask):
        """
        Find the edges of a binary mask.
        
        Args: 
            mask (np.ndarray): Binary mask matrix.
        
        Returns:
            np.ndarray: Binary matrix with only full cells that border an empty cell."""
        edges_matrix = np.zeros(mask.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    if mask[i-1, j] == 0 or mask[i+1, j] == 0 or mask[i, j-1] == 0 or mask[i, j+1] == 0:
                        edges_matrix[i, j] = 1
        # TODO: check to see if it makes more sense to return this as a sparse matrix, as only the edges are highlighted, so it might be sparse enough for large grids? https://stackoverflow.com/a/36971131
        
        return edges_matrix
    
    def compute_variance_of_roughness(self, edges_matrix, center):
        """
        Computes the roughness of an imperfectly drawn circle.
        
        Parameters:
        r_theta (function): A function representing the radius r(θ).
        r0 (float): The average radius of the imperfect circle.
        num_points (int): Number of points to sample along the circle.
        
        Returns:
        float: The roughness R of the imperfect circle.
        """
        edge_points = np.argwhere(edges_matrix == 1)
        radii = edge_points - center
        if (radii.shape[0] == 0):
            return 0
        r0 = np.mean(radii)
        variance_r = np.sum((radii - r0) ** 2)
        return variance_r
    
    def cells_at_tumor_surface(self, mask):
        """
        Compute the number of cells at the tumor surface.

        Args:
            mask (np.ndarray): Binary mask matrix.
            iteration (int): Iteration number.

        Returns:
            int: Number of grid cells at the tumor surface.
        """
        edges_matrix = self.get_edges_of_a_mask(mask)
        return np.sum(edges_matrix), edges_matrix

    def calculate_roughness(self):
        roughness_values = []

        for i in range(len(self.model.N_Ts)):
            mask = (self.model.N_Ts[i] + self.model.Necs[i]) > 0
            N_r, edges_matrix = self.cells_at_tumor_surface(mask)
            center = self.find_geographical_center(mask)
            variance = self.compute_variance_of_roughness(edges_matrix, center)
            if N_r == 0:
                roughness_values.append(0)
            else: 
                roughness = np.sqrt(variance / N_r)
                roughness_values.append(roughness)
        
        return roughness_values
    
    def calculate_radial_distance(self):
        radial_distance = []
        for i in range(len(self.model.N_Ts)):
            mask = self.model.N_Ts[i] > 0
            geographical_center = self.find_geographical_center(mask)
            edges_of_mask = self.get_edges_of_a_mask(mask)
            radial_distance.append(self.calculate_average_distance(edges_of_mask, geographical_center))

        return radial_distance