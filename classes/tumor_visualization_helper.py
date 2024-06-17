import numpy as np

class TumorVisualizationHelper():
    def __init__(self):
        pass

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
    
    def compute_variance_of_roughness(self, mask, center):
        """
        Computes the roughness of an imperfectly drawn circle.
        
        Parameters:
        r_theta (function): A function representing the radius r(θ).
        r0 (float): The average radius of the imperfect circle.
        num_points (int): Number of points to sample along the circle.
        
        Returns:
        float: The roughness R of the imperfect circle.
        """
        edge_points = np.argwhere(mask == 1)
        radii = np.linalg.norm(edge_points - center, axis=1)
        r0 = np.mean(radii)
        variance_r = np.mean((radii - r0) ** 2)
        return variance_r
    
    def cells_at_tumor_surface(self, N_t, mask):
        """
        Compute the number of cells at the tumor surface.

        Args:
            mask (np.ndarray): Binary mask matrix.
            iteration (int): Iteration number.

        Returns:
            int: Number of cells at the tumor surface.
        """
        edges_matrix = self.get_edges_of_a_mask(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if edges_matrix[i, j]:
                    edges_matrix[i, j] = N_t[i, j]
        return np.sum(edges_matrix)