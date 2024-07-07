import numpy as np
from scipy.optimize import curve_fit

class TumorVisualizationHelper():


    def __init__(self, model):
        self.model = model

    def calculate_average_distance(self, edge_mask, center):
        """
        Calculate the average distance from the center of the mask to the edge.

        Args:
            mask (np.ndarray): Binary mask matrix.
            center (tuple): Coordinates of the center

        Returns:
            float: Average distance to the edge.
        """
        border_indeces = np.argwhere(edge_mask)
        if len(border_indeces) == 0:
            return 0
        
        distances = np.linalg.norm(border_indeces - center, axis=1)
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
        
        # if total is 0, return center of mask to prevent division by 0
        if total == 0: 
            return (mask.shape[0] // 2, mask.shape[1] // 2)
        
        filled_indeces = np.argwhere(mask != 0)
        weighted_sum = np.sum(filled_indeces, axis=0)
        return np.round(tuple(weighted_sum / total))

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
                    # check if on the edge to prevent indexing error
                    if i - 1 == -1 or i + 1 == mask.shape[0] or j + 1 == mask.shape[0] or j - 1 == -1:
                        edges_matrix[i, j] = 1 
                    elif mask[i-1, j] == 0 or mask[i+1, j] == 0 or mask[i, j-1] == 0 or mask[i, j+1] == 0:
                        edges_matrix[i, j] = 1
        # TODO: check to see if it makes more sense to return this as a sparse matrix, as only the edges are highlighted, so it might be sparse enough for large grids? https://stackoverflow.com/a/36971131
        
        return edges_matrix
    
    def compute_variance_of_radius(self, edges_matrix, center):
        """
        Computes the radius of an imperfectly drawn circle.
        
        Parameters:
        edges_matrix (ndarray): represents border of the shape.
        center (tuple): approximate center of the shape
        
        Returns:
        float: The variance of radius of the imperfect circle.
        """
        edge_points = np.argwhere(edges_matrix == 1)
        radii = np.linalg.norm(edge_points - center, axis=1)
        if (len(radii) <= 1):
            return 0
        return np.var(radii)

    def calculate_roughness_progression(self):
        """
        Find roughness over time.
        """
        steps = len(self.model.N_Ts)
        roughness_values = np.zeros(steps)

        for i in range(steps):
            N_T, Nec = self.model.N_Ts[i], self.model.Necs[i]
            roughness = self.calculate_roughness(N_T, Nec)
            roughness_values[i] = roughness
        
        return roughness_values
    
    def calculate_roughness(self, N_T, Nec):
        """
        Find roughness of a single snapshot.
        """
        mask = (N_T + Nec) > 0
        edges_matrix = self.get_edges_of_a_mask(mask)

        center   = self.find_geographical_center(edges_matrix)
        variance = self.compute_variance_of_radius(edges_matrix, center)

        roughness = np.sqrt(variance)
        return roughness

    def radius_progression(self):
        """
        Calculates the radial distance of the tumor at each time step.

        Returns:
            list: a list of values representing the radial distance of the tumor at each time step.
        """
        radial_distance = np.zeros(len(self.model.N_Ts))

        for i, N_T in enumerate(self.model.N_Ts):
            radial_distance[i] = self.radius(N_T)

        return radial_distance

    def radius(self, N_T):
        """
        Find radius of a single snapshot.
        """
        mask = N_T > 0
        geographical_center = self.find_geographical_center(mask)
        edges_of_mask = self.get_edges_of_a_mask(mask)

        return self.calculate_average_distance(edges_of_mask, geographical_center)

    def calculate_velocities(self):
        """
        Calculate velocities for every delta_d timesteps
        
        Returns:
            list: a list of velocities for every delta_d timesteps
        """
        velocities = []
        intervals = self.model.radii[::self.model.delta_d]
        for i in range(1, len(intervals)):
            velocities.append((intervals[i] 
                               - intervals[i-1]) / self.model.delta_d)
        return velocities


    def velocity_linear_fit(self):
        """
        Perform linear fit on radius progression.

        Returns: 
            float: velocity of the tumor growth
        """
        fit_func = lambda x, a, b: a * x + b
        skip = 100
        popt, pcov = curve_fit(fit_func, xdata = range(skip, len(self.model.N_Ts)), ydata=self.model.radii[skip:])
        velocity, offset = popt
        return velocity, offset