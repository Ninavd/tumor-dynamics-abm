import numpy as np
from scipy.optimize import curve_fit

class TumorVisualizationHelper:
    """
    Visualization helper class.

    Helper TumorVisualization calculate statistics used in plots, such as
    average radius of the tumor or roughness over time.

    Example usage:
    ```
        TVH = TumorVisualizationHelper(model)
        roughness_over_time = TVH.calculate_roughness_progression()
    ```
    
    Attributes:
        model: Model to analyze (after completed simulation).
    """

    def __init__(self, model):
        """
        Initializes helper class.

        Args:
            model (TumorGrowth): Model to analyze (after completed simulation).
        """
        self.model: object = model

    def calculate_average_distance(self, edge_mask: np.ndarray, center: tuple[int]) -> float:
        """
        Calculate the average distance from the center of the mask to the edge.

        Args:
            mask (np.ndarray): Binary mask matrix of border.
            center (tuple): Coordinates of the center

        Returns:
            float: Average distance to the edge.
        """
        border_indeces = np.argwhere(edge_mask)
        if len(border_indeces) == 0:
            return 0
        
        distances = np.linalg.norm(border_indeces - center, axis=1)
        return np.mean(distances)
    
    def find_geographical_center(self, mask: np.ndarray) -> tuple[int]:
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

    def get_edges_of_a_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Find the edges of a binary mask.
        
        Args: 
            mask (np.ndarray): Binary mask matrix of filled object.
        
        Returns:
            np.ndarray: Binary matrix with only edge cells filled in.
        """
        edges_matrix = np.zeros(mask.shape)
        is_on_edge = lambda i,j : i - 1 == -1 or i + 1 == mask.shape[0] or j + 1 == mask.shape[0] or j - 1 == -1
        
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):

                # check if on the edge to prevent indexing error
                if mask[i,j] and is_on_edge(i,j):
                    edges_matrix[i, j] = 1 

                # check if bordering empty cell
                elif mask[i, j] and (mask[i-1, j] == 0 or mask[i+1, j] == 0 or mask[i, j-1] == 0 or mask[i, j+1] == 0):
                    edges_matrix[i, j] = 1

        # TODO: check to see if better to return a sparse matrix, as only edges are highlighted: https://stackoverflow.com/a/36971131
        return edges_matrix
    
    def compute_variance_of_radius(self, edges_matrix: np.ndarray, center: tuple[int]) -> float:
        """
        Computes the radius of an imperfectly drawn circle.
        
        Args:
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

    def calculate_roughness_progression(self) -> np.ndarray[float]:
        """
        Find roughness of the tumor over time.

        Returns:
            np.ndarray[float]: roughness at each timestep.
        """
        steps = len(self.model.N_Ts)
        roughness_values = np.zeros(steps)

        for i in range(steps):
            N_T, Nec = self.model.N_Ts[i], self.model.Necs[i]
            roughness = self.calculate_roughness(N_T, Nec)
            roughness_values[i] = roughness
        
        return roughness_values
    
    def calculate_roughness(self, N_T, Nec) -> float:
        """
        Find roughness of a single snapshot.
        
        Args:
            N_T (ndarray[int]): 2D grid with number of living agents at each point.
            Nec (ndarray[int]): 2D grid with number of necrotic agents at each point.
        
        Returns:
            float: roughness of tumor.
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
            ndarray: array of values representing the radius of the tumor at each time step.
        """
        radial_distance = np.zeros(len(self.model.N_Ts))

        for i, N_T in enumerate(self.model.N_Ts):
            radial_distance[i] = self.radius(N_T)

        return radial_distance

    def radius(self, N_T):
        """
        Radius of a single snapshot.

        Args:
            N_T (ndarray[int]): 2D grid with with number of living agents at each point.

        Returns:
            float: Estimated radius of the tumor.
        """
        mask = N_T > 0
        geographical_center = self.find_geographical_center(mask)
        edges_of_mask = self.get_edges_of_a_mask(mask)

        return self.calculate_average_distance(edges_of_mask, geographical_center)

    def calculate_velocities(self) -> list[float]:
        """
        Calculate velocities for every delta_d timesteps
        
        Returns:
            list: a list of velocities for every delta_d timesteps
        """
        velocities = []
        intervals = self.model.radii[::self.model.delta_d]

        for i in range(1, len(intervals)):
            velocities.append((intervals[i] - intervals[i-1]) / self.model.delta_d)
        
        return velocities

    def velocity_linear_fit(self) -> tuple[float]:
        """
        Perform linear fit on radius progression.

        Returns: 
            float: Fitted average growth velocity of the tumor.
        """
        fit_func = lambda x, a, b: a * x + b
        skip = 100 # skip initial stage of no growth
        popt, pcov = curve_fit(fit_func, xdata = range(skip, len(self.model.N_Ts)), ydata=self.model.radii[skip:])
        
        velocity, offset = popt
        return velocity, offset