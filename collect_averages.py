import argparse
import numpy as np
from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
from classes.tumor_visualizations import TumorVisualization
import matplotlib.pyplot as plt
import warnings
import pickle
import time
from helpers import save_timestamp_metadata, build_and_save_animation
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(steps, L, seed, payoff, voronoi):
    
    print(f"Running simulation with the following parameters: \nsteps: {steps}, L: {L}, seed: {seed}, payoff: {payoff}, voronoi: {voronoi}")
    model = TumorGrowth(steps=steps, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, seed=seed, distribution='voronoi' if voronoi else 'uniform')

    _, _, _, steps = model.run_model()
    return model, steps

def proportion_cell_types(N_Ts, Necs, proliferating_cells, invasive_cells, necrotic_cells):
    sum_count = np.array([np.sum(N_T) for N_T in N_Ts])
    sum_count += np.array([np.sum(Nec) for Nec in Necs])

    prolif = np.array(proliferating_cells)/sum_count
    invasive = np.array(invasive_cells)/sum_count
    necrotic = np.array(necrotic_cells)/sum_count
    return prolif, invasive, necrotic

def cells_at_tumor_surface(mask):
    """
    Compute the number of cells at the tumor surface.

    Args:
        mask (np.ndarray): Binary mask matrix.
        iteration (int): Iteration number.

    Returns:
        int: Number of grid cells at the tumor surface.
    """
    edges_matrix = get_edges_of_a_mask(mask)
    return np.sum(edges_matrix), edges_matrix

def find_geographical_center(mask):
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
    return np.round(tuple(weighted_sum / total))

def compute_variance_of_roughness(edges_matrix, center):
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
    radii = np.linalg.norm(edge_points - center, axis=1)
    if (len(radii) == 0):
        return 0
    r0 = np.mean(radii)
    variance_r = np.sum((radii - r0) ** 2)
    return variance_r

def calculate_roughness(N_Ts, Necs):
    roughness_values = []

    for i in range(len(N_Ts)):
        mask = (N_Ts[i] + Necs[i]) > 0
        N_r, edges_matrix = cells_at_tumor_surface(mask)
        center = find_geographical_center(edges_matrix)
        variance = compute_variance_of_roughness(edges_matrix, center)
        if N_r == 0:
            roughness_values.append(0)
        else: 
            roughness = np.sqrt(variance / N_r)
            roughness_values.append(roughness)
        
    return roughness_values

def get_edges_of_a_mask(mask):
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
                    # check if on the edge
                    if i - 1 == -1 or i + 1 == mask.shape[0] or j + 1 == mask.shape[0] or j - 1 == -1:
                        edges_matrix[i, j] = 1 
                    elif mask[i-1, j] == 0 or mask[i+1, j] == 0 or mask[i, j-1] == 0 or mask[i, j+1] == 0:
                        edges_matrix[i, j] = 1
        # TODO: check to see if it makes more sense to return this as a sparse matrix, as only the edges are highlighted, so it might be sparse enough for large grids? https://stackoverflow.com/a/36971131
        
        return edges_matrix

def calculate_radial_distance(N_ts, Necs):
    radial_distance = []
    for i in range(len(N_Ts)):
        mask = N_Ts[i] > 0
        geographical_center = find_geographical_center(mask)
        edges_of_mask = get_edges_of_a_mask(mask)
        radial_distance.append(calculate_average_distance(edges_of_mask, geographical_center))
    return radial_distance

def calculate_average_distance(mask, center):
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

if __name__ == "__main__":
    # set-up parsing command line arguments
    parser = argparse.ArgumentParser(description="Simulate Agent-Based tumor growth and save results")

    # adding arguments
    parser.add_argument("n_steps", help="max number of time steps used in simulation", default=1000, type=int)
    parser.add_argument("L_grid", help="Width of grid in number of cells", default=201, type=int)
    parser.add_argument("n_runs", help="Number of runs to average", default = 50, type=int)
    parser.add_argument("-s", "--seed", help="provide seed of simulation", default=np.random.randint(1000), type=int)
    parser.add_argument("-api", "--alpha_pi", help="proliferative probability change when encountering an invasive cell", default=-0.02, type=float)
    parser.add_argument("-app", "--alpha_pp", help="proliferative probability change when encountering an proliferative cell", default=-0.1, type=float)
    parser.add_argument("-bii", "--beta_ii", help="invasive probability change when encountering an invasive cell", default=0.1, type=float)
    parser.add_argument("-bip", "--beta_ip", help="invasive probability change when encountering an proliferative cell", default=0.02, type=float)
    parser.add_argument('--voronoi', action="store_true", help="Initialize ECM grid as voronoi diagram instead of uniform")
    parser.add_argument("--show_plot", action="store_true", help="show plot of final tumor")

    # read arguments from command line
    args = parser.parse_args()

    payoff = [
        [args.alpha_pp, args.alpha_pi], 
        [args.beta_ip, args.beta_ii]
        ]
    list_of_ecm_layers = []
    list_of_nutrient_layers = []
    list_of_N_Ts = []
    list_of_Necs = []
    list_of_births = []
    list_of_deaths = []
    list_of_proliferating_cells = []
    list_of_invasive_cells = []
    list_of_necrotic_cells = []

    # run main with provided arguments
    for i in range(args.n_runs):
        model, steps = main(
            args.n_steps, args.L_grid, np.random.randint(1000), payoff, args.voronoi) 
        if steps != args.n_steps:
            print(f'Run {i+1} of {args.n_runs} skipped due to early stopping at step {steps}')
            continue
        ecm_layers = model.ecm_layers
        nutrient_layers = model.nutrient_layers
        N_Ts = model.N_Ts
        Necs = model.Necs
        births = model.births
        deaths = model.deaths
        proliferating_cells = model.proliferating_cells
        invasive_cells = model.invasive_cells
        necrotic_cells = model.necrotic_cells

        list_of_ecm_layers.append(ecm_layers)
        list_of_nutrient_layers.append(nutrient_layers)
        list_of_N_Ts.append(N_Ts)
        list_of_Necs.append(Necs)
        list_of_births.append(births)
        list_of_deaths.append(deaths)
        list_of_proliferating_cells.append(proliferating_cells)
        list_of_invasive_cells.append(invasive_cells)
        list_of_necrotic_cells.append(necrotic_cells)

        print(f'Run {i+1} of {args.n_runs} completed')
    
    # go through all runs and calculate averages
    average_proliferating_cells = np.mean(list_of_proliferating_cells, axis=0)
    average_invasive_cells = np.mean(list_of_invasive_cells, axis=0)
    average_necrotic_cells = np.mean(list_of_necrotic_cells, axis=0)

    plt.plot(average_proliferating_cells, label='Proliferating')
    plt.plot(average_invasive_cells, label='Invasive')
    plt.plot(average_necrotic_cells, label='Necrotic')
    plt.title(f'Number of Cell Types, Average of {args.n_runs} Runs')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Cells')
    plt.legend()
    plt.grid()
    plt.show()

    # go through all runs and calculate averages of the proportions of cell types
    prolif_proportion = []
    invasive_proportion = []
    necrotic_proportion = []
    for i in range(len(list_of_N_Ts)):
        prolif, invasive, necrotic = proportion_cell_types(list_of_N_Ts[i], list_of_Necs[i], list_of_proliferating_cells[i], list_of_invasive_cells[i], list_of_necrotic_cells[i])
        prolif_proportion.append(prolif)
        invasive_proportion.append(invasive)
        necrotic_proportion.append(necrotic)
    necrotic_proportion = np.mean(necrotic_proportion, axis=0)
    prolif_proportion = np.mean(prolif_proportion, axis=0)
    invasive_proportion = np.mean(invasive_proportion, axis=0)

    plt.plot(prolif_proportion, label='Proliferating')
    plt.plot(invasive_proportion, label='Invasive')
    plt.plot(necrotic_proportion, label='Necrotic')
    plt.title(f'Proportion of Cell Types, Average of {args.n_runs} runs')
    plt.xlabel('Iteration')
    plt.ylabel('Proportion')
    plt.legend()
    plt.grid()
    plt.show()

    # go through all runs and calculate averages of the diameter
    list_of_radius = []
    list_of_roughness = []
    for i in range(len(list_of_N_Ts)):
        radius = calculate_radial_distance(list_of_N_Ts[i], list_of_Necs[i])
        roughness = calculate_roughness(list_of_N_Ts[i], list_of_Necs[i])
        list_of_radius.append(radius)
        list_of_roughness.append(roughness)
    average_radius = np.mean(list_of_radius, axis=0)
    average_roughness = np.mean(list_of_roughness, axis=0)
    
    plt.plot(average_radius, label = 'Radius')
    plt.title(f'Average Radial Distance From Tumor Center to Tumor Edge, Average of {args.n_runs} Runs')
    plt.xlabel('Iteration')
    plt.ylabel('Average Radial Distance')
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot(average_roughness, label = 'Roughness')
    plt.title(f'Average Roughness of Tumor Edge, Average of {args.n_runs} Runs')
    plt.xlabel('Iteration')
    plt.ylabel('Average Roughness')
    plt.legend()
    plt.grid()
    plt.show()
