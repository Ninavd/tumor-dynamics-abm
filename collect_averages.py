import argparse
import numpy as np
from classes.tumor_growth import TumorGrowth
from classes.tumor_visualizations import TumorVisualization
from classes.tumor_visualization_helper import TumorVisualizationHelper
import matplotlib.pyplot as plt
import warnings
import math
from helpers import save_timestamp_metadata, build_and_save_animation
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(steps, L, seed, payoff, voronoi):
    
    print(f"Running simulation with the following parameters: \nsteps: {steps}, L: {L}, seed: {seed}, payoff: {payoff}, voronoi: {voronoi}")
    model = TumorGrowth(steps=steps, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, seed=seed, distribution='voronoi' if voronoi else 'uniform')

    _, _, _, _, steps = model.run_model()
    return model, steps

def calculate_CI(x, z = 1.96):
    stdev = np.std(x, axis=0)
    confidence_interval = z * stdev / math.sqrt(np.array(x).shape[0])
    return confidence_interval

def plot_with_CI(list_of_results, **kwargs):
    """
    Plot the average progression of a list of results with confidence interval.
    """
    mean = np.mean(list_of_results, axis=0)
    plt.plot(mean, **kwargs)
    CI = calculate_CI(mean)
    plt.fill_between(range(len(mean)), mean - CI, mean + CI, alpha=0.1)
    plt.grid()
    plt.xlabel('iteration')

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

    # read arguments from command line
    args = parser.parse_args()

    payoff = [
        [args.alpha_pp, args.alpha_pi], 
        [args.beta_ip, args.beta_ii]
        ]
    list_of_proliferating_cells = []
    list_of_invasive_cells = []
    list_of_necrotic_cells = []

    list_of_proportion_proliferative = []
    list_of_proportion_invasive = []
    list_of_proportion_necrotic = []

    list_of_radius = []
    list_of_roughness = []
    list_of_velocities = []
    
    # run main with provided arguments
    for i in range(args.n_runs):
        model, steps = main(args.n_steps, args.L_grid, np.random.randint(1000), payoff, args.voronoi)
         
        if steps != args.n_steps:
            print(f'Run {i+1} of {args.n_runs} skipped due to early stopping at step {steps}')
            continue

        proliferating_cells = model.proliferating_cells
        invasive_cells = model.invasive_cells
        necrotic_cells = model.necrotic_cells
        total_cells = np.array(proliferating_cells) + np.array(invasive_cells) + np.array(necrotic_cells)

        proliferating_proportion = np.array(proliferating_cells)/total_cells
        invasive_proportion = np.array(invasive_cells)/total_cells
        necrotic_proportion = np.array(necrotic_cells)/total_cells

        list_of_proliferating_cells.append(proliferating_cells)
        list_of_invasive_cells.append(invasive_cells)
        list_of_necrotic_cells.append(necrotic_cells)

        list_of_proportion_proliferative.append(proliferating_proportion)
        list_of_proportion_invasive.append(invasive_proportion)
        list_of_proportion_necrotic.append(necrotic_proportion)

        visualization_helper = TumorVisualizationHelper(model)
        radius = visualization_helper.calculate_radial_distance()
        roughness = visualization_helper.calculate_roughness()
        list_of_radius.append(radius)
        list_of_roughness.append(roughness)

        velocity = visualization_helper.calculate_velocities()
        list_of_velocities.append(velocity)
        
        print(f'Run {i+1} of {args.n_runs} completed\n')
    print(list_of_proportion_proliferative)

    # absolute number of cells plot
    plot_with_CI(list_of_proliferating_cells, label='proliferative')
    plot_with_CI(list_of_invasive_cells, label='invasive')
    plot_with_CI(list_of_necrotic_cells, label='necrotic')
    plt.title(f'Number of Cell Types, Average of {args.n_runs} Runs')
    plt.ylabel('Amount')
    plt.legend()
    plt.grid()
    plt.show()

    # Cell fraction plot
    plot_with_CI(list_of_proportion_proliferative, label='proliferative')
    plot_with_CI(list_of_proportion_invasive, label='invasive')
    plot_with_CI(list_of_proportion_necrotic, label='necrotic')
    plt.title(f'Number of Cell Types, Average of {args.n_runs} Runs')
    plt.ylabel('Fraction')
    plt.legend()
    plt.grid()
    plt.show()

    # Average radius progression
    plot_with_CI(list_of_radius)
    plt.title(f'Average Radial Distance From Tumor Center to Tumor Edge, Average of {args.n_runs} Runs')
    plt.ylabel('$\langle r \\rangle$')
    plt.grid()
    plt.show()

    # Average roughness progression
    plot_with_CI(list_of_roughness)
    plt.title(f'Average Roughness of Tumor Edge, Average of {args.n_runs} Runs')
    plt.ylabel('Average Roughness')
    plt.grid()
    plt.show()

    # Average velocity progression
    plot_with_CI(list_of_velocities)
    plt.title('Average Velocity of the Tumor Over Time')
    plt.ylabel('$\langle v \\rangle$')
    plt.grid()
    plt.show()
