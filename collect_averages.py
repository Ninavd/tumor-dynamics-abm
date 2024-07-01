import argparse
import numpy as np
from classes.tumor_growth import TumorGrowth
from classes.tumor_visualizations import TumorVisualization
from classes.tumor_visualization_helper import TumorVisualizationHelper
import matplotlib.pyplot as plt
import warnings
from helpers import save_timestamp_metadata, build_and_save_animation
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(steps, L, seed, payoff, voronoi):
    
    print(f"Running simulation with the following parameters: \nsteps: {steps}, L: {L}, seed: {seed}, payoff: {payoff}, voronoi: {voronoi}")
    model = TumorGrowth(steps=steps, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, seed=seed, distribution='voronoi' if voronoi else 'uniform')

    _, _, _, steps = model.run_model()
    return model, steps

def calculate_CI(x, z = 1.96):
    stdev = np.std(x, axis=0)
    confidence_interval = z * stdev / sqrt(np.array(x).shape[0])
    return confidence_interval

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
    list_of_proportion_proliferative = []
    list_of_proportion_invasive = []
    list_of_proportion_necrotic = []
    list_of_radius = []
    list_of_roughness = []
    list_of_velocities = []
    

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
        total_cells = np.array(proliferating_cells) + np.array(invasive_cells) + np.array(necrotic_cells)
        proliferating_proportion = np.array(proliferating_cells)/total_cells
        invasive_proportion = np.array(invasive_cells)/total_cells
        necrotic_proportion = np.array(necrotic_cells)/total_cells

        list_of_ecm_layers.append(ecm_layers)
        list_of_nutrient_layers.append(nutrient_layers)
        list_of_N_Ts.append(N_Ts)
        list_of_Necs.append(Necs)
        list_of_births.append(births)
        list_of_deaths.append(deaths)
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

    # average_proliferating_cells = np.mean(list_of_proliferating_cells, axis=0)
    # average_invasive_cells = np.mean(list_of_invasive_cells, axis=0)
    # average_necrotic_cells = np.mean(list_of_necrotic_cells, axis=0)
    # prolif_CI = calculate_CI(list_of_proliferating_cells)
    # invasive_CI = calculate_CI(list_of_invasive_cells)
    # necrotic_CI = calculate_CI(list_of_necrotic_cells)

    # plt.plot(average_proliferating_cells, label='Proliferating')
    # plt.plot(average_invasive_cells, label='Invasive')
    # plt.plot(average_necrotic_cells, label='Necrotic')
    # plt.fill_between([*range(len(average_proliferating_cells))], (average_proliferating_cells-prolif_CI), (average_proliferating_cells+prolif_CI), color='b', alpha=0.1)
    # plt.fill_between([*range(len(average_invasive_cells))], (average_invasive_cells-invasive_CI), (average_invasive_cells+invasive_CI), color='orange', alpha=0.1)
    # plt.fill_between([*range(len(average_proliferating_cells))], (average_necrotic_cells-necrotic_CI), (average_necrotic_cells+necrotic_CI), color='g', alpha=0.1)
    # plt.title(f'Number of Cell Types, Average of {args.n_runs} Runs')
    # plt.xlabel('Iteration')
    # plt.ylabel('Number of Cells')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # # go through all runs and calculate averages of the proportions of cell types
    # average_necrotic_proportion = np.mean(list_of_proportion_necrotic, axis=0)
    # average_prolif_proportion = np.mean(list_of_proportion_proliferative, axis=0)
    # average_invasive_proportion = np.mean(list_of_proportion_invasive, axis=0)

    # necrotic_proportion_CI = calculate_CI(list_of_proportion_necrotic)
    # prolif_proportion_CI = calculate_CI(list_of_proportion_proliferative)
    # invasive_proportion_CI = calculate_CI(list_of_proportion_invasive)

    # plt.plot(average_prolif_proportion, label='Proliferating')
    # plt.plot(average_invasive_proportion, label='Invasive')
    # plt.plot(average_necrotic_proportion, label='Necrotic')
    # plt.fill_between([*range(len(average_prolif_proportion))], (average_prolif_proportion-prolif_proportion_CI), (average_prolif_proportion+prolif_proportion_CI), color='b', alpha=0.1)
    # plt.fill_between([*range(len(average_invasive_proportion))], (average_invasive_proportion-invasive_proportion_CI), (average_invasive_proportion+invasive_proportion_CI), color='orange', alpha=0.1)
    # plt.fill_between([*range(len(average_necrotic_proportion))], (average_necrotic_proportion-necrotic_proportion_CI), (average_necrotic_proportion+necrotic_proportion_CI), color='g', alpha=0.1)
    # plt.title(f'Proportion of Cell Types, Average of {args.n_runs} runs')
    # plt.xlabel('Iteration')
    # plt.ylabel('Proportion')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # average_radius = np.mean(list_of_radius, axis=0)
    # radius_CI = calculate_CI(list_of_radius)
    # plt.plot(average_radius, label = 'Radius')
    # plt.fill_between([*range(len(average_radius))], (average_radius-radius_CI), (average_radius+radius_CI), color='b', alpha=0.1)
    # plt.title(f'Average Radial Distance From Tumor Center to Tumor Edge, Average of {args.n_runs} Runs')
    # plt.xlabel('Iteration')
    # plt.ylabel('Average Radial Distance')
    # plt.legend()
    # plt.grid()
    # plt.show()

    # average_roughness = np.mean(list_of_roughness, axis=0)
    # roughness_CI = calculate_CI(list_of_roughness)
    # plt.plot(average_roughness, label = 'Roughness')
    # plt.fill_between([*range(len(average_roughness))], (average_roughness-roughness_CI), (average_roughness+roughness_CI), color='b', alpha=0.1)
    # plt.title(f'Average Roughness of Tumor Edge, Average of {args.n_runs} Runs')
    # plt.xlabel('Iteration')
    # plt.ylabel('Average Roughness')
    # plt.legend()
    # plt.grid()
    # plt.show()

   
    # plot the average velocity of the tumor over time
    # go through all runs and calculate averages of the proportions of cell types
    print(list_of_velocities)
    average_velocity = np.mean(list_of_velocities, axis=0)
    velocity_CI = calculate_CI(average_velocity)
    plt.plot(average_velocity, label='Velocity')
    plt.fill_between([*range(len(average_velocity))], (average_velocity-velocity_CI), (average_velocity+velocity_CI), color='b', alpha=0.1)
    plt.title('Average Velocity of the Tumor Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Average Velocity of the Tumor')
    plt.grid()
    plt.show()
