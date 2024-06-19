import argparse
from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
from classes.tumor_visualizations import TumorVisualization
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(steps, L, seed, payoff, voroni, summary, save):
    
    model = TumorGrowth(payoff, L, L, seed, distribution='voroni' if voroni else 'uniform')

    model.run_simulation(steps=steps)
    
    print('\n FINAL 10 VALUES OF')
    print('proliferating', model.proliferating_cells[-10:-1])
    print('invasive: ', model.invasive_cells[-10:-1])
    print('necrotic: ', model.necrotic_cells[-10:-1])

    # visualization = TumorVisualization(model)
    # visualization.plot_all(position=[0, 250, -1])
    # visualization.plot_necrotic_cells()

    # visualization.plot_birth_deaths()
    # visualization.plot_max_nutrient()
    # visualization.plot_max_count()
    # visualization.plot_radial_distance()
    # visualization.plot_roughness()

    # visualization.plot_cell_types()
    # visualization.plot_proportion_cell_types()

    #uncomment if you want to save or load the results to a file. timestamp is the time the save_simulation_results_to_file() function was called. If using load_simulation_data_from_file() later (eg during visualization or analysis), make sure to use the correct timestamp and manually hardcode the value 
    if save:
        timestamp = model.save_simulation_results_to_file()
        # timestamp = "1718623603"
        model.load_simulation_data_from_file(timestamp)
        visualization = TumorVisualization(model)
        visualization.plot_all()



if __name__ == "__main__":
    # set-up parsing command line arguments
    parser = argparse.ArgumentParser(description="Simulate Agent-Based tumor growth and save results")

    # adding arguments
    parser.add_argument("n_steps", help="max number of time steps used in simulation", default=1000, type=int)
    parser.add_argument("L_grid", help="Width of grid in number of cells", default=1001, type=int)
    parser.add_argument("-s", "--seed", help="provide seed of simulation", default=913, type=int)
    parser.add_argument("-api", "--alpha_pi", help="proliferative probability change when encountering an invasive cell", default=-0.02, type=float)
    parser.add_argument("-app", "--alpha_pp", help="proliferative probability change when encountering an proliferative cell", default=-0.1, type=float)
    parser.add_argument("-bii", "--beta_ii", help="invasive probability change when encountering an invasive cell", default=0.1, type=float)
    parser.add_argument("-bip", "--beta_ip", help="invasive probability change when encountering an proliferative cell", default=0.02, type=float)
    parser.add_argument('--voroni', action="store_true", help="Initialize ECM grid as voroni diagram instead of uniform")
    parser.add_argument('--summary', action="store_true", help="print summary of simulation results")
    parser.add_argument("--save", action="store_true", help="store results in npy files")

    # read arguments from command line
    args = parser.parse_args()

    payoff = [
        [args.alpha_pp, args.alpha_pi], 
        [args.beta_ip, args.beta_ii]
        ]

    # run main with provided arguments
    main(
        args.n_steps, args.L_grid, args.seed, payoff, args.voroni, args.summary, args.save
        ) 