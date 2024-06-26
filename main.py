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

def main(steps, dd, L, seed, payoff, voronoi, summary, save, show_plot, animate):
    
    model = TumorGrowth(steps=steps, delta_d=dd, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, seed=seed, distribution='voronoi' if voronoi else 'uniform')

    _, _, _, _, steps = model.run_model()
   
    if summary:
        print(
            f"""
            ****************** SUMMARY ***************************
            * iterations \t - {len(model.N_Ts) -1}
            * grid size \t - {L}x{L}
            * seed \t \t - {seed}
            * payoff matrix \t - {payoff}
            * ECM \t\t - {'voronoi' if voronoi else 'uniform'}
            * final #(proliferating) \t - {model.proliferating_cells[-1]}
            * final #(invasive) \t - {model.invasive_cells[-1]}
            * final #(necrotic) \t - {model.necrotic_cells[-1]}
            * final roughness \t - 
            * final tumor size \t - 
            *******************************************************
            """
        )

    if save:
        # model.save_simulation_results_to_file()
        timestamp = str(time.time()).split('.')[0]
        with open(f'save_files/simulation_data_{timestamp}.pickle', 'wb') as f:
            pickle.dump(model, f)
        save_timestamp_metadata(model, timestamp)
    
    if show_plot:
        vis = TumorVisualization(model)
        # vis.plot_all()
        # vis.plot_cell_types()
        # vis.plot_proportion_cell_types()
        # vis.plot_tumor_over_time(steps)
        vis.plot_radial_distance()
        # vis.plot_roughness()
        # vis.plot_distribution()
        # vis.plot_velocities()
    
    if animate:
        n_frames = 100
        stepsize = int(len(model.N_Ts) / n_frames)
        frames = model.N_Ts[::stepsize]
        build_and_save_animation(frames, title='test', iterations=n_frames)

if __name__ == "__main__":
    # set-up parsing command line arguments
    parser = argparse.ArgumentParser(description="Simulate Agent-Based tumor growth and save results")

    # adding arguments
    parser.add_argument("n_steps", help="max number of time steps used in simulation", default=1000, type=int)
    parser.add_argument("L_grid", help="Width of grid in number of cells", default=201, type=int)
    parser.add_argument("-s", "--seed", help="provide seed of simulation", default=np.random.randint(1000), type=int)
    parser.add_argument("-api", "--alpha_pi", help="proliferative probability change when encountering an invasive cell", default=-0.02, type=float)
    parser.add_argument("-app", "--alpha_pp", help="proliferative probability change when encountering an proliferative cell", default=-0.1, type=float)
    parser.add_argument("-bii", "--beta_ii", help="invasive probability change when encountering an invasive cell", default=0.1, type=float)
    parser.add_argument("-bip", "--beta_ip", help="invasive probability change when encountering an proliferative cell", default=0.02, type=float)
    parser.add_argument("-dd", "--delta_d", help="value at which step intervall the distance is determined", default=100, type=int)
    parser.add_argument('--voronoi', action="store_true", help="Initialize ECM grid as voronoi diagram instead of uniform")
    parser.add_argument('--summary', action="store_true", help="print summary of simulation results")
    parser.add_argument("--save", action="store_true", help="store simulation object in pickle file")
    parser.add_argument("--show_plot", action="store_true", help="show plot of final tumor")
    parser.add_argument("--animate", action="store_true", help="save animation video of tumor growth")

    # read arguments from command line
    args = parser.parse_args()

    payoff = [
        [args.alpha_pp, args.alpha_pi], 
        [args.beta_ip, args.beta_ii]
        ]

    # run main with provided arguments
    main(
        args.n_steps, args.delta_d, args.L_grid, args.seed, payoff, args.voronoi, args.summary, args.save, args.show_plot, args.animate
        ) 