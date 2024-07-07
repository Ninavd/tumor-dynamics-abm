import argparse
import numpy as np
import warnings

from classes.tumor_growth import TumorGrowth
from classes.tumor_visualizations import TumorVisualization
from classes.collect_averages import RunCollection
from helpers import save_timestamp_metadata, build_and_save_animation, print_summary_message

# surpress warning about new PropertyLayer class of mesa
warnings.simplefilter(action='ignore', category=FutureWarning)


def main(steps, L, n_runs, seed, payoff, voronoi, summary, save, show_plot, animate):
    """
    Run agent-based tumor growth model with provided command-line arguments.
    """

    # initialize model
    model = TumorGrowth(steps=steps, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, seed=seed, distribution='voronoi' if voronoi else 'uniform')
    
    # run a single simulation or multiple
    if n_runs == 1:
        radius, _, roughness, velocity, steps_taken = model.run_model()
    else:
        runs = RunCollection(n_runs, steps=steps, app=payoff[0][0], api=payoff[0][1], bip=payoff[1][0], bii=payoff[1][1], width=L, height=L, distribution='voronoi' if voronoi else 'uniform')
        model, results = runs.run()
        steps_taken = steps
        roughness   = results['roughness'][-1]
        radius      = results['radius'][-1]
        velocity    = results['velocity'][-1]
    
    if summary:
        print_summary_message(model, steps_taken, payoff, roughness, radius, velocity)

    if save:
        # save pickle of final tumor object 
        timestamp = model.save_simulation_results_to_file()
        save_timestamp_metadata(model, timestamp)
    
    if show_plot:
        vis = TumorVisualization(model)
        vis.plot_necrotic_cells()
        vis.plot_all()
        vis.plot_cell_types()
        vis.plot_proportion_cell_types()
        vis.plot_tumor_over_time(steps)
        vis.plot_radial_distance()
        vis.plot_roughness()
        vis.plot_velocities() if steps_taken > 200 else None
    
    if animate:
        
        # save animation of tumor to mp4
        n_frames = 100
        stepsize = int(len(model.N_Ts) / n_frames)
        frames = model.N_Ts[::stepsize]
        
        title = f'{steps_taken}_steps_{model.distribution}_ECM_seed{seed}'
        
        build_and_save_animation(frames, title=title, iterations=n_frames)


if __name__ == "__main__":
    # set-up parsing command line arguments
    parser = argparse.ArgumentParser(description="Simulate Agent-Based tumor growth and save results")

    # adding arguments
    parser.add_argument("n_steps", help="max number of time steps used in simulation", default=1000, type=int)
    parser.add_argument("L_grid", help="Width of grid in number of cells", default=201, type=int)
    parser.add_argument("-n", help="How many runs to run. If more than one, averaged results are saved to csv. Default is 1.", default=1, type=int)
    parser.add_argument("-s", "--seed", help="provide seed of simulation", default=np.random.randint(1000), type=int)
    parser.add_argument("-api", "--alpha_pi", help="proliferative probability change when encountering an invasive cell", default=-0.02, type=float)
    parser.add_argument("-app", "--alpha_pp", help="proliferative probability change when encountering an proliferative cell", default=-0.1, type=float)
    parser.add_argument("-bii", "--beta_ii", help="invasive probability change when encountering an invasive cell", default=0.1, type=float)
    parser.add_argument("-bip", "--beta_ip", help="invasive probability change when encountering an proliferative cell", default=0.02, type=float)
    parser.add_argument('--voronoi', action="store_true", help="Initialize ECM grid as voronoi diagram instead of uniform")
    parser.add_argument('--summary', action="store_true", help="print summary of simulation results")
    parser.add_argument("--save", action="store_true", help="store simulation object in pickle file")
    parser.add_argument("--show_plot", action="store_true", help="show plot of final tumor and other parameters")
    parser.add_argument("--animate", action="store_true", help="save animation video of tumor growth")

    # read arguments from command line
    args = parser.parse_args()

    payoff = [
        [args.alpha_pp, args.alpha_pi], 
        [args.beta_ip, args.beta_ii]
        ]

    # run main with provided arguments
    main(
        args.n_steps, args.L_grid, args.n, args.seed, payoff, args.voronoi, args.summary, args.save, args.show_plot, args.animate
        ) 