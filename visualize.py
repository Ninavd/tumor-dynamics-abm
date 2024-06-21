import argparse
import pickle

from classes.tumor_visualizations import TumorVisualization
from classes.tumor_growth import TumorGrowth

def visualize(run_id, save):
    # model = TumorGrowth(payoff=[[0,0],[0,0]])
    timestamp = run_id
    with open(f'save_files/simulation_data_{timestamp}.pickle', 'rb') as f:
        model = pickle.load(f)
    visualization = TumorVisualization(model)

    visualization.plot_all(position=[0, 250, -1])
    visualization.plot_necrotic_cells()

    visualization.plot_birth_deaths()
    visualization.plot_max_nutrient()
    visualization.plot_max_count()
    visualization.plot_radial_distance()
    visualization.plot_roughness()

    visualization.plot_cell_types()
    visualization.plot_proportion_cell_types()

if __name__ == "__main__":

    run_id = int(input('run_id: '))

    # run main with provided arguments
    visualize(run_id, save=False)