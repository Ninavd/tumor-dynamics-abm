from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
from classes.tumor_visualizations import TumorVisualization
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    model = TumorGrowth(101, 101, seed=913, distribution='uniform')
    model.run_simulation(steps=1000)
    
    print('FINAL 10 VALUES OF')
    print('proliferating', model.proliferating_cells[-10:-1])
    print('invasive: ', model.invasive_cells[-10:-1])
    print('necrotic: ', model.necrotic_cells[-10:-1])

    visualization = TumorVisualization(model)
    visualization.plot_all(position=[0, 250, -1])
    visualization.plot_necrotic_cells()
    # visualization.plot_birth_deaths()
    # visualization.plot_max_nutrient()
    # visualization.plot_max_count()
    # visualization.plot_radial_distance()
    # visualization.plot_roughness()

    visualization.plot_cell_types()
    visualization.plot_proportion_cell_types()

    #uncomment if you want to save or load the results to a file. timestamp is the time the save_simulation_results_to_file() function was called. If using load_simulation_data_from_file() later (eg during visualization or analysis), make sure to use the correct timestamp and manually hardcode the value 
    # timestamp = model.save_simulation_results_to_file()
    # timestamp = "1718623603"
    # model.load_simulation_data_from_file(timestamp)

if __name__ == "__main__":
    main()