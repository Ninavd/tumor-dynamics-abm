from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    model = TumorGrowth(50, 50, seed=913)
    model.run_simulation(steps=250)

    model.plot_all(position=[0, 250, -1])
    # model.plot_birth_deaths()
    # model.plot_max_nutrient()
    # model.plot_max_count()
    # model.plot_radial_distance()
    # model.plot_roughness()

    model.plot_cell_types()
    model.plot_proportion_cell_types()

    #uncomment if you want to save the results to a file
    # model.save_simulation_results_to_file()
    # model.load_simulation_data_from_file()

if __name__ == "__main__":
    main()