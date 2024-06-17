from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    model = TumorGrowth(50, 50, seed=913)
    model.run_simulation(steps=2000)

    model.plot_all(position=[0, 250, -1])
    # model.plot_birth_deaths()
    # model.plot_max_nutrient()
    # model.plot_max_count()
    # model.plot_radial_distance()
    # model.plot_roughness()

    model.plot_cell_types()
    model.plot_proportion_cell_types()

    #uncomment if you want to save or load the results to a file. timestamp is the time the save_simulation_results_to_file() function was called. If using load_simulation_data_from_file() later (eg during visualization or analysis), make sure to use the correct timestamp and manually hardcode the value 
    # timestamp = model.save_simulation_results_to_file()
    # timestamp = "1718623603"
    # model.load_simulation_data_from_file(timestamp)

if __name__ == "__main__":
    main()