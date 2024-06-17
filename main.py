from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    model = TumorGrowth(51, 51, seed=913)
    model.run_simulation(steps=250)

    model.plot_all(position=[0, 125, -1])
    model.plot_NT()
    model.plot_birth_deaths()
    model.plot_max_nutrient()
    model.plot_max_count()
    model.plot_radial_distance()
    model.plot_roughness()

if __name__ == "__main__":
    main()