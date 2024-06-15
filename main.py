from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main():
    model = TumorGrowth(51, 51, seed=143)
    model.run_simulation(steps=400)

    model.plot_all(position=[0, 250, -1])

    model.plot_deaths()

    # model.plot_NT()

if __name__ == "__main__":
    main()