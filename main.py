from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt

def main():
    model = TumorGrowth(51, 51)
    model.run_simulation(steps=500)

    model.plot_all()

    model.plot_NT()

if __name__ == "__main__":
    main()