from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt

def main():
    model = TumorGrowth(200, 200)
    model.run_simulation(steps=350)

    model.show_ecm()
    model.show_tumor()

    # model.degredation()

    # model.show_ecm()

    # for i in range(100):
    #     model.diffusion()

    model.show_nutrients()

    # model.show_tumor()

    # model.cell_death()


if __name__ == "__main__":
    main()