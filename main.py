from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt

def main():
    model = TumorGrowth()
    model.degredation()

    # model.show_ecm()

    for i in range(100):
        model.diffusion()

    # model.show_nutrients()

    model.show_tumor()

if __name__ == "__main__":
    main()