from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt

def main():
    model = TumorGrowth(101, 101)
    model.run_simulation(steps=250)

    model.show_ecm()
    model.show_tumor()
    model.show_nutrients()

    print(model.N_T[model.N_T > 0])
    plt.imshow(model.N_T)
    plt.title('tumor cells')
    plt.colorbar()
    plt.show()

if __name__ == "__main__":
    main()