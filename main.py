from classes.tumor_growth import TumorGrowth
from classes.tumor_cell import TumorCell
import matplotlib.pyplot as plt

def main():
    model = TumorGrowth()
    model.degredation()

    plt.imshow(model.ecm_layer.data)
    plt.title('ECM field')
    plt.colorbar()
    plt.show()

    for i in range(100):
        model.diffusion()

    plt.imshow(model.nutrient_layer.data)
    plt.title('Nutrient field')
    plt.colorbar()
    plt.show()

main()