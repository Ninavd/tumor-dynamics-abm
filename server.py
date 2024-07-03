import mesa
import random

from classes.tumor_growth import TumorGrowth
from mesa.datacollection import DataCollector

class TumorWrapper(TumorGrowth):
    def __init__(self, height=201, width=201, steps=1000, delta_d=100, D=1 * 10 ** -4, k=0.02, gamma=5 * 10 ** -4, phi_c=0.02, theta_p=0.2, theta_i=0.2, app=-0.1, api=-0.02, bip=0.02, bii=0.1, seed=random.randint(0, 1000), distribution=False):
        distribution = 'voronoi' if distribution == True else 'uniform'

        super().__init__(height, width, steps, delta_d, D, k, gamma, phi_c, theta_p, theta_i, app, api, bip, bii, seed, distribution)
        self.datacollector = DataCollector(
            {"proliferative":lambda m: m.proliferating_cells[-1],
              "invasive":    lambda m: m.invasive_cells[-1], 
              "necrotic":    lambda m: m.necrotic_cells[-1]
            }
            )
    
    def step(self):
        for i in range(5):
            self.degredation()
            self.diffusion()
            self.cell_death()
            self.new_state()
            self.scheduler.step()
        
        self.count_states()
        self.datacollector.collect(self)

        if self.touches_border():
            self.running = False

def agent_portrayal(agent):
    """
    Portrayal Method for canvas
    """
    if agent is None:
        return
    agents_in_cell = len(agent.model.grid.get_cell_list_contents([agent.pos]))
    radius = 0.2 * agents_in_cell
    radius = radius if radius <= 0.95 else 0.95
    portrayal = {"Shape": "rect", "w": radius, "h":radius, "Filled": "true", "Layer": 2}

    if agent.state == 'proliferating':
        portrayal["Color"] = 'Green'
        portrayal['Layer'] = 2
    else:
        portrayal["Color"] = 'Red'
        portrayal['Layer'] = 1
    return portrayal

model_params = {
    "height": 50,
    "width": 50,
    "app":  mesa.visualization.Slider(name="proliferation inhibition", value=-0.1, min_value=-1, max_value=1.0, step=0.1),
    "bii":  mesa.visualization.Slider(name="invasiveness enhancement", value=0.1, min_value=0.00, max_value=1.0, step=0.1),
    "phi_c":mesa.visualization.Slider(name="death threshold", value=0.02, min_value=0.00, max_value=0.1, step=0.02),
    "gamma":mesa.visualization.Slider(name="ECM degradation speed", value=5*10**-4, min_value=0.00, max_value=0.005, step=0.0005),
    "distribution":mesa.visualization.Choice("distribution of healthy tissue (ECM)", value="random", choices=["random", "voronoi"])
}

grid = mesa.visualization.CanvasGrid(agent_portrayal, model_params['height'], model_params['width'], 400, 400)


# Create a dynamic linegraph
chart = mesa.visualization.ChartModule(
    [
    {"Label": "proliferating", "Color": "green"},
    {"Label": "invasive", "Color": "red"},
    {"Label":"necrotic",  "Color":"blue"}
    ],
    data_collector_name='datacollector')


# text = mesa.visualization.TextData("Progression of cell types") #TextElement("")
class HappyElement(mesa.visualization.TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Progression of cell types"
    
description = HappyElement()

class textElement(mesa.visualization.TextElement):
    def __init__(self):
        pass

    def render(self, model):
        return "Tumor growth model"
    
description2 = textElement()


server = mesa.visualization.ModularServer(TumorWrapper, [description2, grid, description, chart], "Tumor Growth", model_params)
server.launch()
