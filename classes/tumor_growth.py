import random
import copy 

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer 
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
from math import e
import matplotlib.pyplot as plt
from classes.tumor_cell import TumorCell

class TumorGrowth(Model):
    '''
    Tumor Growth Model
    '''
    def __init__(self, height = 201, width = 201):
        # height and width still to be adjusted for now smaller values
        super().__init__()

        self.height = height
        self.width = width

        self.center = int((height - 1) /2)

        self.ecm_layer = PropertyLayer("ECM", self.height, self.width, default_value=np.float64(0.0))
        self.nutrient_layer = PropertyLayer("Nutrients", self.height, self.width, default_value=np.float64(1.0))
        self.grid = MultiGrid(self.height, self.width, torus=False, property_layers=[self.ecm_layer, self.nutrient_layer])


        self.N_T = np.zeros((self.height, self.width))
        self.k = 0.02
        self.tau = 1
        self.gamma = 5*10**-4
        self.D = 1*10**-4
        self.h = 0.1
        self.lam = self.D * self.tau / (self.h**2)
        self.phi_c = 0.02

        self.init_grid()

        # Place single proliferative cell in the center
        self.add_agent('proliferating', self.next_id(), (self.center, self.center))

    def init_grid(self):
        '''
        This method initializes the ECM and nutrient field
        '''
        for x in range(self.width):
            for y in range(self.height):
                value = random.uniform(0,1)
                self.ecm_layer.set_cell((x,y), value)
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.nutrient_layer.set_cell((x,y), value=1)
                else:
                    self.nutrient_layer.set_cell((x,y), value=1) # Nutrient grid intialized as 1

        # First tumor cell does not survive when nutrients are initialized like this
        # while self.nutrient_layer.data[self.center, self.center] == 0:
        #     self.diffusion()

    def add_agent(self, state, id, pos):
        tumorcell = TumorCell('proliferating', 0, self)
        self.grid.place_agent(tumorcell, pos)
        self.N_T[pos] += 1

    def degredation(self):
        """
        Attack the ECM.
        """
        # select cells with non-zero ECM
        active_cells = self.ecm_layer.select_cells(lambda data: data != 0)
        
        # degrade ECM
        for x, y in active_cells:
            neighbors = self.grid.get_neighborhood(pos=(x, y), moore = True, include_center = True)
            amount_of_tumor_cells = len(self.grid.get_cell_list_contents(neighbors))
            updated_value = self.ecm_layer.data[x,y] - self.gamma * amount_of_tumor_cells

            # Updated ECM cannot be negative
            updated_value = updated_value if updated_value > 0 else 0 
            self.ecm_layer.set_cell((x,y), updated_value)

    def diffusion(self):
        for j in range(1, self.grid.width-1):
            for l in range(1, self.grid.height-1):
                N_t = len(self.grid.get_cell_list_contents([j,l]))
                value = self.diffusion_equation(N_t, j, l)
                self.nutrient_layer.set_cell((j,l), value)

    def diffusion_equation(self, N_t, x, y):
        # This equation breaks if you don't update after each grid visited and if you dont move from x,y = 0,0 to x,y max (when quation about this ask thomas or kattelijn)
        part1 = (1 - self.k * N_t * self.tau - 2 * self.lam) / (1 + 2 * self.lam) * self.nutrient_layer.data[x, y]
        part2 = self.lam / (1 + 2 * self.lam)
        part3 = self.nutrient_layer.data[x + 1, y] + self.nutrient_layer.data[x, y + 1] + self.nutrient_layer.data[x - 1, y] + self.nutrient_layer.data[x, y - 1]
        return part1 + part2*part3
    
    def cell_death(self):
        # cell_with_agents = self.grid.select_cells(self.grid.is_cell_empty()) #Filter for non-empty gridpoints with non-necrotic cells
        # non_necrotic_cells = self.grid.select_cells([agent for agent in self.grid.get_cell_list_contents(cell_with_agents) if agent.state != 'necrotic'])
        all_cells = self.grid.select_cells({'ECM':lambda ecm: True})
        for x, y in all_cells:
            phi = self.nutrient_layer.data[x, y]
            if self.phi_c > phi:
                for agent in self.grid.get_cell_list_contents([x, y]):
                    agent.die()

    def new_state(self):
        # cell_with_agents = self.grid.select_cells(lambda data: data != 0) 
        
        #if cell_death works implement those selection methods in this function as well
        for agents, coord in self.grid.coord_iter():
            x, y = coord
            phi = self.nutrient_layer.data[x, y]
            for agent in agents: #self.grid.get_cell_list_contents([x,y]):
                agent.generate_next_state(phi)
             
    def cell_step(self):
        #if cell_death works implement those selection methods in this function as well 
        iterator = self.grid.coord_iter()
        for agents, _ in iterator:
            agents = copy.copy(agents)
            for agent in agents: 
                agent.step(self.ecm_layer, self.nutrient_layer)
                #print(self.N_T)


    def step(self):
        # degradation ecm
        self.degredation()
        # nutrient diffusion step
        self.diffusion()
        # determine cell death
        self.cell_death()
        # determine cell proliferation or migration
        self.new_state()
        # update cell distribution
        self.cell_step()

    def run_simulation(self, steps=10):
        for i in range(steps):
            if self.if_touch_border():
                print("Tumor touches border")
                return
            print(self.N_T)
            self.step()
    
    def if_touch_border(self):
        for l in range(self.width-1):
            if sum(self.grid.get_cell_list_contents([l, 0])) > 0:
                return True
            elif sum(self.grid.get_cell_list_contents([0,l])) > 0:
                return True
            elif sum(self.grid.get_cell_list_contents([l,self.width-1])) > 0:
                return True
            elif sum(self.grid.get_cell_list_contents([self.width-1,l])) > 0:
                return True
        return False
    
    def show_ecm(self):
        plt.imshow(self.ecm_layer.data)
        plt.title('ECM field')
        plt.colorbar()
        plt.show()

    def show_nutrients(self):
        plt.imshow(self.nutrient_layer.data)
        plt.title('Nutrient field')
        plt.colorbar()
        plt.show()
    
    def show_tumor(self):
        # plot different cell types w/ diff color?
        plt.imshow(self.N_T > 0)
        plt.colorbar()
        plt.show()