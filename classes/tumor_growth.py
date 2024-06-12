import random

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer 
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
from math import e

class TumorGrowth(Model):
    '''
    Tumor Growth Model
    '''
    def __init__(self, height = 21, width = 21):
        # height and width still to be adjusted for now smaller values
        super().__init__()

        self.height = height
        self.width = width

        self.center = (height - 1) /2 
        self.grid = MultiGrid(self.height, self.width)
        self.ecm_layer = PropertyLayer(self.height, self.width)
        self.nutrient_layer = PropertyLayer(self.height, self.width)

        self.k = 0.02
        self.tau = 1
        self.gamma = 5*10**-4
        self.D = 1*10**-4
        self.h = 0.1
        self.lam = self.D * self.tau / (self.h**2)

        self.init_grid()

    def init_grid(self):
        '''
        This method initializes the ECM and nutrient field
        '''
        for x in range(self.width):
            for y in range(self.height):
                value = random.uniform(0,1)
                self.ecm_layer.set_cell((x,y), value)
                if x == 0 or x == self.width or y == 0 or y == self.height:
                    self.nutrient_layer.set_cell((x,y), value=1)
                else:
                    self.nutrient_layer.set_cell((x,y), value=0)

        while self.nutrient_layer[self.center, self.center] == 0:
            self.diffusion(self)

    def degredation(self):
        # TODO: loop over cells with active agents (non-necrotic) instead of all cells
        for x in range(self.width):
            for y in range(self.height):
                if self.ecm_layer[x,y] != 0:
                    neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = True)
                    amount_of_tumor_cells = len(self.model.grid.get_cell_list_contents([neighbors]))

                    self.ecm_layer.set_cell((x,y), (self.ecm_layer - self.gamma * amount_of_tumor_cells))
                else:
                    pass

    def diffusion(self, N_t):

        for j in range(1, self.grid.width-1):
            for l in range(1, self.grid.height-1):
                N_t = len(self.model.grid.get_cell_list_contents([j,l]))
                value = self.diffusion_equation(N_t, j, l)
                self.nutrient_layer.set_cell((j,l), value)

    def diffusion_equation(self, N_t, x, y):
        # This equation breaks if you don't update after each grid visited and if you dont move from x,y = 0,0 to x,y max (when quation about this ask thomas or kattelijn)
        u_next_step = ((1 + self.k * N_t * self.tau - 2 * self.lam)/ 1 + 2* self.lam) * self.nutrient_layer[x, y] + self.lam / (1 + 2 * self.lam) * (self.nutrient_layer[x + 1, y] + self.nutrient_layer[x, y + 1] + self.nutrient_layer[x - 1, y] + self.nutrient_layer[x, y - 1])
        
        return u_next_step

    def step(self):
        # degradation ecm
        self.degredation(self)
        # nutrient diffusion step
        self.diffusion(self)
        # determine cell death

        # determine cell proliferation or migration
        # update cell distribution
        pass