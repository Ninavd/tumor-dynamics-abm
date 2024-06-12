import random

from mesa import Model, Agent
from mesa.space import MultiGrid, PropertyLayer 
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
from math import e

class TumorCell(Agent):
    '''
    This class is for the use of agent Tumor cell
    '''
    def __init__(self):
        super().__init__()
        self.state = 'proliferating'
        self.age = 0
        self.health = 1
        self.app = 0.1
        self.api = 0.1
        self.bii = 0.1
        self.bip = 0.1
        self.ECM = 1
        self.theta_i = 1
        self.theta_p = 1
        self.nutrient_threshold = 1

    def step(self):
        '''
        This method should implement the next step and determine if the tumor cell will migrate or proliferate (or changes to necrotic state).
        '''
        # get nutrients of the cell?
        # get probabilites to determine self.state
        # get neighbouring tumor cells states
        # get neighbors with 0 ECM
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        list_of_0_ECM_neighbors = []
        for neighbor in neighbors:
            if self.model.grid.get_cell_list_contents(neighbor) == 0:
                list_of_0_ECM_neighbors.append(neighbor)
        open_cell = np.random.shuffle(list_of_0_ECM_neighbors)[0]
        
        if (self.nutrient_threshold > self.model.nutrient_layer[self.pos]):
            self.state = 'necrotic'
        probability_of_proliferate = self.probability_proliferate()/3
        probability_of_invasion = self.probability_invasion()/3
        probability_of_necrotic = self.probability_necrotic()/3
        probability_do_nothing = 1 - probability_of_proliferate - probability_of_invasion - probability_of_necrotic

        ranodm_value = np.random(seed = 1)
        if ranodm_value < probability_of_proliferate:
            self.state = 'proliferating'
        elif ranodm_value < probability_of_proliferate + probability_of_invasion:
            self.state = 'migrating'
        elif ranodm_value < probability_of_proliferate + probability_of_invasion + probability_of_necrotic:
            self.state = 'necrotic'
        else:
            self.state = 'stationary'

        if self.state == 'proliferating':
            # open_cell refers to the chosen cell which has a ECM of 0 and is open to migrate or prolifarate to.
            self.proliferate(open_cell)
        elif self.state == 'migrating':
            self.invade(open_cell)
        elif self.state == 'necrotic':
            self.die()
        elif self.state == 'stationary':
            pass

    def probability_proliferate(self):
        '''
        This method should return the probability of the tumor cell to proliferate.
        '''
        N_T = 0
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        cell_contents.remove(self)
        N_T = len(cell_contents)
        left = (1 - e^(-(self.ECM/(N_T * self.theta_p))**2)) 

        
        right = 1
        for i in range(len(cell_contents)):
            if cell_contents[i].state == 'proliferating':
                right *= 1 + self.app
            if cell_contents[i].state == 'invasion':
                right *= 1 + self.api
        return left * right

    def probability_invasion(self):
        '''
        This method should return the probability of the tumor cell to invade.
        '''
        N_T = 0
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        cell_contents.remove(self)
        N_T = len(cell_contents)
        left = (1 - e^(-(self.ECM/(N_T * self.theta_p))**2)) 

        
        right = 1
        for i in range(len(cell_contents)):
            if (cell_contents[i].state == 'proliferating'):
                right *= 1 + self.bip
            if (cell_contents[i].state == 'invasion'):
                right *= 1 + self.bii
        return left * right

    def probability_necrotic(self):
        '''
        This method should return the probability of the tumor cell to become necrotic.
        '''
        pass
    
    def invade(self, open_cell):
        '''
        This method should implement a migrating step.
        '''
        self.model.grid.move_agent(self, pos = open_cell)

    def proliferate(self, open_cell):
        '''
        This method should implement a proliferation of the tumor cell.
        '''
        self.model.new_agent(TumorCell, open_cell)

    def die(self):
        '''
        This method should implement the death of the tumor cell.
        '''
        self.model.grid._remove_agent(self.pos, self)

    
    

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


    def init_grid(self):
        '''
        This method 
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
