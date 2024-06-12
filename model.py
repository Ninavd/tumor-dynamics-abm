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

    def step(self):
        '''
        This method should implement the next step and determine if the tumor cell will migrate or proliferate (or changes to necrotic state).
        '''
        # get nutrients of the cell?
        # get probabilites to determine self.state
        # get neighbouring tumor cells states
        # get neighbors with 0 ECM
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        for neighbor in neighbors:
            if self.model.grid.get_cell_list_contents(neighbor) == 0:
                open_cell = neighbor
                break
        
        
        if self.state == 'proliferating':
            # open_cell refers to the chosen cell which has a ECM of 0 and is open to migrate or prolifarate to.
            self.proliferate(open_cell)
        elif self.state == 'migrating':
            self.migrate(open_cell)
        elif self.state == 'necrotic':
            self.die()
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

    def probability_migrate(self):
        '''
        This method should return the probability of the tumor cell to migrate.
        '''
        pass

    def probability_necrotic(self):
        '''
        This method should return the probability of the tumor cell to become necrotic.
        '''
        pass
    
    def migrate(self, open_cell):
        '''
        This method should implement a migrating step.
        '''
        self.model.grid.move_agent(self, pos = open_cell)
        pass

    def proliferate(self, open_cell):
        '''
        This method should implement a proliferation of the tumor cell.
        '''
        self.model.new_agent(TumorCell, open_cell)
        pass

    def die(self):
        '''
        This method should implement the death of the tumor cell.
        '''
        self.model.grid._remove_agent(self.pos, self)
        pass

    
    

class TumorGrowth(Model):
    '''
    Tumor Growth Model
    '''
    def __init__(self, height = 21, width = 21):
        #height and width still to be adjusted for now smaller values
        super().__init__()

        self.height = height
        self.width = width


        self.grid = MultiGrid(self.height, self.width)
        self.ecm_layer = PropertyLayer(self.height, self.width)

    def init_grid(self):
        '''
        This method 
        '''
        Proper

    def step(self):
        # degradation ecm
        # diffusion step
        # determine cell death
        # tumor cell steps
        


