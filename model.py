import random

from mesa import Model, Agent
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np

class TumorCell(Agent):
    '''
    This class is for the use of agent Tumor cell
    '''
    def __init__(self):
        super().__init__()
        self.state = 'proliferating'
        self.age = 0
        self.health = 1

    def step(self):
        '''
        This method should implement the next step and determine if the tumor cell will migrate or proliferate (or changes to necrotic state).
        '''
        # get nutrients of the cell?
        # get probabilites to determine self.state
        # get neighbouring tumor cells states
        # get neighbors with 0 ECM
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        
        
        
        if self.state == 'proliferating':
            # open_cell refers to the chosen cell which has a ECM of 0 and is open to migrate or prolifarate to.
            self.proliferate(open_cell)
        elif self.state == 'migrating':
            self.migrate(open_cell)
        elif self.state == 'necrotic':
            self.die()
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

        


