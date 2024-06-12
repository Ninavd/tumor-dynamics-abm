from mesa import Agent
import numpy as np
from math import e

class TumorCell(Agent):
    '''
    This class is for the use of agent Tumor cell. 
    input: state (str) - the state of the tumor cell. Options are: proliferating, migrating, necrotic, stationary.
    '''
    def __init__(self, state):
        super().__init__()
        self.state = state
        self.age = 0
        self.app = 0.1
        self.api = -0.02
        self.bii = 0.1
        self.bip = 0.02
        self.ECM = 1
        self.theta_i = 1
        self.theta_p = 1
        self.nutrient_threshold = 0.02
        self.chance_of_randomly_dying = 0.01

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

        random_value = np.random(seed = 1)
        if random_value < probability_of_proliferate:
            self.state = 'proliferating'
        elif random_value < probability_of_proliferate + probability_of_invasion:
            self.state = 'migrating'
        elif random_value < probability_of_proliferate + probability_of_invasion + probability_of_necrotic:
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
        return self.chance_of_randomly_dying
    
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
