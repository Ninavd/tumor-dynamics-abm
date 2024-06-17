from mesa import Agent
import numpy as np
from math import e

class TumorCell(Agent):
    '''
    This class is for the use of agent Tumor cell. 
    input: state (str) - the state of the tumor cell. Options are: proliferating, invasive, necrotic.
    '''
    def __init__(self, state, unique_id, model, seed):
        super().__init__(unique_id, model)
        self.state = state
        self.seed = seed
        np.random.seed(self.seed)

        self.next_state = self.state
        self.age = 0
        self.app = 0.1
        self.api = -0.02
        self.bii = 0.1
        self.bip = 0.02
        self.ECM = 1
        self.theta_i = 0.2
        self.theta_p = 0.2
        self.nutrient_threshold = 0.02
        self.chance_of_randomly_dying = 0 # change to > 0 if you want to implement random death of a cell
        self.seed = seed

    def generate_next_state(self, nutrient_score):
        probability_of_proliferate = self.probability_proliferate(nutrient_score)
        probability_of_invasion = self.probability_invasion(nutrient_score)
        normalized_proliferate = probability_of_proliferate / (probability_of_proliferate + probability_of_invasion)
        normalized_invasion = probability_of_invasion / (probability_of_proliferate + probability_of_invasion)

        random_value = np.random.random()

        if normalized_invasion < normalized_proliferate:
            if random_value < normalized_proliferate:
                self.next_state = 'proliferating'
        else:
            if random_value < normalized_invasion:
                self.next_state = 'invasive'

    def probability_proliferate(self, nutrient_score):
        '''
        This method should return the probability of the tumor cell to proliferate.
        '''
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        N_T = len(cell_contents)
        left = 1 - e**(-(nutrient_score/(N_T * self.theta_p))**2)
        
        right = 1
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = True)
        neighboring_cells = self.model.grid.get_cell_list_contents(neighbors)
        for i in range(len(neighboring_cells)):
            if neighboring_cells[i].state == 'proliferating':
                right *= 1 + self.app
            if neighboring_cells[i].state == 'invasive':
                right *= 1 + self.api
        return left * right

    def probability_invasion(self, nutrient_score):
        '''
        This method should return the probability of the tumor cell to invade.
        '''
        cell_contents = self.model.grid.get_cell_list_contents(self.pos)
        N_T = len(cell_contents)
        left = e**(-(nutrient_score/(N_T * self.theta_i))**2)

        right = 1
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        neighboring_cells = self.model.grid.get_cell_list_contents(neighbors)
        for i in range(len(neighboring_cells)):
            if (neighboring_cells[i].state == 'proliferating'):
                right *= 1 + self.bip
            if (neighboring_cells[i].state == 'invasive'):
                right *= 1 + self.bii
        return left * right
    
    def step(self, ecm_grid, nutrient_grid, N_T_copy):
        #print(self.next_state)
        self.state = self.next_state
        if self.state == 'invasive':
            self.invade(ecm_grid, nutrient_grid, N_T_copy)
        elif self.state == 'proliferating':
            self.proliferate(ecm_grid, nutrient_grid, N_T_copy)
        
    def invade(self, ecm_grid, nutrient_grid, N_T_copy):
        '''
        This method should implement a migrating step.
        '''
        # N_T_copy = copy.copy(self.N_T)
        neighbors = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = False)
        list_of_0_ECM_neighbors = []
        for neighbor in neighbors:
            if ecm_grid.data[neighbor] == 0:
                list_of_0_ECM_neighbors.append(neighbor)

        if len(list_of_0_ECM_neighbors) == 0:
            return
        else:
            lowest_amount = 1000
            for ecm_0 in list_of_0_ECM_neighbors:
                # amount_of_agents = N_T_copy[ecm_0]
                amount_of_agents = len(self.model.grid.get_cell_list_contents(ecm_0))
                if amount_of_agents < lowest_amount:
                    lowest_amount = amount_of_agents
                    open_cell = ecm_0
                elif amount_of_agents == lowest_amount:
                    if nutrient_grid.data[open_cell] < nutrient_grid.data[ecm_0]:
                        open_cell = ecm_0
                    elif nutrient_grid.data[open_cell] == nutrient_grid.data[ecm_0]:
                        open_cell = tuple(np.random.choice(open_cell, ecm_0))
        # TODO: move cells simultaneously
        try:
            self.model.displace_agent(self, new_pos=open_cell)
        except ValueError:
            print(open_cell)
            print(type(open_cell))
            return


    def proliferate(self, ecm_grid, nutrient_grid, N_T_copy):
        '''
        This method should implement a proliferation of the tumor cell.
        '''
        # gets neighbors of the cell and appends the ones with 0 ECM to a list
        cells = self.model.grid.get_neighborhood(self.pos, moore = True, include_center = True)
        list_of_0_ECM_cells = []
        for cell in cells:
            if ecm_grid.data[cell] == 0:
                list_of_0_ECM_cells.append(cell)
        lowest_amount = 1000
        # for each cell with 0 ECM, check the amount of agents in the cell and choose the one with the lowest amount. If they are the same, then choose the one with the higher nutrient value.
        open_cell = self.pos
        for ecm_0 in list_of_0_ECM_cells:
            # amount_of_agents = N_T_copy[ecm_0]
            amount_of_agents = len(self.model.grid.get_cell_list_contents(ecm_0))
            if amount_of_agents < lowest_amount:
                lowest_amount = amount_of_agents
                open_cell = ecm_0
            elif amount_of_agents == lowest_amount:
                if nutrient_grid.data[open_cell] < nutrient_grid.data[ecm_0]:
                    open_cell = ecm_0
                elif nutrient_grid.data[open_cell] == nutrient_grid.data[ecm_0]:
                    open_cell = np.random.choice(open_cell, ecm_0)

        self.model.add_agent('proliferating', self.model.next_id(), open_cell) if open_cell else None # TODO: make unique id ! 

    def die(self):
        '''
        This method should implement the death of the tumor cell.
        '''
        self.state = 'necrotic'
        # self.model.remove_agent()
    