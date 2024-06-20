from mesa import Agent
import numpy as np
from math import e
import random 
import sys

class TumorCell(Agent):
    '''
    This class is for the use of agent Tumor cell. 
    input: state (str) - the state of the tumor cell. Options are: proliferating, invasive, necrotic.
    '''
    def __init__(self, state, unique_id, model, seed):
        super().__init__(unique_id, model)

        self.state = state
        self.next_state = self.state

        self.seed = seed
        np.random.seed(self.seed)
        
        # self.theta = 0.2
        self.nutrient_threshold = 0.02

    def generate_next_state(self, nutrient_score):
        """
        Alter state with probability p. 

        Args:
            nutrient_score (float): nutrient concentration in current cell.
        """
        probability_of_proliferate = self.probability_proliferate(nutrient_score)
        probability_of_invasion = self.probability_invasion(nutrient_score)
        normalized_proliferate = probability_of_proliferate / (probability_of_proliferate + probability_of_invasion)
        normalized_invasion = 1 - normalized_proliferate

        random_value = np.random.random()
        if random_value < normalized_proliferate:
            self.next_state = 'proliferating'
        else:
            self.next_state = 'invasive'

    def p_proliferate_invasive(self, nutrient_score, which):
        """
        Return probability of the tumor cell be(com)ing proliferative/invasive.

        Args:
            nutrient_score (float): nutrient concentration in current cell.
            which (str): which probability to return, invasive or proliferate.
        """
        assert which in ['invasive', 'proliferate'], 'which must be invasive or proliferate' 
        proliferate = (which == 'proliferate')
        theta = self.model.theta_p if proliferate else self.model.theta_i
        left = e**(-(nutrient_score / (self.get_N_T() * theta))**2)
        left = 1 - left if proliferate else left
        
        right = 1
        neighboring_cells = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True if proliferate else False) 
        
        for neighbor in neighboring_cells:
            if neighbor.state == 'proliferating' and neighbor != self:
                right *= 1 + self.model.app if proliferate else 1 + self.model.bip
            elif neighbor.state == 'invasive' and neighbor != self: 
                right *= 1 + self.model.api if proliferate else 1 + self.model.bii

        return left * right

    def probability_proliferate(self, nutrient_score):
        """
        Return probability of the tumor cell be(com)ing proliferative.

        Args:
            nutrient_score (float): nutrient concentration in current cell.
        """
        return self.p_proliferate_invasive(nutrient_score, which='proliferate')

    def probability_invasion(self, nutrient_score):
        """
        Return probability of the tumor cell be(com)ing invasive.

        Args:
            nutrient_score (float): nutrient concentration in current cell.
        """
        return self.p_proliferate_invasive(nutrient_score, which='invasive')
    
    def step(self, nutrient_grid):
        """
        Exhibit proliferative or invasive behavior.
        """
        self.state = self.next_state
        if self.state == 'invasive':
            self.invade(nutrient_grid)
        elif self.state == 'proliferating':
            self.proliferate(nutrient_grid)
        
    def invade(self, nutrient_grid):
        """
        Migrate to neighboring site if possible.
        """
        best_cell = self.get_best_neighbor_site(nutrient_grid)
        self.model.displace_agent(self, new_pos=best_cell) # TODO: move cells simultaneously?

    def proliferate(self, nutrient_grid):
        """
        Create daughter cell and place on grid.
        """
        best_cell = self.get_best_neighbor_site(nutrient_grid)
        self.model.add_agent('proliferating', self.model.next_id(), best_cell)

    def die(self):
        """
        State of tumor cell is set to necrotic.
        """
        self.state = 'necrotic'
        self.model.grid.remove_agent(self)
        self.remove()
    
    def get_N_T(self):
        """
        Number of living tumour cells at current position.
        """
        N_T = 0
        for cell in self.model.grid.get_cell_list_contents(self.pos):
            if cell.state != 'necrotic':
                N_T += 1
        
        return N_T
    
    def get_best_neighbor_site(self, nutrient_grid) -> tuple[int, int]:
        """
        Find optimal neighboring site for migration/daughter cell.
        """
        zero_ECM_sites = self.get_empty_ecm_sites()
        open_cell = self.pos
        lowest_amount = sys.maxsize

        for ecm_0 in zero_ECM_sites:
            number_of_agents = self.model.N_T[ecm_0] + self.model.Nec[ecm_0]

            # prefer cells with low tumor density
            if number_of_agents < lowest_amount:
                lowest_amount = number_of_agents
                open_cell = ecm_0

            # if multiple have lowest density, choose on nutrients
            elif number_of_agents == lowest_amount:
                if nutrient_grid.data[open_cell] < nutrient_grid.data[ecm_0]:
                    open_cell = ecm_0
                elif nutrient_grid.data[open_cell] == nutrient_grid.data[ecm_0]:
                    open_cell = random.choice((open_cell, ecm_0))

        return open_cell
    
    def get_empty_ecm_sites(self) -> list[tuple[int, int]]:
        """
        Find all neighboring sites with zero ECM.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        zero_ECM_neighbors = []

        for neighbor in neighbors:

            if self.model.ecm_layer.data[neighbor] == 0:
                zero_ECM_neighbors.append(neighbor)

        return zero_ECM_neighbors