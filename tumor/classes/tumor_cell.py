from mesa import Agent
import numpy as np
from math import e
import random 
import sys

class TumorCell(Agent):
    """
    This class represents a tumor cell agent.

    A tumor cell can be proliferating, invasive or necrotic. Once it becomes necrotic,
    it is removed. Tumor cells can switch between proliferating and invading based on their surroundings.
    They take into account local nutrient and tumor cell density, as well as behavior of neighbors. 
    """


    def __init__(self, state, unique_id, model, seed):
        """
        Initialize TumorCell object.

        Args:
            state (str): initial state of the cell, proliferating or invasive.
            unique_id (int): unique identifier of agent. Required by \'Agent\' ancestor class.
            model (TumorGrowth): model aget is active in. 
            seed (int): to seed the agent.
        """
        super().__init__(unique_id, model)

        self.state: str = state
        self.next_state: str = self.state

        self.seed: int = seed
        np.random.seed(self.seed)
        random.seed(self.seed)

    def generate_next_state(self, nutrient_score):
        """
        Alter state with probability p. 

        Args:
            nutrient_score (float): nutrient concentration in current cell.
        """
        probability_of_proliferate = self.probability_proliferate(nutrient_score)
        probability_of_invasion = self.probability_invasion(nutrient_score)
        normalized_proliferate = probability_of_proliferate / (probability_of_proliferate + probability_of_invasion)

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

        # formula from paper by Chen et al., see https://www.nature.com/articles/srep17992 
        left = e**(-(nutrient_score / (self.model.N_T[self.pos] * theta))**2)
        left = 1 - left if proliferate else left
        
        right = 1
        neighboring_cells = self.model.grid.get_neighbors(self.pos, moore=True, include_center=True if proliferate else False) 
        
        # incorporate behavior of neighbors 
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
    
    def step(self):
        """
        Exhibit proliferative or invasive behavior.
        """
        self.state = self.next_state
        best_cell = self.get_best_neighbor_site(self.model.nutrient_layer)

        if self.state == 'invasive':
            self.invade(best_cell)
        elif self.state == 'proliferating':
            self.proliferate(best_cell) 

    def invade(self, best_cell):
        """
        Migrate to neighboring site if possible.

        Args:
            best_cell (tuple): coordinates of best neighboring grid cell to invade.
        """
        self.model.displace_agent(self, new_pos=best_cell) 

    def proliferate(self, best_cell):
        """
        Create daughter cell and place on grid.

        Args:
            best_cell (tuple): coordinates of best neighboring grid cell to invade.
        """
        self.model.add_agent('proliferating', self.model.next_id(), best_cell)

    def die(self):
        """
        State of tumor cell is set to necrotic and agent is removed.
        """
        self.state = 'necrotic'
        self.model.grid.remove_agent(self)
        self.remove()
    
    def get_best_neighbor_site(self, nutrient_grid) -> tuple[int, int]:
        """
        Find optimal neighboring site for migration/daughter cell.

        Sites with low tumor cells and high nutrient score are preferred.
        Only grid cells with zero ECM (healthy tissue) are considered.

        Returns:
            tuple[int, int]: coordinates of best neighboring site.
        """
        # initial values
        zero_ECM_sites = self.get_empty_ecm_sites()
        open_cell = self.pos
        lowest_amount = sys.maxsize

        # loop over options
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

        Returns:
            list[tuple]: list of coordinates of neighboring zero ECM sites.
        """
        neighbors = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)
        zero_ECM_neighbors = []

        for neighbor in neighbors:

            if self.model.ecm_layer.data[neighbor] == 0:
                zero_ECM_neighbors.append(neighbor)

        return zero_ECM_neighbors