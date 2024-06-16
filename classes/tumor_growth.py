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
    def __init__(self, height = 201, width = 201, seed = np.random.randint(1000)):
        # height and width still to be adjusted for now smaller values
        super().__init__()

        self.height = height
        self.width = width
        self.seed = seed
        np.random.seed(self.seed)

        self.center = int((height - 1) /2)

        self.ecm_layer = PropertyLayer("ECM", self.height, self.width, default_value=np.float64(0.0))
        self.nutrient_layer = PropertyLayer("Nutrients", self.height, self.width, default_value=np.float64(1.0))
        self.grid = MultiGrid(self.height, self.width, torus=False, property_layers=[self.ecm_layer, self.nutrient_layer])

        self.ecm_layers = []
        self.nutrient_layers = []
        self.N_Ts = []
        self.births = []
        self.deaths = []

        self.N_T = np.zeros((self.height, self.width))
        self.k = 0.02
        self.tau = 1
        self.gamma = 5*10**-4
        self.D = 1*10**-4
        self.h = 0.1
        self.lam = self.D * self.tau / (self.h**2)
        self.phi_c = 0.02
        self.number_births = 0
        self.number_deaths = 0

        self.init_grid()

        # Place single proliferative cell in the center
        self.add_agent('proliferating', self.next_id(), (self.center, self.center))
        self.save_iteration_data()

    def init_grid(self):
        """
        Initializes the ECM and nutrient field.
        """
        for x in range(self.width):
            for y in range(self.height):
                value = np.random.uniform(0,1)
                self.ecm_layer.set_cell((x,y), value)
                if x == 0 or x == self.width - 1 or y == 0 or y == self.height - 1:
                    self.nutrient_layer.set_cell((x,y), value=1)
                else:
                    nutrient_value = np.random.uniform(0,1)
                    self.nutrient_layer.set_cell((x,y), nutrient_value)

        # First tumor cell does not survive when nutrients are initialized like this
        # while self.nutrient_layer.data[self.center, self.center] == 0:
        #     self.diffusion()

    def add_agent(self, state, id, pos):
        """
        Create new agent and update agent distribution.

        Args:
            state (str): necrotic, proliferative or invasive
            id (int): unique id that identifies agent.
            pos (tuple[int, int]): intial x, y position of new agent.
        """
        assert state in ['proliferating', 'invasive', 'necrotic'], 'Invalid state. State must be: necrotic, proliferating or invasive.'
        
        tumorcell = TumorCell(state, id, self, seed=self.seed)
        self.grid.place_agent(tumorcell, pos)
        self.N_T[pos] += 1
        self.number_births += 1
    
    def displace_agent(self, agent: TumorCell, new_pos):
        """
        Move agent and update agent distribution.

        Args:
            agent (TumorCell): cell to move.
            new_pos (tuple): new position of agent.
        """
        self.N_T[agent.pos] -= 1
        self.grid.move_agent(agent, pos = new_pos)
        self.N_T[new_pos] += 1

    def degredation(self):
        """
        Update ECM. Tumor cells attack and lower the ECM.
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
        """
        Update nutrient field.
        """
        for j in range(1, self.grid.width-1):
            for l in range(1, self.grid.height-1):
                N_t = len(self.grid.get_cell_list_contents([j,l]))
                value = self.diffusion_equation(N_t, j, l)
                self.nutrient_layer.set_cell((j,l), value)

    def diffusion_equation(self, N_t, x, y):
        """
        Discretized reaction-diffusion equation.

        Args:
            N_t (int): Total number of agents in current grid cell.
            x, y (int, int): position of grid cell.
        
        Returns:
            (int): Updated nutrient concentration in grid cell
        """
        # This equation breaks if you don't update after each grid visited and if you dont move from x,y = 0,0 to x,y max (when quation about this ask thomas or kattelijn)
        part1 = (1 - self.k * N_t * self.tau - 2 * self.lam) / (1 + 2 * self.lam) * self.nutrient_layer.data[x, y]
        part2 = self.lam / (1 + 2 * self.lam)
        part3 = self.nutrient_layer.data[x + 1, y] + self.nutrient_layer.data[x, y + 1] + self.nutrient_layer.data[x - 1, y] + self.nutrient_layer.data[x, y - 1]
        return part1 + part2 * part3
    
    def cell_death(self):
        """
        Make agents necrotic if nutrients below threshold.
        """
        all_cells = self.grid.select_cells({'ECM':lambda ecm: True})
        for x, y in all_cells:
            phi = self.nutrient_layer.data[x, y]
            if self.phi_c > phi:
                for agent in self.grid.get_cell_list_contents([x, y]):
                    agent.die() # NOTE: self.N_T is not updated! Might be better in the future...
                    self.number_deaths += 1

    def new_state(self):
        """
        Update the state of all agents in random order.
        """
        for agent in self.agents.shuffle():
            phi = self.nutrient_layer.data[agent.pos]
            if agent.state != 'necrotic':
                agent.generate_next_state(phi)
             
    def cell_step(self):
        """
        All agents proliferate or invade (based on current state). 
        Agents activated in random order to prevent directional bias.
        Updates the distribution of agents across the grid.
        """
        # update steps depend on CURRENT cell distribution
        N_T_copy = copy.copy(self.N_T) 

        for agent in self.agents.shuffle():
            agent.step(self.ecm_layer, self.nutrient_layer, N_T_copy)

    def step(self):
        """
        Single simulation step. 
        Updates ECM, nutrients, state of agents and their distribution.
        """
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
        """
        Grow tumour for number of steps or until tumour touches border.
        """
        for i in range(steps):
            print(f'Running... step: {i+1}/{steps}', end='\r')
            if self.if_touch_border():
                print("\n Simulation stopped: Tumor touches border")
                return
            self.step() 
            self.save_iteration_data()
    
    def if_touch_border(self) -> bool:
        """
        Returns True if tumor touches border, else False. 
        """
        # NOTE: this assumes height = width
        # TODO: improve by using self.N_T (ndarray)
        for l in range(self.height):
            if len(self.grid.get_cell_list_contents([l, 0])) > 0:
                return True
            elif len(self.grid.get_cell_list_contents([0,l])) > 0:
                return True
            elif len(self.grid.get_cell_list_contents([l,self.width-1])) > 0:
                return True
            elif len(self.grid.get_cell_list_contents([self.width-1,l])) > 0:
                return True
        return False
    
    def save_iteration_data(self):
        self.ecm_layers.append(copy.deepcopy(self.ecm_layer.data))
        self.nutrient_layers.append(copy.deepcopy(self.nutrient_layer.data))
        self.N_Ts.append(copy.deepcopy(self.N_T))
        self.births.append(copy.copy(self.number_births))
        self.deaths.append(copy.copy(self.number_deaths))
    
    def show_ecm(self, position = -1, show=True):
        """
        Plot current ECM density field.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.ecm_layers[position])
        return fig, axs

    def show_nutrients(self, position = -1, show=True):
        """
        Plot current nutrient concentration field.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.nutrient_layers[position])
        return fig, axs
    
    def show_tumor(self, position = -1, show=True):
        """
        Plot mask of the tumor. Includes necrotic cells.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.N_Ts[position])
        return fig, axs
    
    def plot_all(self, position = -1):
        """
        Plot ECM, nutrient field and tumour in a single figure.
        """
        if type(position) == int:
            positions = [position]
        else:
            positions = position
        
        for position in positions:
            if (position > len(self.ecm_layers) - 1):
                print(f"Position {position} out of range. Max position: {len(self.ecm_layers) - 1}, skipping graphic iteration...")
                continue
            final_fig, final_axs = plt.subplots(1, 3, figsize=(15, 5))

            ecm_fig, ecm_axs = self.show_ecm(position = position)
            nutrient_fig, nutrient_axs = self.show_nutrients(position = position)
            tumor_fig, tumor_axs = self.show_tumor(position = position)

            ecm_axs = final_axs[0].imshow(ecm_axs.get_images()[0].get_array())
            final_axs[0].axis("off")
            final_axs[0].set_title('ECM Field Concentration')
            nutrient_axs = final_axs[1].imshow(nutrient_axs.get_images()[0].get_array())
            final_axs[1].set_title('Nutrient Field Concentration')
            final_axs[1].axis("off")
            tumor_axs = final_axs[2].imshow(tumor_axs.get_images()[0].get_array())
            final_axs[2].set_title('Tumor Cell Count')
            final_axs[2].axis("off")

            plt.close(ecm_fig)
            plt.close(nutrient_fig)
            plt.close(tumor_fig)
            final_fig.colorbar(ecm_axs, ax=final_axs[0], fraction=0.046, pad=0.04)
            final_fig.colorbar(nutrient_axs, ax=final_axs[1], fraction=0.046, pad=0.04)
            final_fig.colorbar(tumor_axs, ax=final_axs[2], fraction=0.046, pad=0.04)
            plt.suptitle(f'ECM, Nutrient, and Tumor Values at Iteration {position%(len(self.ecm_layers))} of {len(self.ecm_layers)-1} for a {self.height}x{self.width} Grid')
            plt.show()

    def plot_NT(self):
        plt.imshow(self.N_T)
        plt.title('tumor cells')
        plt.colorbar()
        plt.show()

    def plot_birth_deaths(self):
        birth_rel_death = [(self.births[i]) /(self.births[i] + self.deaths[i]) for i in range(len(self.births))]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.births, label='Births')
        ax1.plot(self.deaths, label='Deaths')
        ax2.plot(birth_rel_death, label='Births / (Births + Deaths)', color='g')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='green', which='both')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.title('Cumulative Number of Births and Deaths')
        plt.show()
    
    def plot_max_nutrient(self):
        min_nutrient = [np.min(nutrient) for nutrient in self.nutrient_layers]
        sum_nutrient = [np.sum(nutrient) for nutrient in self.nutrient_layers]
        relative_count = [min_nutrient[i]/sum_nutrient[i] for i in range(len(min_nutrient))]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(min_nutrient, label='Min Nutrient Value')
        ax1.plot(sum_nutrient, label='Sum of Nutrient Values')
        ax2.plot(relative_count, 'g', label='Relative Percentage of Max Cells in Grid')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Nutrient Value')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='green', which='both')
        plt.title('Nutrient Value in Grid')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.show()
    
    def plot_max_count(self):
        max_count = [np.max(N_T) for N_T in self.N_Ts]
        sum_count = [np.sum(N_T) for N_T in self.N_Ts]
        relative_count = [max_count[i]/sum_count[i] for i in range(len(max_count))]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(max_count, label='Max Cells in a Subsection of Grid')
        ax1.plot(sum_count, label='Total Number of Cells in Grid')
        ax2.plot(relative_count, 'g', label='Relative Percentage of Max Cells to Total Number of Cells in Grid')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cell Count')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='g', which='both')
        ax2.set_ylabel('Relative Percentage', color='g')
        
        plt.title('Cell Count in Grid')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.show()
    
    def plot_radial_distance(self):
        """
        Plot the radial distance of the tumor from the center of the grid.
        """
        radial_distance = []
        for i in range(len(self.N_Ts)):
            mask = self.N_Ts[i] > 0
            geographical_center = self.find_geographical_center(mask)
            edges_of_mask = self.get_edges_of_a_mask(mask)
            radial_distance.append(self.calculate_average_distance(edges_of_mask, geographical_center))
        plt.plot(radial_distance)
        plt.title('Average Radial Distance form Tumor Center to Tumor Edge')
        plt.show()

    def calculate_average_distance(self, mask, center):
        """
        Calculate the average distance from the center of the mask to the edge.

        Args:
            mask (np.ndarray): Binary mask matrix.

        Returns:
            float: Average distance to the edge.
        """
        distances = []
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    cell = np.array([i, j])
                    distance = np.linalg.norm(cell - center)
                    distances.append(distance)
        return np.mean(distances)
    
    def find_geographical_center(self, mask):
        """
        Find the geographical center of the mask.

        Args:
            mask (np.ndarray): Binary mask matrix.

        Returns:
            tuple: Coordinates of the geographical center.
        """
        total = np.sum(mask)
        weighted_sum = np.array([0, 0])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    cell = np.array([i, j])
                    weighted_sum += cell
        return tuple(weighted_sum // total)

    def get_edges_of_a_mask(self, mask):
        """
        Find the edges of a binary mask.
        
        Args: 
            mask (np.ndarray): Binary mask matrix.
        
        Returns:
            np.ndarray: Binary matrix with only full cells that border an empty cell."""
        edges_matrix = np.zeros(mask.shape)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j]:
                    if mask[i-1, j] == 0 or mask[i+1, j] == 0 or mask[i, j-1] == 0 or mask[i, j+1] == 0:
                        edges_matrix[i, j] = 1
        # TODO: check to see if it makes more sense to return this as a sparse matrix, as only the edges are hightlighted, so it might be sparse enough for large grids? https://stackoverflow.com/a/36971131
        return edges_matrix
        