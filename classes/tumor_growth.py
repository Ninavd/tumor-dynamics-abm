import copy 

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer 
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
import numpy as np
from math import e
import matplotlib.pyplot as plt
from classes.tumor_cell import TumorCell
import time as time
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

        self.proliferating_cells = [1]
        self.invasive_cells = [0]
        self.necrotic_cells = [0]

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
            if agent.state != 'necrotic':
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

        count_proliferating = 0
        count_invasive = 0
        count_necrotic = 0
    
        for contents, _ in self.grid.coord_iter():
            for agent in contents:
                if agent.state == 'proliferating':
                    count_proliferating += 1
                elif agent.state =='invasive':
                    count_invasive += 1
                elif agent.state =='necrotic':
                    count_necrotic += 1
            
        self.proliferating_cells.append(count_proliferating)
        self.invasive_cells.append(count_invasive)
        self.necrotic_cells.append(count_necrotic)

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
    
    # def plot_NT(self):
    #     plt.imshow(self.N_T)
    #     plt.title('tumor cells')
    #     plt.colorbar()
    #     plt.show()

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
        # TODO: check to see if it makes more sense to return this as a sparse matrix, as only the edges are highlighted, so it might be sparse enough for large grids? https://stackoverflow.com/a/36971131
        return edges_matrix
        
    def plot_roughness(self):
        """
        Plot the roughness of the tumor.
        """
        roughness_values = []

        for i in range(len(self.N_Ts)):
            mask = self.N_Ts[i] > 0
            N_t = self.cells_at_tumor_surface(self.N_Ts[i], mask)
            center = self.find_geographical_center(mask)
            variance = self.compute_variance_of_roughness(mask, center)
            if N_t == 0:
                roughness_values.append(0)
            else: 
                roughness = np.sqrt(variance / N_t)
                roughness_values.append(roughness)
            #print(f"Roughness of the imperfect circle: {roughness}")
        plt.plot(roughness_values)
        plt.title('Roughness of the Tumor')
        plt.xlabel('Iteration')
        plt.ylabel(r'Roughness $\sqrt{ \frac{1}{N_T} \sum_{i=1}^{N} (r_i-r_0)^{2}}$')
        plt.show()
    
    def compute_variance_of_roughness(self, mask, center):
        """
        Computes the roughness of an imperfectly drawn circle.
        
        Parameters:
        r_theta (function): A function representing the radius r(θ).
        r0 (float): The average radius of the imperfect circle.
        num_points (int): Number of points to sample along the circle.
        
        Returns:
        float: The roughness R of the imperfect circle.
        """
        edge_points = np.argwhere(mask == 1)
        radii = np.linalg.norm(edge_points - center, axis=1)
        r0 = np.mean(radii)
        variance_r = np.mean((radii - r0) ** 2)
        return variance_r
    
    def cells_at_tumor_surface(self, N_t, mask):
        """
        Compute the number of cells at the tumor surface.

        Args:
            mask (np.ndarray): Binary mask matrix.
            iteration (int): Iteration number.

        Returns:
            int: Number of cells at the tumor surface.
        """
        edges_matrix = self.get_edges_of_a_mask(mask)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if edges_matrix[i, j]:
                    edges_matrix[i, j] = N_t[i, j]
        return np.sum(edges_matrix)

    def save_simulation_results_to_file(self):
        """
        Save simulation results to a file. Namely the parameters in a txt file and the results stored in self.ECM, Nutrient, N_T, Births, Deaths lists as a pkl file. 

        Returns:
            string: timestamp of when the files were saved.
        """
        # Save parameters to a readable txt file
        timestamp = str(time.time()).split('.')[0]
        with open(f'save_files/simulation_parameters_{timestamp}.txt', 'w') as f:
            f.write(f"Seed:{self.seed}\n")
            f.write(f"Height:{self.height}\n")
            f.write(f"Width:{self.width}\n")
            f.write(f"Total_Number_of_Births:{self.number_births}\n")
            f.write(f"Total_Number_of_Deaths:{self.number_deaths}\n")
            f.write(f"k:{self.k}\n")
            f.write(f"tau:{self.tau}\n")
            f.write(f"gamma:{self.gamma}\n")
            f.write(f"D:{self.D}\n")
            f.write(f"h:{self.h}\n")
            f.write(f"lam:{self.lam}\n")
            f.write(f"phi_c:{self.phi_c}\n")
            f.write(f"Number_Iterations:{len(self.N_Ts) - 1}\n")

        # Save simulation data (ecm data, nutrient data, tumor cell data, deahts and births data to a npy file
        with open(f'save_files/ecm_layers_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.ecm_layers)
        with open(f'save_files/nutrient_layers_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.nutrient_layers)
        with open(f'save_files/n_ts_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.ecm_layers)
        with open(f'save_files/births_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.births)
        with open(f'save_files/deaths_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.deaths)
    
        print(f"Simulation data saved to file with timestamp: {timestamp}")
        return timestamp

    def load_simulation_data_from_file(self, timestamp):
        """Loads simulation data from file.

        Args:
            timestamp (string): timestmap that the files were originally saved at (see filename you want to upload to find this value)
        """
        timestamp = str(timestamp)
        parameter_values = []

        with open(f'save_files/simulation_parameters_{timestamp}.txt', 'r') as f:
            for line in f:
                parameter_values.append(line.split(':')[1].split('\n')[0])
            self.seed = float(parameter_values[0])
            self.height = float(parameter_values[1])
            self.width = float(parameter_values[2])
            self.number_births = float(parameter_values[3])
            self.number_deaths = float(parameter_values[4])
            self.k = float(parameter_values[5])
            self.tau = float(parameter_values[6])
            self.gamma = float(parameter_values[7])
            self.D = float(parameter_values[8])
            self.h = float(parameter_values[9])
            self.lam = float(parameter_values[10])
            self.phi_c = float(parameter_values[11])

        with open(f'save_files/ecm_layers_data_{timestamp}.npy', 'rb') as f:
            self.ecm_layers = np.load(f)
        with open(f'save_files/nutrient_layers_data_{timestamp}.npy', 'rb') as f:
            self.nutrient_layers = np.load(f)
        with open(f'save_files/n_ts_data_{timestamp}.npy', 'rb') as f:
            self.N_Ts = np.load(f)
        with open(f'save_files/births_data_{timestamp}.npy', 'rb') as f:
            self.births = np.load(f)
        with open(f'save_files/deaths_data_{timestamp}.npy', 'rb') as f:
            self.deaths = np.load(f)
            
    def plot_cell_types(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.proliferating_cells, label = 'Proliferative Cells')
        ax1.plot(self.invasive_cells, label = 'Invasive Cells')
        ax1.plot(self.necrotic_cells, label = 'Necrotic Cells')
        plt.legend()
        plt.show()

    def plot_proportion_cell_types(self):
        fig, ax1 = plt.subplots()
        sum_count = np.array([np.sum(N_T) for N_T in self.N_Ts])
        # relative_proliferating = [self.proliferating_cells[i]/sum_count[i] for i in range(len(self.proliferating_cells))]
        # relative_invasive = [self.invasive_cells[i]/sum_count[i] for i in range(len(self.proliferating_cells))]
        # relative_necrotic = [self.necrotic_cells[i]/sum_count[i] for i in range(len(self.proliferating_cells))]
        ax1.plot(np.array(self.proliferating_cells)/sum_count, label = 'Proliferative Cells')
        ax1.plot(np.array(self.invasive_cells)/sum_count, label = 'Invasive Cells')
        ax1.plot(np.array(self.necrotic_cells)/sum_count, label = 'Necrotic Cells')
        plt.legend()
        plt.show()

