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
    def __init__(self, height = 401, width = 401, seed = np.random.randint(1000), distribution= 'uniform'):
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
        self.Necs = []
        self.births = []
        self.deaths = []

        self.N_T = np.zeros((self.height, self.width))
        self.Nec = np.zeros((self.height, self.width))
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
        
        if distribution == 'uniform':
            self.init_uni_grid()
        elif distribution == 'voronoi':
            self.init_vor_grid()

        # Place single proliferative cell in the center
        self.add_agent('proliferating', self.next_id(), (self.center, self.center))
        self.save_iteration_data()

    def init_uni_grid(self):
        """
        Initializes the ECM and nutrient field for a uniform distribution.
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
    
    def init_vor_grid(self):
        """
        Initializes the ECM and nutrient field for a voronoi tesselation.
        """
        num_seed_points = 10
        
        seed_points = np.random.rand(num_seed_points, 2)
        seed_points[:, 0] *= self.width
        seed_points[:, 1] *= self.height
        
        densities = np.random.rand(num_seed_points)

        for i in range(len(self.grid)):
            




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
        living_agents = self.agents.select(lambda agent: agent.state != 'necrotic')
        for agent in living_agents:
            phi = self.nutrient_layer.data[agent.pos]
            if phi < self.phi_c:
                agent.die()
                self.number_deaths += 1
                self.N_T[agent.pos] -= 1
                self.Nec[agent.pos] += 1

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
        # N_T_copy = copy.copy(self.N_T) Remove this??
        # NOTE: Ja het idee was om die copy te gebruiken voor N_T, zodat ze bewegen op basis van de huidige cel verdeling
        # ipv dat het steeds verandert, dus om ervoor te zorgen dat N_T in de formules die van de huidige timestep is, voordat ze gingen bewegen
        # maar miss maakt het niet uit omdat ze nu in random volgorde bewegen..

        for agent in self.agents.shuffle():
            if agent.state != 'necrotic':
                agent.step(self.nutrient_layer)


    def count_states(self):
        """
        Iterates through the grid counting the number of cells of each state.
        """
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
        # count number of cells of each state
        self.count_states()

    def run_simulation(self, steps=10):
        """
        Grow tumour for number of steps or until tumour touches border.
        """
        for i in range(steps):
            print(f'Running... step: {i+1}/{steps}', end='\r')
            if self.touches_border():
                print("\n Simulation stopped: Tumor touches border")
                return
            self.step() 
            self.save_iteration_data()
    
    def touches_border(self) -> bool:
        """
        Returns True if tumor touches border, else False. 
        """
        return sum(self.N_T[:, 0] + self.N_T[0, :] + self.N_T[:, self.height - 1] + self.N_T[self.width - 1, :]) != 0
        
    def save_iteration_data(self):
        self.ecm_layers.append(copy.deepcopy(self.ecm_layer.data))
        self.nutrient_layers.append(copy.deepcopy(self.nutrient_layer.data))
        self.N_Ts.append(copy.deepcopy(self.N_T))
        self.Necs.append(copy.deepcopy(self.Nec))
        self.births.append(copy.copy(self.number_births))
        self.deaths.append(copy.copy(self.number_deaths))
    
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
            np.save(f, self.ecm_layers)# does this not need to be self.N_Ts
        with open(f'save_files/necs_data_{timestamp}.npy', 'wb') as f:
            np.save(f, self.Necs)
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
        with open(f'save_files/necs_data_{timestamp}.npy', 'rb') as f:
            self.Necs = np.load(f)
        with open(f'save_files/births_data_{timestamp}.npy', 'rb') as f:
            self.births = np.load(f)
        with open(f'save_files/deaths_data_{timestamp}.npy', 'rb') as f:
            self.deaths = np.load(f)
            