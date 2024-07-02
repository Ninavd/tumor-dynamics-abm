import copy 
import numpy as np
import sys
import time
import pickle

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer 
from scipy.spatial import cKDTree

from classes.tumor_cell import TumorCell
from helpers import select_non_zero
from classes.tumor_visualization_helper import TumorVisualizationHelper as TVH

np.set_printoptions(threshold=sys.maxsize)


class TumorGrowth(Model):
    '''
    Tumor Growth Model
    '''
    def __init__(self, height = 201, width = 201, steps = 1000, delta_d=100,
                D= 1*10**-4, k = 0.02, gamma = 5*10**-4, phi_c= 0.02,
                theta_p=0.2, theta_i=0.2, app=-0.1, api=-0.02, bip=0.02, bii=0.1, 
                seed = 913, distribution= 'uniform'):
        
        super().__init__(seed=seed)
        self.TVH = TVH(self)

        self.height = height
        self.width = width
        self.center = int((self.height - 1) /2)

        self.steps = steps
        self.seed = seed

        self.ecm_layer = PropertyLayer("ECM", self.height, self.width, default_value=np.float64(0.0))
        self.nutrient_layer = PropertyLayer("Nutrients", self.height, self.width, default_value=np.float64(1.0))
        self.grid = MultiGrid(self.height, self.width, torus=False, property_layers=[self.ecm_layer, self.nutrient_layer])

        # model parameters
        self.app, self.api = app, api
        self.bip, self.bii = bip, bii
        self.k = k
        self.tau = 1
        self.gamma = gamma
        self.D = D
        self.h = 0.1
        self.theta_p = theta_p
        self.theta_i = theta_i
        self.lam = self.D * self.tau / (self.h**2)
        self.phi_c = phi_c
        
        # seed the simulation
        np.random.seed(self.seed)
        
        # initialize ECM field
        self.distribution = distribution
        if distribution == 'uniform':
            self.init_uniform_ECM()
        elif distribution == 'voronoi':
            self.init_voronoi_ECM()

        # initialize nutrient field
        self.init_nutrient_layer()

        # keep track of living & necrotic tumor cells
        self.N_T = np.zeros((self.height, self.width))
        self.Nec = np.zeros((self.height, self.width))
        self.number_births = 0
        self.number_deaths = 0

        # place single proliferative cell in the center
        self.add_agent('proliferating', self.next_id(), (self.center, self.center))
        
        # for recording grid snapshots at every iteration
        self.ecm_layers = []
        self.nutrient_layers = []
        self.N_Ts = []
        self.Necs = []
        self.births = []
        self.deaths = [] 
        self.radii = []
        
        self.distances = []
        self.delta_d = delta_d

        self.proliferating_cells = [1]
        self.invasive_cells = [0]
        self.necrotic_cells = [0]

        # save initial state
        self.save_iteration_data() 

    def init_nutrient_layer(self):
        """
        Initializes the nutrient concentration field by sampling 
        from uniform distribution.
        """
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):

                nutrient_value = np.random.uniform(0,1)
                self.nutrient_layer.set_cell((x, y), nutrient_value)

    def init_uniform_ECM(self):
        """
        Initializes the ECM distribution for a uniform distribution.
        """
        for x in range(self.width):
            for y in range(self.height):

                value = np.random.uniform(0,1)
                self.ecm_layer.set_cell((x, y), value)     

    def init_voronoi_ECM(self):
        """
        Initializes the ECM for a voronoi tesselation.
        """
        num_seed_points = round(self.height/2)
        
        seed_points = np.random.rand(num_seed_points, 2)
        seed_points[:, 0] *= self.width
        seed_points[:, 1] *= self.height

        voronoi_kdtree = cKDTree(seed_points)
        
        densities = np.random.rand(num_seed_points)

        grid = np.fromiter(self.grid.coord_iter(), tuple)
        grid = [x[1] for x in grid]

        _, seed_point_regions = voronoi_kdtree.query(grid, k=1)

        for i in range(len(grid)):
            value = densities[seed_point_regions[i]]
            self.ecm_layer.set_cell(grid[i], value)

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
        active_cells = self.nutrient_layer.select_cells(select_non_zero)
        
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
        # This equation breaks if you don't update after each grid visited and if you dont move from x,y = 0,0 to x,y max (when question about this ask thomas or kattelijn)
        part1 = (1 - self.k * N_t * self.tau - 2 * self.lam) / (1 + 2 * self.lam) * self.nutrient_layer.data[x, y]
        part2 = self.lam / (1 + 2 * self.lam)
        part3 = self.nutrient_layer.data[x + 1, y] + self.nutrient_layer.data[x, y + 1] + self.nutrient_layer.data[x - 1, y] + self.nutrient_layer.data[x, y - 1]
        return part1 + part2 * part3
    
    def cell_death(self):
        """
        Make agents necrotic if nutrients below threshold.
        """
        for agent in self.agents:
            phi = self.nutrient_layer.data[agent.pos]
            if phi < self.phi_c:
                self.number_deaths += 1
                self.N_T[agent.pos] -= 1
                self.Nec[agent.pos] += 1
                agent.die()

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
        Agents are activated in random order to prevent uniform directional bias.
        Updates the distribution of agents across the grid.
        """
        for agent in self.agents.shuffle():
            agent.step()

    def count_states(self):
        """
        Iterates through the grid counting the number of cells of each state.
        """
        count_proliferating = 0
        count_invasive = 0
    
        for agent in self.agents:
            if agent.state == 'proliferating':
                count_proliferating += 1
            elif agent.state =='invasive':
                count_invasive += 1

        self.proliferating_cells.append(count_proliferating)
        self.invasive_cells.append(count_invasive)
        self.necrotic_cells.append(self.number_deaths)

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

    def run_model(self):
        """
        Grow tumour for number of steps or until tumour touches border.

        Returns:
            results (tuple) - Radius, number of agents, roughness, growth velocity and timestep 
        """
        for i in range(self.steps):
            print(f'Running... step: {i+1}/{self.steps}         ', end='\r')

            # stop simulation if tumor touches border
            if self.touches_border():
                print("\n Simulation stopped: Tumor touches border")
                results = self.collect_results(i)
                return results
            
            self.step() 
            self.save_iteration_data()
        
        results = self.collect_results(self.steps)
        return results
    
    def touches_border(self) -> bool:
        """
        Returns True if tumor touches border, else False. 
        """
        return sum(self.N_T[:, 0] + self.N_T[0, :] + self.N_T[:, self.height - 1] + self.N_T[self.width - 1, :]) != 0
        
    def save_iteration_data(self):
        """
        Save snapshots of current timestep.
        """
        self.ecm_layers.append(copy.deepcopy(self.ecm_layer.data))
        self.nutrient_layers.append(copy.deepcopy(self.nutrient_layer.data))
        self.N_Ts.append(copy.deepcopy(self.N_T))
        self.Necs.append(copy.deepcopy(self.Nec))
        self.births.append(copy.copy(self.number_births))
        self.deaths.append(copy.copy(self.number_deaths))
    
    def collect_results(self, step):
        """
        Collect final stats of the simulation.
        """
        roughness = self.TVH.calculate_roughness(self.N_Ts[-1], self.Necs[-1])
        self.radii = self.TVH.radius_progression()
        radius = self.TVH.radius(self.N_T)
        velocity = self.TVH.velocity_linear_fit()
        # velocity = np.mean(self.TVH.calculate_velocities())
        n_agents = len(self.agents)
        return radius, n_agents, roughness, velocity, step

    def save_simulation_results_to_file(self):
        """
        Save simulation results to a file. Namely the parameters in a txt file and the results stored in self.ECM, Nutrient, N_T, Births, Deaths lists as a pkl file. 

        Returns:
            string: timestamp of when the files were saved.
        """
        timestamp = str(time.time()).split('.')[0]
        with open(f'save_files/simulation_data_{timestamp}.pickle', 'wb') as f:
            pickle.dump(self, f)
        return timestamp

    def load_simulation_data_from_file(self, timestamp):
        """Loads simulation data from file.

        Args:
            timestamp (string): timestamp that the files were originally saved at (see filename you want to upload to find this value)
        """
        timestamp = str(timestamp)
        with open(f'save_files/simulation_data_{timestamp}.pickle', 'r') as f:
            model = pickle.load(f)
        return model