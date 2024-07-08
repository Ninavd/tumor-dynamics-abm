import copy 
import numpy as np
import time
import pickle

from mesa import Model
from mesa.space import MultiGrid, PropertyLayer 
from mesa.time import RandomActivation
from scipy.spatial import cKDTree

from tumor.classes.tumor_cell import TumorCell
from tumor.classes.tumor_visualization_helper import TumorVisualizationHelper as TVH


class TumorGrowth(Model):
    """
    Tumor Growth Model.

    Simulate tumor growth in healthy tissue for a number of time steps.
    Tumor is initialized with a single proliferative cell in the center of the two-dimensional grid.
    Snapshots of the healthy tissue and nutrient distribution, cell counts and tumor shapes are saved.
    
    Example usage:
        model = TumorGrowth() # initialize model with default parameters. 
        results = model.run_model() # run model for 1000 steps. 

    Attributes: 
        scheduler (RandomActivation): Schedules activation of agents.
        TVH (TumorVisualisationHelper): Calculates properties of the tumor.
        height, width (int, int): Grid size.
        center (int): Center of the grid, assuming grid is square.
        steps (int): Defines maximum number of steps taken in a simulation.
        seed (int): Defines seed of the simulation.
        ecm_layer (PropertyLayer): Represents healthy tissue distribution (ECM).
        nutrient_layer (PropertyLayer): Represents nutrient concentration field.
        grid (MultiGrid): Discretized grid containing the TumorCell agents. 
        app, api, bip, bii: Matrix elements in payoff matrix.
        k (float): Indicates how many nutrients an agent consumes.
        tau (float): Parameter of reaction-diffusion equation.
        gamma (float): Inidicates with what factor an agent degrades the ecm.
        D (float): Diffusion constant.
        theta_i, theta_p (float, float): Shape parameters.
        lam (float): parameter in discretized reaction-diffusion equation.
        phi_c (float): Indicates critical nutrient threshold, below which agents die.
        distribution (str): Defines ECM distribution. Either \'random\' or \'voronoi\'.
        N_T (ndarray): Current distribution of living agents, excluding necrotic agents. 
        Nec (ndarray): Current distribution of necrotic agents.
        number_births (int): total number of births.
        number_deaths (int): total number of deaths.
        ecm_layers (list[ndarray]): Contains snapshots of ECM distribution.
        nutrient_layers (list[ndarray]): Contains snapshots of nutrient distribution.
        N_Ts (list[ndarray]): Contains snapshots of spatial living agent distribution.
        Necs (list[ndarray]): Contains snapshots of spatial necrotic agent distribution. 
        births (list[int]): Records cumulative number of births.
        radii (list[float]): Records radius of tumor at the end of simulation.
        delta_d (int): Time interval to calculate velocity on.
        proliferating_cells (list[int]): Number of proliferating agents at each iteration.
        invasive_cells (list[int]): Number of invasive agents at each iteration.
        necrotic_cells (list[int]): Total number of necrotic agents (deaths).
    """
    def __init__(self, height = 101, width = 101, steps = 1000, delta_d=100,
                D= 1*10**-4, k = 0.02, gamma = 5*10**-4, phi_c= 0.02,
                theta_p=0.2, theta_i=0.2, app=-0.1, api=-0.02, bip=0.02, bii=0.1, 
                seed = 913, distribution= 'uniform'):
        """
        Initializes model based on provided parameters.

        Args:
            height, width (int, int): grid size.
            steps (int): Defines maximum number of steps taken in a simulation.
            seed (int): Defines seed of the simulation.
            app, api, bip, bii: Matrix elements in payoff matrix.
            k (float): Indicates how many nutrients an agent consumes.
            tau (float): Parameter of reaction-diffusion equation.
            gamma (float): Inidicates with what factor an agent degrades the ecm.
            D (float): Diffusion constant.
            theta_i, theta_p (float, float): Shape parameters.
            lam (float): parameter in discretized reaction-diffusion equation.
            phi_c (float): Indicates critical nutrient threshold, below which agents die.
            distribution (str): Defines ECM distribution. Either \'random\' or \'voronoi\'.
        """
        super().__init__(seed=seed)
        self.scheduler = RandomActivation(self)
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
        self.radii = []
        
        self.delta_d = delta_d

        self.proliferating_cells = [1]
        self.invasive_cells = [0]
        self.necrotic_cells = [0]

        # save initial state
        self.save_iteration_data() 

    def init_nutrient_layer(self):
        """
        Initializes the nutrient layer by sampling 
        from a uniform distribution U(0, 1) at each site.
        """
        for x in range(1, self.width - 1):
            for y in range(1, self.height - 1):
                nutrient_value = np.random.uniform(0,1)
                self.nutrient_layer.set_cell((x, y), nutrient_value)

    def init_uniform_ECM(self):
        """
        Initializes the ECM distribution by sampling 
        from a uniform distribution U(0, 1) at each site..
        """
        for x in range(self.width):
            for y in range(self.height):
                value = np.random.uniform(0,1)
                self.ecm_layer.set_cell((x, y), value)     

    def init_voronoi_ECM(self):
        """
        Initializes the ECM as a voronoi tesselation.
        """
        num_seed_points = self.height
        
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

    def add_agent(self, state: str, id: int, pos: tuple[int]):
        """
        Create new agent and add to grid and scheduler.

        Args:
            state (str): \'necrotic\', \'proliferative\' or \'invasive\'.
            id (int): unique identifier of the agent.
            pos (tuple[int, int]): intial x, y position of new agent.
        """
        assert state in ['proliferating', 'invasive', 'necrotic'], 'Invalid state. State must be: necrotic, proliferating or invasive.'
        
        tumorcell = TumorCell(state, id, self, seed=self.seed)
        self.grid.place_agent(tumorcell, pos)
        self.N_T[pos] += 1
        self.number_births += 1
        self.scheduler.add(tumorcell)
    
    def displace_agent(self, agent: TumorCell, new_pos: tuple[int]):
        """
        Move agent and update agent distribution.

        Args:
            agent (TumorCell): Agent to move.
            new_pos (tuple): new position of agent.
        """
        self.N_T[agent.pos] -= 1
        self.grid.move_agent(agent, pos = new_pos)
        self.N_T[new_pos] += 1

    def degredation(self):
        """
        All living tumor cells degrade neighboring tissue (ECM).
        """
        non_empty_cells = np.argwhere(self.N_T > 0)
        for x, y in non_empty_cells:

            neighbors = self.grid.get_neighborhood(pos=(x, y), moore = True, include_center = True)
            for coord in neighbors:

                # Updated ECM cannot be negative
                updated_value = self.ecm_layer.data[coord] - self.gamma * self.N_T[x, y]
                updated_value = updated_value if updated_value > 0 else 0 
                self.ecm_layer.set_cell(coord, updated_value)

    def diffusion(self):
        """
        Update nutrient field with discretized reaction-diffusion equation.
        """
        for j in range(1, self.grid.width-1):
            for l in range(1, self.grid.height-1):
                N_t = self.N_T[j, l]
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
        # This equation breaks if you don't update after each grid visited and if you dont move from x,y = 0,0 to x,y=max
        part1 = (1 - self.k * N_t * self.tau - 2 * self.lam) / (1 + 2 * self.lam) * self.nutrient_layer.data[x, y]
        part2 = self.lam / (1 + 2 * self.lam)
        part3 = self.nutrient_layer.data[x + 1, y] + self.nutrient_layer.data[x, y + 1] + self.nutrient_layer.data[x - 1, y] + self.nutrient_layer.data[x, y - 1]
        return part1 + part2 * part3
    
    def cell_death(self):
        """
        Agents are removed if local nutrients are below threshold.
        """
        for agent in self.agents:
            phi = self.nutrient_layer.data[agent.pos]
            if phi < self.phi_c:
                self.number_deaths += 1
                self.N_T[agent.pos] -= 1
                self.Nec[agent.pos] += 1
                self.scheduler.remove(agent)
                agent.die()

    def new_state(self):
        """
        Update the state of all agents in random order.
        """
        for agent in self.agents.shuffle():
            phi = self.nutrient_layer.data[agent.pos]
            if agent.state != 'necrotic':
                agent.generate_next_state(phi)

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
        self.degredation()
        self.diffusion()
        self.cell_death()
        self.new_state()
        self.scheduler.step()
        self.count_states()

    def run_model(self, print_progress=True):
        """
        Grow tumour for number of steps or until tumour touches border.

        Args:
            print_progress (bool): prints \'current step/ total\' if true. 

        Returns:
            results (tuple): Final radius, living agents, roughness, growth velocity and timestep.
        """
        for i in range(self.steps):
            print(f'Running... step: {i+1}/{self.steps}         ', end='\r') if print_progress else None

            # stop simulation if tumor touches border
            if self.touches_border():
                print("\n Simulation stopped: Tumor touches border")
                self.running = False
                results = self.collect_results(i)
                return results
            
            self.step() 
            self.save_iteration_data()
        
        self.running = False
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
    
    def collect_results(self, step):
        """
        Collect final statistics of the simulation.
        """
        roughness = self.TVH.calculate_roughness(self.N_Ts[-1], self.Necs[-1])
        self.radii = self.TVH.radius_progression()
        radius = self.TVH.radius(self.N_T)
        velocity, _ = self.TVH.velocity_linear_fit() if self.steps > 200 else (None, None)
        n_agents = len(self.agents)
        return radius, n_agents, roughness, velocity, step

    def save_simulation_results_to_file(self):
        """
        Save simulation results to a pickle file.

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
            timestamp (string): timestamp that the files were originally saved at (see filename to find this value).
        """
        timestamp = str(timestamp)
        with open(f'save_files/simulation_data_{timestamp}.pickle', 'r') as f:
            model = pickle.load(f)
        return model