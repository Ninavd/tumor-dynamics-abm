from classes.tumor_visualization_helper import TumorVisualizationHelper as TVH
from classes.tumor_growth import TumorGrowth

from multiprocessing import Pool
import matplotlib.pyplot as plt

import numpy as np
import warnings 
warnings.simplefilter(action='ignore', category=FutureWarning)

def init_pool_processes():
    """
    Set different seed for every process.
    """
    np.random.seed()

def run_model(params):
    """
    Run model with provided parameters.
    """
    seed = np.random.randint(0, 1000)
    model = TumorGrowth(steps=params['steps'], width=params['size'], height=params['size'], seed=seed)
    radius, n_agents, roughness, velocity, steps = model.run_model() 
    # TODO: add collect statistics method to tumorgrowth and return model?
    return model.radii, TVH(model).calculate_roughness_progression(), 

if __name__ == '__main__':
    # define run parameters
    params = {'steps':500, 'size':101}
    runs = 4
    args = [params for i in range(runs)]
    n_processes = 4

    # create pool of processes and run in parallel 
    pool = Pool(processes=n_processes, initializer=init_pool_processes)
    async_outputs = pool.map_async(run_model, args)
    outputs = async_outputs.get()

    # outputs of all runs 
    outputs = np.array(outputs)
    average_progression = np.mean(outputs, axis=0)

    # plot average with std as CI
    plt.plot(range(len(average_progression)), average_progression)
    plt.fill_between(range(len(average_progression)), average_progression - np.std(outputs, axis=0), average_progression + np.std(outputs, axis=0), alpha=0.5)
    plt.show()