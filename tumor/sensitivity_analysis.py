import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
import pickle
import time
import SALib
import warnings

from SALib.sample import sobol
from SALib import ProblemSpec
from glob import glob

from tumor.classes.tumor_growth import TumorGrowth

# surpress annoying warning about new feature in mesa
warnings.simplefilter(action='ignore', category=FutureWarning)


def run_model(param_values, **kwargs):
    """
    Wrapper function used to run model in parallel when provided with array of params.

    Args:
        param_values (ndarray): 2D array of one or many sets of parameter values to use in TumorGrowth model. 

    Returns:
        ndarray: Contains main result of the simulations, used for sensitivity analysis. Currently average radial growth velocity. 
    """
    print(f"run {param_values.shape} on pid {os.getpid()}")

    # initialize parameter names and results dictionary
    params = ['theta_p', 'theta_i', 'app', 'api', 'bip', 'bii', 'diameter', 'roughness', 'living_agents', 'velocity', 'steps_taken', 'model_id']
    results_dict = {param:[] for param in params}
    results = np.zeros(param_values.shape[0])

    # run model with different combinations of parameters
    for i, params in enumerate(param_values):
        model = TumorGrowth(
            theta_p = params[0],
            theta_i=params[1],
            app = params[2],
            api = params[3],
            bip = params[4],
            bii = params[5],
            delta_d = 200,            
            **kwargs
        )
        diameter, living_agents, roughness, velocity, steps_taken = model.run_model()

        # save most important result for sensitivity analysis
        results[i] = velocity
        
        # print progress
        print(f'\n run {i}/{param_values.shape[0]} on pid {os.getpid()}')

        # save parameters
        for i, key in enumerate(results_dict):
            if key != 'diameter' and key != 'living_agents' and key != 'roughness' and key != 'model_id' and key != 'velocity' and key != 'steps_taken':
                results_dict[key].append(params[i])
        
        # save results
        model_id = f'{timestamp}_{os.getpid()}_{str(i)}'
        results_dict['diameter'].append(diameter)
        results_dict['living_agents'].append(living_agents)
        results_dict['roughness'].append(roughness)
        results_dict['model_id'].append(model_id)
        results_dict['steps_taken'].append(steps_taken)
        results_dict['velocity'].append(velocity)
    
    # write results to csv 
    results_df = pd.DataFrame(results_dict)
    timestamp = str(time.time()).split('.')[0]
    results_df.to_csv(f'{result_dir}/cpu_{os.getpid()}_results_steps_{steps}_grid_{grid_size}_params_varied_{param_values.shape[1]}_id_{timestamp}.csv', index=False)
    
    return results

if __name__=="__main__":

    problem = ProblemSpec({
    'num_vars': 6, 
    'names': ['theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
    })

    n_vars_varied = problem['num_vars']
    distinct_samples = 1024 # NOTE: use small value for testing
    grid_size = 101
    steps = 1000
    distribution = 'voronoi'
    result_dir = f'../save_files/SA_analysis_{distinct_samples}_distinct_samples_{distribution}'

    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    # generate parameter samples
    problem = problem.sample(sobol.sample, distinct_samples, calc_second_order=False)

    # run model with the samples in parallel
    problem.evaluate(run_model, steps=steps, height=grid_size, width=grid_size, nprocs=12) 

    # find all CSV files in the specified directory matching the pattern
    pattern = f'*results_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}_id*'  
    csv_files = glob(os.path.join(result_dir, pattern))
    print(csv_files)

    # print matching files (for debugging)
    print(f"Found {len(csv_files)} files: {csv_files}")

    # read and concatenate all CSV files
    concatenated_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

    # write the result to a new CSV file
    output_file = f'{result_dir}/concatenated_results_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv'
    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved as: {output_file}")

    # perform sensitivity analysis on results
    Si_tumor = problem.analyze(SALib.analyze.sobol.analyze, calc_second_order=False, print_to_console=False)

    # pickle sensitivity analysis
    time_stamp = str(time.time()).split('.')[0]
    with open(f'{result_dir}/Si_tumor_{time_stamp}.pickle', 'wb') as file:
        pickle.dump(Si_tumor, file)
    with open(f'{result_dir}/problem_{time_stamp}.pickle', 'wb') as file:
        pickle.dump(problem, file)

    # save results of sensitivity analysis
    total, first = Si_tumor.to_df() # returns list of dfs 
    total.to_csv(f'{result_dir}/ST_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv')
    first.to_csv(f'{result_dir}/S1_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv')

    # plot first and total order sensitivity
    axes = Si_tumor.plot()
    plt.show()