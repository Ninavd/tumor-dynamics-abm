import SALib
from SALib.sample import sobol
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
from classes.tumor_growth import TumorGrowth
import pickle
import time
import os 

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

problem = ProblemSpec({
    'num_vars': 6, # reduce to ~5 using 1st order
    'names': ['theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
})

n_vars_varied = problem['num_vars']
distinct_samples = 1024 #1024 # NOTE: use small value for testing
grid_size = 101
steps = 1000
distribution = 'voronoi'
result_dir = f'./save_files/SA_analysis_{distinct_samples}_distinct_samples_{distribution}'

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def run_model(param_values, **kwargs):
    """
    Wrapper function to run model when provided with array of params.
    """
    print(f"run {param_values.shape} on pid {os.getpid()}")

    params = ['theta_p', 'theta_i', 'app', 'api', 'bip', 'bii', 'diameter', 'roughness', 'living_agents', 'velocity', 'steps_taken', 'model_id']
    results_dict = {param:[] for param in params}

    results = np.zeros(param_values.shape[0])
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

        timestamp = str(time.time()).split('.')[0]
        model_id = f'{timestamp}_{os.getpid()}_{str(i)}'

        # NOTE: do not uncomment unless you hate your laptop
        # with open(f'{result_dir}/model_pickles/model_{model_id}.pickle', 'wb') as f:
        #     pickle.dump(model, f)

        results[i] = velocity
        
        print(f'\n run {i}/{param_values.shape[0]} on pid {os.getpid()}')
        for i, key in enumerate(results_dict):
            if key != 'diameter' and key != 'living_agents' and key != 'roughness' and key != 'model_id' and key != 'velocity' and key != 'steps_taken':
                results_dict[key].append(params[i])
            
        results_dict['diameter'].append(diameter)
        results_dict['living_agents'].append(living_agents)
        results_dict['roughness'].append(roughness)
        results_dict['model_id'].append(model_id)
        results_dict['steps_taken'].append(steps_taken)
        results_dict['velocity'].append(velocity)
    
    results_df = pd.DataFrame(results_dict)
    timestamp = str(time.time()).split('.')[0]
    results_df.to_csv(f'{result_dir}/cpu_{os.getpid()}_results_steps_{steps}_grid_{grid_size}_params_varied_{param_values.shape[1]}_id_{timestamp}.csv', index=False)
    return results


# generate parameter samples
problem = problem.sample(sobol.sample, distinct_samples, calc_second_order=False)

# run model with the samples in parallel
problem.evaluate(run_model, steps=steps, height=grid_size, width=grid_size, nprocs=12) # NOTE: can increase nprocs even more maybe

from glob import glob

pattern = f'*results_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}_id*'      # Replace with the pattern for matching file names
output_file = f'{result_dir}/concatenated_results_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv'

# Find all CSV files in the specified directory matching the pattern
csv_files = glob(os.path.join(result_dir, pattern))
print(csv_files)

# Print matching files (for debugging)
print(f"Found {len(csv_files)} files: {csv_files}")

# Read and concatenate all CSV files
concatenated_df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

# Write the result to a new CSV file
concatenated_df.to_csv(output_file, index=False)
print(f"Concatenated CSV saved as: {output_file}")

Si_tumor = problem.analyze(SALib.analyze.sobol.analyze, calc_second_order=False, print_to_console=False)
time_stamp = str(time.time()).split('.')[0]

with open(f'{result_dir}/Si_tumor_{time_stamp}.pickle', 'wb') as file:
    pickle.dump(Si_tumor, file)
with open(f'{result_dir}/problem_{time_stamp}.pickle', 'wb') as file:
    pickle.dump(problem, file)


total, first = Si_tumor.to_df() # returns list of dfs 
total.to_csv(f'{result_dir}/ST_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv')
first.to_csv(f'{result_dir}/S1_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_{n_vars_varied}.csv')

axes = Si_tumor.plot()
plt.show()