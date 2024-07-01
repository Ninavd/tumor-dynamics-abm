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
    'num_vars': 10, # reduce to ~5 using 1st order
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[10**(-5), 10**(-3)], [0.01, 0.05], [5*10**(-5), 5*10**(-3)], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
})

distinct_samples = 128 #1024 # NOTE: small value for testing, used to be 16 -> debraj said do 128, maybe leave out reduce param space to five
grid_size = 101
steps = 1000
distribution = 'uniform'
result_dir = f'./save_files/SA_analysis_{distinct_samples}_distinct_samples_{distribution}'
# NOTE: generate 1024 samples together, run in batches and on parallel computers to generate results
if not os.path.exists(result_dir):
    os.mkdir(result_dir)

def run_model(param_values, **kwargs):
    """
    Wrapper function to run model when provided with array of params.
    """
    print(f"run {param_values.shape} on pid {os.getpid()}")

    params = ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii', 'diameter', 'roughness', 'living_agents', 'model_id']
    results_dict = {param:[] for param in params}

    results = np.zeros(param_values.shape[0])
    for i, params in enumerate(param_values):
        model = TumorGrowth(
            D = params[0],
            k = params[1],
            gamma = params[2],
            phi_c = params[3],
            theta_p = params[4],
            theta_i=params[5],
            app = params[6],
            api = params[7],
            bip = params[8],
            bii = params[9],
            steps = steps, # NOTE: choose 100 for testing
            delta_d = 200,
            height = grid_size, # choose 51 for testing
            width = grid_size   # choose 51 for testing
        )
        diameter, living_agents, roughness = model.run_model()

        timestamp = str(time.time()).split('.')[0]
        model_id = f'{timestamp}_{os.getpid()}_{str(i)}'
        # with open(f'{result_dir}/model_pickles/model_{model_id}.pickle', 'wb') as f:
        #     pickle.dump(model, f)

        results[i] = diameter
        
        print(f'\n run {i}/{param_values.shape[0]} on pid {os.getpid()}')
        for i, key in enumerate(results_dict):
            if key != 'diameter' and key != 'living_agents' and key != 'roughness' and key != 'model_id':
                results_dict[key].append(params[i])
            
        results_dict['diameter'].append(diameter)
        results_dict['living_agents'].append(living_agents)
        results_dict['roughness'].append(roughness)
        results_dict['model_id'].append(model_id)
    
    results_df = pd.DataFrame(results_dict)
    timestamp = str(time.time()).split('.')[0]
    results_df.to_csv(f'{result_dir}/cpu_{os.getpid()}_results_steps_{steps}_grid_{grid_size}_params_varied_{param_values.shape[1]}_id_{timestamp}.csv', index=False)
    return results


# generate parameter samples
problem = problem.sample(sobol.sample, distinct_samples, calc_second_order=False)

# run model with the samples in parallel
problem.evaluate(run_model, steps=steps, grid_size=grid_size, nprocs=12) # NOTE: can increase nprocs even more maybe

from glob import glob
# NOTE: TODO: REMOVE HARD CODED PARAM COUNT!!!!!!
pattern = f'*results_steps_{steps}_grid_{grid_size}_params_varied_10_id*'      # Replace with the pattern for matching file names
output_file = f'{result_dir}/concatenated_results_steps_{steps}_grid_{grid_size}_params_varied_10.csv'

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

axes = Si_tumor.plot()

# NOTE: TODO: REMOVE HARD CODED PARAM COUNT!!!!!!
total, first = Si_tumor.to_df() # returns list of dfs i think..
total.to_csv(f'{result_dir}/ST_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_10.csv')
first.to_csv(f'{result_dir}/S1_tumor_diameter_steps_{steps}_grid_{grid_size}_params_varied_10.csv')

plt.show()