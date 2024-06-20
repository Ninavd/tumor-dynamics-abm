import SALib
from SALib.sample import sobol
# from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from mesa.batchrunner import batch_run
from classes.tumor_growth import TumorGrowth

problem = {
    'num_vars': 10,
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[10**(-5), 10**(-3)], [0.01, 0.05], [5*10**(-5), 5*10**(-3)], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
}
replicates = 10
steps = 1000
distinct_samples = 16

# # Set the outputs
# model_reporters = {"Wolves": lambda m: m.schedule.get_breed_count(Wolf),
#              "Sheep": lambda m: m.schedule.get_breed_count(Sheep)}

param_values = sobol.sample(problem, distinct_samples)

# print("sample succeeded")

batch = batch_run(TumorGrowth, max_steps=1000, parameters={name:[] for name in problem['names']})

print(batch)

# count = 0 
# data = pd.DataFrame(index=range(replicates*len(param_values)), 
#                                 columns=problem['names'])
# data['Run'], data['Sheep'], data['Wolves'] = None, None, None

# for i in range(replicates):
#     for vals in param_values: 
#         # Change parameters that should be integers
#         vals = list(vals)
#         vals[2] = int(vals[2])
#         # Transform to dict with parameter names and their values
#         variable_parameters = {}
#         for name, val in zip(problem['names'], vals):x
#             variable_parameters[name] = val

#         batch.run_iteration(variable_parameters, tuple(vals), count)
#         iteration_data = batch.get_model_vars_dataframe().iloc[count]
#         iteration_data['Run'] = count # Don't know what causes this, but iteration number is not correctly filled
#         data.iloc[count, 0:3] = vals
#         data.iloc[count, 3:6] = iteration_data
#         count += 1

#         clear_output()
#         print(f'{count / (len(param_values) * (replicates)) * 100:.2f}% done')

# Si_sheep = sobol.analyze(problem, data['Sheep'].values, print_to_console=True)
# Si_wolves = sobol.analyze(problem, data['Wolves'].values, print_to_console=True)