import SALib
from SALib.sample import sobol
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from classes.tumor_growth import TumorGrowth
import pickle
import time

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

problem = ProblemSpec({
    'num_vars': 10, # reduce to ~5 using 1st order
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[10**(-5), 10**(-3)], [0.01, 0.05], [5*10**(-5), 5*10**(-3)], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
})

distinct_samples = 128 #1024 # NOTE: small value for testing, used to be 16 -> debraj said do 128, maybe leave out reduce param space to five
grid_size = 150
steps = 1000
# NOTE: generate 1024 samples together, run in batches and on parallel computers to generate results

def run_model(param_values, **kwargs):
    """
    Wrapper function to run model when provided with array of params.
    """
    print(param_values.shape)
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
            height = grid_size, # choose 51 for testing
            width = grid_size   # choose 51 for testing
        )
        results[i] = model.run_model()
        print('\n', i)
    return results


# generate parameter samples
problem = problem.sample(sobol.sample, distinct_samples, calc_second_order=False)

# run model with the samples in parallel
problem.evaluate(run_model, steps=steps, grid_size=grid_size, nprocs=12) # NOTE: can increase nprocs even more maybe


Si_tumor = problem.analyze(SALib.analyze.sobol.analyze, calc_second_order=False, print_to_console=False)
time_stamp = str(time.time()).split('.')[0]
# pickle(f'Si_tumor_{time_stamp}.pkl', Si_tumor)
with open(f'Si_tumor_{time_stamp}.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(Si_tumor, file)
with open(f'problem_{time_stamp}.pickle', 'wb') as file:
    # Serialize and save the object to the file
    pickle.dump(problem, file)

axes = Si_tumor.plot()

total, first = Si_tumor.to_df() # returns list of dfs i think..
print(total, type(total)) 
total.to_csv('total.csv')
# from SALib.plotting.bar import plot as barplot
# def plot_result(result):
#     Si_df = result.to_df()

#     if isinstance(Si_df, (list, tuple)):
#         import matplotlib.pyplot as plt  # type: ignore

#         if ax is None:
#             fig, ax = plt.subplots(1, len(Si_df))

#         for idx, f in enumerate(Si_df):
#             barplot(f, ax=ax[idx])

#         axes = ax
#     else:
#         axes = barplot(Si_df, ax=ax)

#     return axes

# # for ax in axes:
#     plt.figure()
#     ax.plot()

plt.show()