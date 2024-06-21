import SALib
from SALib.sample import sobol
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from classes.tumor_growth import TumorGrowth

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

problem = ProblemSpec({
    'num_vars': 10,
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[10**(-5), 10**(-3)], [0.01, 0.05], [5*10**(-5), 5*10**(-3)], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
})

replicates = 10 # NOTE: not used rn.. idk what it should be used for tbh
steps = 1000
distinct_samples = 2 # NOTE: small value for testing, used to be 16

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
            steps = 100, # NOTE: small value chosen for testing
            height = 51, # small value chosen for testing
            width = 51   # small value chosen for testing
        )
        results[i] = model.run_model()
        print('\n', i)
    return results

# generate parameter samples
param_values = problem.sample(sobol.sample, distinct_samples)

# run model with the samples in parallel
problem.evaluate(run_model, nprocs=16) # NOTE: can increase nprocs even more maybe
Si_tumor = problem.analyze(SALib.analyze.sobol.analyze, print_to_console=False)
axes = Si_tumor.plot()
print(axes)
plt.show()
df = Si_tumor.analysis.to_df()