import SALib
from SALib.sample import sobol
from SALib import ProblemSpec
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from classes.tumor_growth import TumorGrowth

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

problem = ProblemSpec({
    'num_vars': 10,
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[10**(-5), 10**(-3)], [0.01, 0.05], [5*10**(-5), 5*10**(-3)], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
})

replicates = 10 # NOTE: not used rn..
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

# run model with the samples
problem.evaluate(run_model, nprocs=4) # NOTE: can increase nprocs even more maybe
print(problem.results)

# analyze the results
Si_sheep = problem.analyze(SALib.analyze.sobol.analyze, print_to_console=True)

def plot_index(s, params, i, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): dictionary {'S#': dict, 'S#_conf': dict} of dicts that hold
            the values for a set of parameters
        params (list): the parameters taken from s
        i (str): string that indicates what order the sensitivity is.
        title (str): title for the plot
    """

    if i == '2':
        p = len(params)
        params = list(combinations(params, 2))
        indices = s['S' + i].reshape((p ** 2))
        indices = indices[~np.isnan(indices)]
        errors = s['S' + i + '_conf'].reshape((p ** 2))
        errors = errors[~np.isnan(errors)]
    else:
        indices = s['S' + i]
        errors = s['S' + i + '_conf']
        plt.figure()

    l = len(indices)

    plt.title(title)
    plt.ylim([-0.2, len(indices) - 1 + 0.2])
    plt.yticks(range(l), params)
    plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
    plt.axvline(0, c='k')

# NOTE: this is not working yet sad, copied from notebook 4
for Si in (Si_sheep):
    pass
    # First order
    plot_index(Si, problem['names'], '1', 'First order sensitivity')
    plt.show()

    # Second order
    plot_index(Si, problem['names'], '2', 'Second order sensitivity')
    plt.show()

    # Total order
    plot_index(Si, problem['names'], 'T', 'Total order sensitivity')
    plt.show()

# SALib.analyze.sobol.analyze(
# problem, Y, calc_second_order=True, num_resamples=100, conf_level=0.95, print_to_console=False, parallel=False, n_processors=None, keep_resamples=False, seed=None
# )
# see : https://salib.readthedocs.io/en/latest/api/SALib.analyze.html#module-SALib.analyze.sobol 