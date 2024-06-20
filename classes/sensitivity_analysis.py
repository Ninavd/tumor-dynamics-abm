import SALib
from SALib.sample import saltelli
from mesa.batchrunner import FixedBatchRunner
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from mesa.batchrunner import BatchRunner
from tumor_growth import TumorGrowth

problem = {
    'num_vars': 10,
    'names': ['D', 'k', 'gamma', 'phi_c', 'theta_p', 'theta_i', 'app', 'api', 'bip', 'bii'],
    'bounds': [[1e-3,1e-5], [0.01, 0.05], [5e-3, 5e-5], [0.01, 0.05], [0.1, 0.5], [0.1, 0.5], [-0.95, 0], [-0.05, -0.01], [0.01, 0.05], [0, 0.95]]
}

replicates = 10
steps = 1000
distinct_samples = 10

param_values = saltelli.sample(problem, distinct_samples)

batch = BatchRunner(TumorGrowth(payoff, 201, 201, 913).run_simulation(), 
                    max_steps=max_steps,
                    variable_parameters={name:[] for name in problem['names']},
                    model_reporters=model_reporters)