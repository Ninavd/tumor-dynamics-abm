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

with open(f'save_files/SA_analysis_2_distinct_samples/Si_tumor_1719139099.pickle', 'rb') as file:
    # Serialize and save the object to the file
    Si_tumor = pickle.load(file)

axes = Si_tumor.plot()

plt.show()