import matplotlib.pyplot as plt
import pickle

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

with open(f'save_files/SA_analysis_2_distinct_samples/Si_tumor_1719139099.pickle', 'rb') as file:
    # Serialize and save the object to the file
    Si_tumor = pickle.load(file)

axes = Si_tumor.plot()

plt.show()