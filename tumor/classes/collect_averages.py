import copy
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings

from tumor.classes.tumor_growth import TumorGrowth
from tumor.classes.tumor_visualization_helper import TumorVisualizationHelper

warnings.simplefilter(action='ignore', category=FutureWarning)


class RunCollection:

    """
    Execute multiple runs of TumorGrowth with the same parameters but different seeds.
    """

    def __init__(self, N: int, log=True, **kwargs) -> None:
        """
        Initialize RunCollection object.

        Args:
            N (int): Number of runs.
            log (bool): if True, stdout and stderr are save in seperate logfiles; 
                        output_log.txt and error_log.txt
        """
        self.N = N
        kwargs.pop('seed', None)
        self.kwargs = kwargs

        # pipe output to files if log is True
        if log:
            sys.stdout = open('save_files/averaged_runs/output_log.txt', 'w') # redirect output to log file
            sys.stderr = open('save_files/averaged_runs/error_log.txt', 'w') 
    
    def _init_saving_arrays(self, steps) -> None:
        """
        Initialize array for saving results of each run.
        Each array has dimension N x iterations, where N is the number of runs.

        Args:
            steps (int): Number of iterations per run.
        """
        steps = steps + 1
        self.all_absolute_cell_counts = {
            "proliferating":np.zeros((self.N, steps)),
            "invasive"     :np.zeros((self.N, steps)),
            "necrotic"     :np.zeros((self.N, steps))
        }

        self.all_cell_fractions= {
            "proliferating":np.zeros((self.N, steps)),
            "invasive"     :np.zeros((self.N, steps)),
            "necrotic"     :np.zeros((self.N, steps))
        }

        self.arr_of_radius = np.zeros((self.N, steps))
        self.arr_of_roughness = np.zeros((self.N, steps))
        self.list_of_velocities = []

    def run(self) -> dict[str, np.ndarray]:
        """
        Run N simulations, return the averaged results and confidence intervals and save to csv.
        
        Results includes the averaged progression over time of the absolute and
        proportional cell counts, roughness, radius and growth velocity of the tumor.

        Returns:
            dict[str, ndarray]: parameter name and the averaged progression over time.
        """

        # create array for saving results
        steps = self.kwargs.get('steps', 1000)
        self._init_saving_arrays(steps)

        print('Runs started on:', datetime.datetime.now())

        # execute N runs of the model 
        for i in range(self.N):

            model = TumorGrowth(seed=np.random.randint(1000), **self.kwargs)
            results = model.run_model(print_progress=False)
            steps = results[-1]

            if steps != model.steps:
                print(f'Run {i+1} of {self.N} skipped due to early stopping at step {steps}')
                continue

            self.collect_results(model, i)
            print(f'Run {i+1} of {self.N} completed\n')

        print('Runs ended on:', datetime.datetime.now())

        # find mean and CI of all statistics and save
        results = self.collect_mean_results()
        self.save_to_csv(model, copy.copy(results))
        
        return model, results
         
    def collect_results(self, model: TumorGrowth, i: int):
        """
        Collect results of a single run.      

        Args:
            model (TumorGrowth): Tumor model of completed simulation.
            i (int): current run number.  
        """
        proliferating = np.array(model.proliferating_cells)
        invasive = np.array(model.invasive_cells)
        necrotic = np.array(model.necrotic_cells)
        total = proliferating + invasive + necrotic

        absolute_cell_counts = {
            "proliferating":proliferating,
            "invasive":invasive,
            "necrotic":necrotic
        }
        
        for cell_type in self.all_absolute_cell_counts:
            self.all_absolute_cell_counts[cell_type][i] = absolute_cell_counts[cell_type]
            self.all_cell_fractions[cell_type][i] = absolute_cell_counts[cell_type] / total

        TVH = TumorVisualizationHelper(model)
        self.arr_of_radius[i] = TVH.radius_progression()
        self.arr_of_roughness[i] = TVH.calculate_roughness_progression()

        velocity = TVH.calculate_velocities()
        self.list_of_velocities.append(velocity)

    def collect_mean_results(self) -> dict[str, np.ndarray]:
        """
        Average all the results, and find the 95% confidence interval and return.

        Returns:
            dict[str, ndarray]: averaged results and its confidence interval.
        """
        # absolute cell counts
        P_mean, P_CI = self.calculate_CI(self.all_absolute_cell_counts["proliferating"])
        I_mean, I_CI = self.calculate_CI(self.all_absolute_cell_counts["invasive"])
        N_mean, N_CI = self.calculate_CI(self.all_absolute_cell_counts["necrotic"])

        # fractions of cells
        P_frac_mean, P_frac_CI = self.calculate_CI(self.all_cell_fractions["proliferating"])
        I_frac_mean, I_frac_CI = self.calculate_CI(self.all_cell_fractions["invasive"])
        N_frac_mean, N_frac_CI = self.calculate_CI(self.all_cell_fractions["necrotic"])

        # radius, roughness and radial growth velocity
        radii, radii_CI = self.calculate_CI(self.arr_of_radius)
        roughness, roughness_CI = self.calculate_CI(self.arr_of_roughness)
        velocities, velocities_CI = self.calculate_CI(np.array(self.list_of_velocities))
        
        # concatenate results into a dict
        results = {
            'P_count'   :P_mean,      'P_count_conf':P_CI, 
            'I_count'   :I_mean,      'I_count_conf':I_CI, 
            'N_count'   :N_mean,      'N_count_conf':N_CI, 
            'P_fraction':P_frac_mean, 'P_fraction_conf':P_frac_CI, 
            'I_fraction':I_frac_mean, 'I_fraction_conf':I_frac_CI, 
            'N_fraction':N_frac_mean, 'N_fraction_conf':N_frac_CI,
            'radius'    :radii,       'radius_conf':radii_CI, 
            'roughness' :roughness,   'roughness_conf':roughness_CI,
            'velocity'  :velocities,  'velocity_conf':velocities_CI
        }
        return results
    
    def calculate_CI(self, arr_of_results: np.ndarray, z: float = 1.96):
        """
        Find mean and confidence interval of list of run results.

        Args:
            arr_of_results (ndarray): list of progressions, where each row represents one run. 
            z (float): z-value in confidence interval calculation. Default is 1.96.
        """
        stdev = np.std(arr_of_results, axis=0)
        confidence_interval = z * stdev / math.sqrt(arr_of_results.shape[0])
        return np.mean(arr_of_results, axis=0), confidence_interval

    def plot_with_CI(mean: np.ndarray, CI: np.ndarray, ylabel: str = None, **kwargs):
        """
        Plot the average progression with confidence interval.

        Args:
            mean (ndarray[float]): Averaged progression over time.
            CI (ndarray[float]): Confidence interval of averaged progression.
            ylabel (str): label for y-axis of plot. Default is None.
        """        
        plt.plot(mean, **kwargs)
        plt.fill_between(range(len(mean)), mean - CI, mean + CI, alpha=0.1)
        plt.grid()
        plt.ylabel(ylabel)
        plt.xlabel('iteration')
        return mean, CI
    
    def save_to_csv(self, model, results: dict[str, np.ndarray], save_dir: str='save_files/averaged_runs'):
        """
        Save the averaged results and confidence interval to csv.

        Args:
            model (TumorGrowth): A TumorGrowth model containing the run parameters.
            results (dict[str, ndarray]): The averaged results to save to csv.
            save_dir (str): Directory to save csv to. Default is \'save_files/averaged_runs\'
        """        
        # do velocity seperately as it is a different length 
        velocities = {
            'velocity'     :results.pop('velocity'),
            'velocity_conf':results.pop('velocity_conf')
        }
        velocity_df = pd.DataFrame(velocities)
        results_df = pd.DataFrame(results)

        title = f'{model.distribution}_{self.N}_runs_{model.steps}_iters_app_{model.app}_api_{model.api}_bii_{model.bii}_bip_{model.bip}_L_{model.width}'
        pd.concat([results_df, velocity_df], axis=1).to_csv(f'{save_dir}/{title}.csv') 

# TODO: implement running in parallel