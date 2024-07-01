import numpy as np
import matplotlib.pyplot as plt
from classes.tumor_visualization_helper import TumorVisualizationHelper as TVH
# import 

class TumorVisualization():
    def __init__(self, model) -> None:
        self.model = model
        self.TVH = TVH(model)

    def show_ecm(self, position = -1, show=True):
        """
        Plot current ECM density field.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.ecm_layers[position])
        return fig, axs

    def show_nutrients(self, position = -1, show=True):
        """
        Plot current nutrient concentration field.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.nutrient_layers[position])
        return fig, axs
    
    def show_tumor(self, position = -1, show=True):
        """
        Plot mask of the tumor. Includes necrotic cells.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.N_Ts[position])
        return fig, axs
    
    def plot_necrotic_cells(self, position = -1, show=True):
        """
        Plot mask of the tumor. Includes necrotic cells.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.Necs[position])
        plt.title(f'Necrotic cell distribution for a {self.model.height}x{self.model.width} Grid at iteration {len(self.model.ecm_layers)-1}')
        #fig.colorbar(axs, fraction=0.046, pad=0.04)  #TODO: colorbar keeps getting errors
        return fig, axs
    
    def plot_tumor_over_time(self, steps):
        """
        Plot tumor over time.
        """
        final_fig, final_axs = plt.subplots(1, 3, figsize=(15, 5))

        tumor_initial_fig, tumor_initial_axs = self.show_tumor(position = 0)
        tumor_middle_fig, tumor_middle_axs = self.show_tumor(position = round(steps/2))
        tumor_final_fig, tumor_final_axs = self.show_tumor(position = steps)

        tumor_initial_axs = final_axs[0].imshow(tumor_initial_axs.get_images()[0].get_array(), vmin=0, vmax=5, cmap = 'BuPu')
        final_axs[0].axis("off")
        final_axs[0].set_title('t=0')
        tumor_middle_axs = final_axs[1].imshow(tumor_middle_axs.get_images()[0].get_array(), vmin=0, vmax=5, cmap = 'BuPu')
        final_axs[1].set_title(f't={round(steps/2)}')
        final_axs[1].axis("off")
        tumor_final_axs = final_axs[2].imshow(tumor_final_axs.get_images()[0].get_array(), vmin=0, vmax=5, cmap = 'BuPu')
        final_axs[2].set_title(f't={steps}')
        final_axs[2].axis("off")

        plt.close(tumor_initial_fig)
        plt.close(tumor_middle_fig)
        plt.close(tumor_final_fig)
        # final_fig.colorbar(tumor_initial_axs, ax=final_axs[0], fraction=0.046, pad=0.04)
        # final_fig.colorbar(tumor_middle_axs, ax=final_axs[1], fraction=0.046, pad=0.04)
        # im_ratio = data.shape[0]/data.shape[1]
        im_ratio = 1

        final_fig.colorbar(tumor_final_axs, ax=final_axs.ravel().tolist(), fraction=0.046*im_ratio, pad=0.04)
        plt.suptitle(f'Tumor Growth Over Time For a {self.model.height}x{self.model.width} Grid')
        plt.show()

    def plot_all(self, position = -1):
        """
        Plot ECM, nutrient field and tumour in a single figure.
        """
        if type(position) == int:
            positions = [position]
        else:
            positions = position
        
        for position in positions:
            if (position > len(self.model.ecm_layers) - 1):
                print(f"Position {position} out of range. Max position: {len(self.model.ecm_layers) - 1}, skipping graphic iteration...")
                continue
            final_fig, final_axs = plt.subplots(1, 3, figsize=(15, 5))

            ecm_fig, ecm_axs = self.show_ecm(position = position)
            nutrient_fig, nutrient_axs = self.show_nutrients(position = position)
            tumor_fig, tumor_axs = self.show_tumor(position = position)

            ecm_axs = final_axs[0].imshow(ecm_axs.get_images()[0].get_array(), vmin = 0, vmax = 1, cmap = 'BuPu')
            final_axs[0].axis("off")
            final_axs[0].set_title('ECM Field Concentration')
            nutrient_axs = final_axs[1].imshow(nutrient_axs.get_images()[0].get_array(), cmap = 'BuPu')
            final_axs[1].set_title('Nutrient Field Concentration')
            final_axs[1].axis("off")
            tumor_axs = final_axs[2].imshow(tumor_axs.get_images()[0].get_array(), cmap = 'BuPu')
            final_axs[2].set_title('Tumor Cell Count')
            final_axs[2].axis("off")

            plt.close(ecm_fig)
            plt.close(nutrient_fig)
            plt.close(tumor_fig)
            final_fig.colorbar(ecm_axs, ax=final_axs[0], fraction=0.046, pad=0.04)
            final_fig.colorbar(nutrient_axs, ax=final_axs[1], fraction=0.046, pad=0.04)
            final_fig.colorbar(tumor_axs, ax=final_axs[2], fraction=0.046, pad=0.04)
            plt.suptitle(f'ECM, Nutrient, and Tumor Values at Iteration {position%self.model.steps + 1} of {self.model.steps} for a {self.model.height}x{self.model.width} Grid')
            plt.show()

    def plot_distribution(self):
        # total_cells = []
        plt.figure(figsize=(12, 7))
        dead_distributions = []
        living_distributions = []
        time_points = [round(self.model.steps/4), round(self.model.steps/2), round(3*self.model.steps/4), self.model.steps-1]
        for i in time_points:
            living_distr, dead_distr = self.model.cell_distribution(iteration = i)
            dead_distributions.append(dead_distr)
            living_distributions.append(living_distr)

        names = ['dead', 'living']
        distributions = [dead_distributions, living_distributions]
        for i in range(2):

            plt.subplot(121 + i)
            plt.title(names[i])

            for i, distr in enumerate(distributions[i]):
                if len(distr) == 0:
                    continue
                d = np.diff(np.unique(distr)).min()
                left_of_first_bin = distr.min() - float(d)/2
                right_of_last_bin = distr.max() + float(d)/2
                plt.hist(np.array(distr).flatten(), np.arange(left_of_first_bin, right_of_last_bin + d), d, label=f't={time_points[i]}', histtype='step')
            plt.legend()

        plt.show()

    def plot_birth_deaths(self):
        birth_rel_death = [(self.model.births[i]) /(self.model.births[i] + self.model.deaths[i]) for i in range(len(self.model.births))]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.model.births, label='Births')
        ax1.plot(self.model.deaths, label='Deaths')
        ax2.plot(birth_rel_death, label='Births / (Births + Deaths)', color='g')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='green', which='both')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.title('Cumulative Number of Births and Deaths')
        plt.show()
    
    def plot_max_nutrient(self):
        min_nutrient = [np.min(nutrient) for nutrient in self.model.nutrient_layers]
        sum_nutrient = [np.sum(nutrient) for nutrient in self.model.nutrient_layers]
        relative_count = [min_nutrient[i]/sum_nutrient[i] for i in range(len(min_nutrient))]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(min_nutrient, label='Min Nutrient Value')
        ax1.plot(sum_nutrient, label='Sum of Nutrient Values')
        ax2.plot(relative_count, 'g', label='Relative Percentage of Max Cells in Grid')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Nutrient Value')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='green', which='both')
        plt.title('Nutrient Value in Grid')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.show()
    
    def plot_max_count(self):
        max_count = [np.max(N_T) for N_T in self.model.N_Ts]
        sum_count = [np.sum(N_T) for N_T in self.model.N_Ts]
        relative_count = [max_count[i]/sum_count[i] for i in range(len(max_count))]
        fig, ax1 = plt.subplots()

        ax2 = ax1.twinx()
        ax1.plot(max_count, label='Max Cells in a Subsection of Grid')
        ax1.plot(sum_count, label='Total Number of Cells in Grid')
        ax2.plot(relative_count, 'g', label='Relative Percentage of Max Cells to Total Number of Cells in Grid')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cell Count')
        ax2.set_ylabel('Relative Percentage', color='g')
        ax2.tick_params(colors='g', which='both')
        ax2.set_ylabel('Relative Percentage', color='g')
        
        plt.title('Cell Count in Grid')
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.show()
    
    def plot_radial_distance(self):
        """
        Plot the radial distance of the tumor from the center of the grid.
        """
        # tvh = TVH(self.model)
        radial_distance = self.TVH.calculate_radial_distance()
        plt.plot(radial_distance)
        velocity = self.TVH.linear_fit()
        offset = 0
        linear_func = lambda a, b, x : a*x + b
        plt.plot(range(len(self.model.N_Ts)), [linear_func(velocity, offset, x) for x in range(len(self.model.N_Ts))], color='orange')
        plt.title('Average Radial Distance From Tumor Center to Tumor Edge')
        plt.xlabel('Iteration')
        plt.ylabel('Average Radial Distance')
        plt.grid()
        plt.show()

    def plot_roughness(self):
        """
        Plot the roughness of the tumor.
        """
        roughness_values = self.TVH.calculate_roughness()
        plt.plot(roughness_values)
        plt.title('Roughness of the Tumor Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Roughness of the Tumor')
        plt.grid()
        plt.show()

    def plot_cell_types(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.model.proliferating_cells, label = 'Proliferative Cells')
        ax1.plot(self.model.invasive_cells, label = 'Invasive Cells')
        ax1.plot(self.model.necrotic_cells, label = 'Necrotic Cells')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Number of Cells')
        ax1.set_title('Cell Types Over Time')
        plt.grid()
        plt.legend()
        plt.show()

    def plot_proportion_cell_types(self):
        fig, ax1 = plt.subplots()
        sum_count = np.array([np.sum(N_T) for N_T in self.model.N_Ts])
        sum_count += np.array([np.sum(Necs) for Necs in self.model.Necs])

        ax1.plot(np.array(self.model.proliferating_cells)/sum_count, label = 'Proliferative Cells')
        ax1.plot(np.array(self.model.invasive_cells)/sum_count, label = 'Invasive Cells')
        ax1.plot(np.array(self.model.necrotic_cells)/sum_count, label = 'Necrotic Cells')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fraction of Cells')
        ax1.set_title('Fraction of Cell Types Over Time')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_velocities(self):
        """
        Plot the velocities of the tumor.
        """
        velocities = self.TVH.calculate_velocities()
        plt.plot(velocities)
        moving_average = 25
        ret = np.cumsum(velocities, dtype=float)
        ret[moving_average:] = ret[moving_average:] - ret[:-moving_average]
        ma = ret[moving_average - 1:] / moving_average
        plt.plot(ma)
        plt.title('Velocity of the Tumor Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Velocity of the Tumor')
        plt.grid()
        plt.show()