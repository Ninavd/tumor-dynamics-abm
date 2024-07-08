import numpy as np
import matplotlib.pyplot as plt

from tumor.classes.tumor_visualization_helper import TumorVisualizationHelper as TVH


class TumorVisualization:
    """
    Visualize final results of tumor growth simulation.

    Example usage:
        visualize = TumorVisualization(model)
        visualize.plot_all() # to plot final ECM and nutrient distribution and tumor shape.
    
    Attributes:
        model (TumorGrowth): Model to visualize.
        TVH (TumorVisualizationHelper): Helper class that provides analysis of results.
    """

    def __init__(self, model):
        """
        Initializes visualisation.

        Args:
            model (TumorGrowth): Model to visualize.
        """
        self.model = model
        self.TVH = TVH(model)

    def show_ecm(self, position = -1):
        """
        Plot current ECM density field.

        Args:
            position (int): which snapshot to plot. Default plots final snapshot.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.ecm_layers[position])
        return fig, axs

    def show_nutrients(self, position = -1):
        """
        Plot current nutrient concentration field.

        Args:
            position (int): which snapshot to plot. Default plots final snapshot.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.nutrient_layers[position])
        return fig, axs
    
    def show_tumor(self, position = -1):
        """
        Plot mask of the tumor. Excludes necrotic cells.

        Args:
            position (int): which snapshot to plot. Default plots final snapshot.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.N_Ts[position])
        return fig, axs
    
    def plot_necrotic_cells(self, position = -1):
        """
        Plot mask of the tumor, only including necrotic cells.

        Args:
            position (int): which snapshot to plot. Default plots final snapshot.
        """
        fig, axs = plt.subplots()
        im = axs.imshow(self.model.Necs[position], cmap='BuPu')
        plt.title(f'Necrotic cell distribution for a {self.model.height}x{self.model.width} Grid at iteration {len(self.model.ecm_layers)-1}')
        plt.colorbar(im, fraction=0.046, pad=0.04)  #TODO: colorbar keeps getting errors
        return fig, axs
    
    def plot_tumor_over_time(self, steps):
        """
        Plot tumor over time.

        Args:
            steps (int): Number of iterations performed in simulation.
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
        im_ratio = 1

        final_fig.colorbar(tumor_final_axs, ax=final_axs.ravel().tolist(), fraction=0.046*im_ratio, pad=0.04)
        plt.suptitle(f'Tumor Growth Over Time For a {self.model.height}x{self.model.width} Grid')
        plt.show()

    def plot_all(self, position = -1):
        """
        Plot ECM, nutrient field and tumour shape in a single figure.

        Args:
            position (int): which snapshot to plot. Default plots final snapshot.
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

    def plot_birth_deaths(self):
        """
        Plot cumulative number of births and deaths.
        """
        birth_rel_death = [(self.model.births[i]) /(self.model.births[i] + self.model.necrotic_cells[i]) for i in range(len(self.model.births))]
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax1.plot(self.model.births, label='Births')
        ax1.plot(self.model.necrotic_cells, label='Deaths')
        ax2.plot(birth_rel_death, label='Births / (Births + Deaths)', color='g')

        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Count')
        ax2.set_ylabel('Relative fracion', color='g')
        ax2.tick_params(colors='green', which='both')
        ax2.set_ylim(0, 1.05)
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(handles1 + handles2, labels1 + labels2)
        plt.title('Cumulative Number of Births and Deaths')
        plt.show()
    
    def plot_radial_distance(self):
        """
        Plot the average radius of the tumor over time.
        """
        plt.plot(self.model.radii)
        plt.title('Average Radial Distance From Tumor Center to Tumor Edge')
        plt.xlabel('Iteration')
        plt.ylabel('Average Radial Distance')
        plt.grid()
        plt.show()

    def plot_roughness(self):
        """
        Plot the roughness of the tumor over time.
        """
        roughness_values = self.TVH.calculate_roughness_progression()
        plt.plot(roughness_values)
        plt.title('Roughness of the Tumor Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Roughness of the Tumor')
        plt.grid()
        plt.show()

    def plot_cell_types(self):
        """
        Plot progression of absolute counts of each cell type.
        """
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
        """
        Plot progression of fractional cell distribution over time.
        """
        fig, ax1 = plt.subplots()
        sum_count = np.sum([self.model.proliferating_cells, self.model.invasive_cells, self.model.necrotic_cells], axis=0)

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
        Plot the progression of the growth velocity of the tumor.
        """
        velocities = self.TVH.calculate_velocities()
        plt.plot(velocities)
        plt.title('Velocity of the Tumor Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Velocity of the Tumor')
        plt.grid()
        plt.show()