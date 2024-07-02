import json 
import os 
import numpy as np 
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def save_timestamp_metadata(self, timestamp):
    app, api = self.app, self.api
    bii, bip = self.bii, self.bip

    metadata = {
        "run_id":timestamp,
        "iterations":len(self.N_Ts) - 1,
        "grid_height":self.height,
        "grid_width": self.width,
        "seed": self.seed,
        "a_pp":app, 
        "a_pi":api,
        "b_ii":bii,
        "b_ip":bip,
        "k":self.k, 
        "tau":self.tau, 
        "gamma":self.gamma,
        "D": self.D,
        "h": self.h, 
        "lambda": round(self.lam, 3),
        "phi_c": self.phi_c,
        "Voroni": self.distribution != 'uniform'
    }

    # save timestamp metadata
    with open('save_files/metadata.json', 'r') as f:
        data = json.load(f)
        data.append(metadata)

        with open('save_files/metadata.json', 'w') as f:
            json.dump(data, f, indent=4)

def filter_runs(params: dict):
    """
    Filters existing simulation data, returns timestamp[s] with correct param values.

    Args:
        params (dict): dictionary containing parameter names (keys) and their desired value
    """
    metadata = update_metadata()

    filtered = []
    for entry in metadata:

        match = True
        for name in params:
            if entry.get(name) != params[name]:
                match = False
                break

        filtered.append(entry['timestamp']) if match else None
    
    return filtered

def update_metadata():
    """
    Removes metadata from the metadata.json if runs were deleted.
    Deletion is checked via simulation_parameters_run_id.txt
    """
    # list all run id's in save_files
    run_ids = [f[f.rfind('_')+1:f.rfind('.')] for f in os.listdir('save_files/') if f.endswith('.txt')]
  
    with open('save_files/metadata.json', 'r') as file:
        metadata = json.load(file)

    # compare to all runs ids in metadata.json, remove if not in saved runs
    metadata = [entry for entry in metadata if entry.get('run_id') in run_ids]

    # save updated metadata
    with open('save_files/metadata.json', 'w') as file:
        json.dump(metadata, file, indent=4)

    return metadata

def select_non_zero(data):
    return data != 0

def build_and_save_animation(data_frames, title, iterations):
    """
    Animates list of 2D-arrays and saves to mp4.

    Args:
        data_frames (list[ndarray]): list of frames to be animated.
        title (str): title of the output file
        iterations: Number of frames needed.
    """
    fig = plt.figure()
    im = plt.imshow(np.random.randint(low=0, high=data_frames[-1].max(), size=(5, 5)), animated=True, interpolation="nearest", origin="upper", cmap='BuPu')
    plt.colorbar()

    def animate(frame_number):
        im.set_data(data_frames[frame_number])
        return im,

    anim = animation.FuncAnimation(fig, animate, frames=iterations, interval=200, blit=True) 

    # saving to m4 using ffmpeg writer 
    writervideo = animation.FFMpegWriter(fps=15) 
    anim.save(f'videos/{title}.mp4', writer=writervideo) 
    plt.close()

def print_summary_message(model, steps_taken, payoff, roughness, radius, velocity):
    """
    Prints summary message of completed simulation.
    """

    print(
            f"""
            +--------------------------+---------------------------------+
            |                       SUMMARY                              |
            +--------------------------+---------------------------------+
            | Iterations               | {steps_taken:<31} |
            | Grid size                | {str(model.width)+'x'+str(model.height):<31} |
            | Seed                     | {model.seed:<31} |
            | Payoff matrix            | {repr(payoff):<31} |
            | ECM                      | {model.distribution:<31} |
            | Final #(proliferating)   | {model.proliferating_cells[-1]:<31} |
            | Final #(invasive)        | {model.invasive_cells[-1]:<31} |
            | Final #(necrotic)        | {model.necrotic_cells[-1]:<31} |
            | Final roughness          | {roughness:<31.3f} |
            | Final tumor size         | {radius:<31.3f} |
            | Average growth velocity  | {velocity:<31.3f} |
            +--------------------------+---------------------------------+
            """
        )