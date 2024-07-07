import json 
import os 
import numpy as np 
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def save_timestamp_metadata(model, timestamp):
    """
    Save parameter values of pickeled model object.

    Args:
        model (TumorGrowth): model that was pickled.
        timestamp (str): unique identifier of model, which was saved as 
                         simulation_data_<timestamp>.pickle.
    """
    update_metadata()
    metadata = {
        "run_id":timestamp,
        "iterations":len(model.N_Ts) - 1,
        "grid_height":model.height,
        "grid_width": model.width,
        "seed": model.seed,
        "a_pp":model.app, 
        "a_pi":model.api,
        "b_ii":model.bii,
        "b_ip":model.bip,
        "k":model.k, 
        "tau":model.tau, 
        "gamma":model.gamma,
        "D": model.D,
        "h": model.h, 
        "lambda": round(model.lam, 3),
        "phi_c": model.phi_c,
        "voronoi": model.distribution != 'uniform'
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
            
            # no match if parameter value does not match
            if entry.get(name) != params[name]:
                match = False
                break
        
        # add run_id to list if parameters matched
        filtered.append(entry['timestamp']) if match else None
    
    return filtered

def update_metadata():
    """
    Removes metadata from the metadata.json if runs were deleted.
    Deletion is checked via presence of simulation_parameters_run_id.pickle files.
    """
    # list all run id's in save_files
    run_ids = [f[f.rfind('_')+1:f.rfind('.')] for f in os.listdir('save_files/') if f.endswith('.pickle')]
  
    with open('save_files/metadata.json', 'r') as file:
        metadata = json.load(file)

    # compare to all runs ids in metadata.json, remove if not in saved runs
    metadata = [entry for entry in metadata if entry.get('run_id') in run_ids]

    # save updated metadata
    with open('save_files/metadata.json', 'w') as file:
        json.dump(metadata, file, indent=4)

    return metadata

def build_and_save_animation(data_frames, title, iterations):
    """
    Animates list of 2D-arrays and saves to mp4.

    Args:
        data_frames (list[ndarray]): list of frames to be animated.
        title (str): title of the output file
        iterations (int): Number of frames needed.
    """
    fig = plt.figure()
    im = plt.imshow(np.random.randint(low=0, high=data_frames[-1].max(), size=data_frames[0].shape), animated=True, interpolation="nearest", origin="upper", cmap='BuPu')
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

    Args:
        model (TumorGrowth): model of finished simulation
        steps_taken (int): number of steps completed, returned by run_model() method of model.
        payoff (ndarray[float]): payoff matrix used in the simulation.
        roughness (float): final roughness of tumor, returned by run_model() method.
        radius (float): final radius of tumor, returned by run_model() method.
        velocity (float): Average radial growth velocity, returned by run_model() method.
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