import json 
import os 

def save_timestamp_metadata(timestamp, self):
    app, api = self.payoff[0][0], self.payoff[0][1]
    bii, bip = self.payoff[0][0], self.payoff[1][1] 

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

