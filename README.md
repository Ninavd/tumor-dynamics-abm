<img src="videos/tumor_growth-gif.gif" width="50%"/>

# Agent-Based Tumor Growth

Project on ABM tumor growth for the Agent-Based Modeling course at the UvA 2023-2024 (group 15).

## Installation

## Project structure

## Usage
```
usage: main.py [-h] [-s SEED] [-api ALPHA_PI] [-app ALPHA_PP] [-bii BETA_II] [-bip BETA_IP] [-dd DELTA_D] [--voronoi] [--summary] [--save]
               [--show_plot] [--animate]
               n_steps L_grid

Simulate Agent-Based tumor growth and save results

positional arguments:
  n_steps               max number of time steps used in simulation
  L_grid                Width of grid in number of cells

options:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  provide seed of simulation (default is random)
  -api ALPHA_PI, --alpha_pi ALPHA_PI
                        proliferative probability change when encountering an invasive cell
  -app ALPHA_PP, --alpha_pp ALPHA_PP
                        proliferative probability change when encountering an proliferative cell
  -bii BETA_II, --beta_ii BETA_II
                        invasive probability change when encountering an invasive cell
  -bip BETA_IP, --beta_ip BETA_IP
                        invasive probability change when encountering an proliferative cell
  --voronoi             Initialize ECM grid as voronoi diagram instead of uniform 
  --summary             print summary of simulation results
  --save                store simulation object in pickle file
  --show_plot           show plots of final tumor and other parameters
  --animate             save animation video of simulated tumor growth
```

## References