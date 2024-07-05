# Agent-Based Tumor Growth
<img src="videos/tumor_growth-gif.gif" width="50%"/>

Project on ABM tumor growth for the Agent-Based Modeling course at the UvA 2023-2024 (group 15).

## Installation
To get started, clone the repository and install the required packages: 
```bash
git clone https://github.com/Ninavd/tumor-dynamics-abm.git
cd tumor-dynamics-abm
pip install -r requirements.txt
```

If you want to be able to save animations of your simulation, you might need to install [FFmpeg](https://www.ffmpeg.org/download.html), which can not be installed using `pip`. It is easiest to install on Linux/WSL or Homebrew via

```bash
brew install ffmpeg
```
or 
```bash
sudo apt install ffmpeg
```

## Running an interactive Simulation
By executing `mesa runserver` or `python server.py` in the root directory, an interface will be opened in your browser, allowing for interactive simulations on a fixed 50x50 grid.  
![screenshot of interactive simulation](save_files/image.png)

## Running via the CLI
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

## Project structure


## References
