import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

# dictionary to format ugly parameter names to latex
to_latex = {
    'app':'$\\alpha_{pp}$',
    'bii':'$\\beta_{ii}$',
    'api':'$\\alpha_{pi}$',
    'bip':'$\\beta_{ip}$',
    'theta_p':'$\\theta_p$',
    'theta_i':'$\\theta_i$',
    'gamma': '$\gamma$',
    'phi_c': '$\phi_c$',
    "D": 'D',
    'k': 'k'
}

# results of first order analysis
S1_result_files = [
    '../save_files/SA_analysis_128_distinct_samples_uniform/S1_tumor_diameter_steps_1000_grid_101_params_varied_10.csv',
    '../save_files/SA_analysis_128_distinct_samples_uniform_no_gamma/S1_tumor_diameter_steps_1000_grid_101_params_varied_9.csv',
    '../save_files/SA_analysis_128_distinct_samples_voronoi/S1_tumor_diameter_steps_1000_grid_101_params_varied_10.csv',
    '../save_files/SA_analysis_128_distinct_samples_voronoi_no_gamma/S1_tumor_diameter_steps_1000_grid_101_params_varied_9.csv',
    '../save_files/SA_analysis_1024_distinct_samples_uniform/S1_tumor_diameter_steps_1000_grid_101_params_varied_6.csv',
    '../save_files/SA_analysis_1024_distinct_samples_voronoi/S1_tumor_diameter_steps_1000_grid_101_params_varied_6.csv',
    ]

# results of second order analysis
ST_result_files = [
    '../save_files/SA_analysis_128_distinct_samples_uniform/ST_tumor_diameter_steps_1000_grid_101_params_varied_10.csv',
    '../save_files/SA_analysis_128_distinct_samples_uniform_no_gamma/ST_tumor_diameter_steps_1000_grid_101_params_varied_9.csv',
    '../save_files/SA_analysis_128_distinct_samples_voronoi/ST_tumor_diameter_steps_1000_grid_101_params_varied_10.csv',
    '../save_files/SA_analysis_128_distinct_samples_voronoi_no_gamma/ST_tumor_diameter_steps_1000_grid_101_params_varied_9.csv',
    '../save_files/SA_analysis_1024_distinct_samples_uniform/ST_tumor_diameter_steps_1000_grid_101_params_varied_6.csv',
    '../save_files/SA_analysis_1024_distinct_samples_voronoi/ST_tumor_diameter_steps_1000_grid_101_params_varied_6.csv'
]

def plot_layout():
    plt.xlim(-0.1)
    plt.tick_params(direction='out', size=6)
    plt.xticks(np.arange(0, 1.2, 0.4), fontsize=18)
    plt.grid(axis='y', alpha=0.5)
    plt.xlim(-0.1, 1.15)

if __name__=="__main__":
    save_dir = '../save_files'

    # create pretty plots for all results and save
    for S1, ST in zip(S1_result_files, ST_result_files):

        plt.figure(figsize=(8, 7))

        print(S1)
        print(ST)
        df = pd.read_csv(S1)

        plt.subplot(121)
        plt.errorbar(x=df['S1'], y=[to_latex[param] for param in df['Unnamed: 0']], xerr=df['S1_conf'], barsabove=True, fmt='o', markersize=6, capsize=8, color='black')
        plt.vlines(x = 0, color='black', linestyle='dotted', alpha=0.7, ymin=-1, ymax=0.5+len(df['S1']))
        plt.ylim(-0.5, len(df['S1']))
        plt.xlabel('$S_1$', fontsize=18)
        yticks, _ = plt.yticks(fontsize=18)
        plot_layout()

        df = pd.read_csv(ST)

        plt.subplot(122)
        plt.errorbar(
            x=df['ST'], 
            y=[to_latex[param] for param in df['Unnamed: 0']], 
            xerr=df['ST_conf'], 
            barsabove=True, 
            fmt='o', markersize=6, capsize=8, color='black'
        )
        plt.vlines(
            x = 0, 
            color='black', 
            linestyle='dotted', 
            alpha=0.7, 
            ymin=-1, 
            ymax=0.5+len(df['ST'])
        )
        
        plt.yticks(ticks=yticks, labels=[])
        plt.ylim(-0.5, len(df['ST']))
        plt.xlabel('$S_T$', fontsize=18)
        plot_layout()

        plt.tight_layout()

        title = S1.split('/')[1][12:] + 'ST_' + S1.split('/')[-1][:-4]
        plt.savefig(f'{save_dir}/{title}', dpi=300)

        plt.show()