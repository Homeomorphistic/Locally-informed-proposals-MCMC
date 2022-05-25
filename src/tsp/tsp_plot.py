"""TODO description of the script"""

import argparse
import os
from json import load
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsp_mcmc import TravelingSalesmenMCMC


def parse_arguments():
    """Parse arguments from terminal.

    Returns
    -------
    X: Tuple[str, bool, float, int, int]
        Tuple of arguments parsed from terminal.
    """
    parser = argparse.ArgumentParser(description="Locally informed MCMC")
    parser.add_argument('--data', default='berlin52',
                        help='Traveling salesmen problem name in results '
                             'folder: str = berlin52')
    parser.add_argument('--scaling', default=0.01,
                        help='How to scale the difference between weights: '
                             'float = 0.01')
    parser.add_argument('--save', default=False,
                        help='Save flag: bool = False')
    args = parser.parse_args()
    args.save = True if args.save == 'True' else False

    return (args.data, float(args.scaling), args.save)


# Get arguments for running TSP solver.
data, scaling, save = parse_arguments()
# Prepare paths for reading.
path_nonlocal = f'results/sprint-3/{data}/scaling={scaling}/locally=False'
path_local = f'results/sprint-3/{data}/scaling={scaling}/locally=True'

# Read nonlocal results.
result_dict, num_steps, time, distances = {}, [], [], []
result_files = os.listdir(path_nonlocal)
for file in result_files:
    file = os.path.join(path_nonlocal, file)
    result_dict = load(open(file, 'r'))
    num_steps.append(int(result_dict['num_steps']))
    time.append(float(result_dict['time']))
    distances.append(float(result_dict['distance']))
else:
    scale = result_dict['scale']

# Plot results for nonlocal.
order = np.argsort(num_steps)
num_steps = np.array(num_steps)[order]
time = np.array(time)[order]
distances = np.array(distances)[order]
plt.plot(num_steps, distances)

# Add columns to table.
results_table = pd.DataFrame({'Num_steps': num_steps, 'Time_non': time,
                              'Distance_non': distances})

# Read local results.
result_dict, time, distances = {}, [], []
result_files = os.listdir(path_local)
for file in result_files:
    file = os.path.join(path_local, file)
    result_dict = load(open(file, 'r'))
    time.append(result_dict['time'])
    distances.append(result_dict['distance'])

# Plot results for local.
time = np.array(time)[order]
distances = np.array(distances)[order]
plt.plot(num_steps, distances)
# Add columns to table.
results_table['Time_loc'] = time
results_table['Distance_loc'] = distances

# Reading the true optimum.
opt_dict = load(open('data/tsp_optimal.json', 'r'))
# Plot optimum.
plt.axhline(y=opt_dict[data], linestyle='dashed', color="red")

# Finishing touches for chart.
plt.title(f'{data}, scale={scale}')
plt.xlabel('Number of steps')
plt.ylabel('Distance')
plt.legend(['Nonlocal', 'Local', 'optimum'], loc='center right')

# Saving figure or printing depending on --save param.
if save:
    filename = f'results/sprint-3/charts/{data}/{data}_scaling={scaling}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    results_table.to_csv(filename[:-3]+"csv")
else:
    print(results_table)
    plt.show()
