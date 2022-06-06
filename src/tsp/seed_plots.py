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
    parser.add_argument('--sprint', default='7-seed',
                        help='Sprint with results: str = 6')
    parser.add_argument('--data', default='berlin52',
                        help='Traveling salesmen problem name in results '
                             'folder: str = berlin52')
    parser.add_argument('--locally', default=False,
                        help='Use local distribution: bool = False')
    parser.add_argument('--temperature', default=2.00,
                        help='Temperature of first step: float =  2.00')
    parser.add_argument('--cooling', default=1.00,
                        help='Cooling of first step: float =  1.00')
    parser.add_argument('--save', default=False,
                        help='Save flag: bool = False')
    args = parser.parse_args()
    args.save = True if args.save == 'True' else False

    return (args.data, args.locally, args.sprint, float(args.temperature),
            float(args.cooling), args.save)


# Get arguments for running TSP solver.
data, locally, sprint, temperature, cooling, save = parse_arguments()
# Prepare paths for reading.

path_local = f'results/sprint-{sprint}/{data}/locally=True/temp={temperature:0.2f}/' \
             f'cool={cooling:0.2f}'

for seed in {1, 2, 3, 4}:
    path_nonlocal = f'results/sprint-{sprint}/seed={seed}/{data}/' \
                    f'locally={locally}/temp={temperature:0.2f}/cool={cooling:0.2f}'
    # Read nonlocal results.
    result_dict, num_steps, time, distances = {}, [], [], []
    result_files = os.listdir(path_nonlocal)
    for file in result_files:
        file = os.path.join(path_nonlocal, file)
        result_dict = load(open(file, 'r'))
        num_steps.append(int(result_dict['num_steps']))
        time.append(float(result_dict['time']))
        distances.append(float(result_dict['distance']))

    # Plot results for nonlocal.
    order = np.argsort(num_steps)
    num_steps = np.array(num_steps)[order]
    time = np.array(time)[order]
    distances = np.array(distances)[order]
    plt.plot(num_steps, distances, marker='o')


# Finishing touches for chart.
plt.title(f'{data}, locally={locally}')
plt.xlabel('Number of steps')
plt.ylabel('Distance')
plt.legend([1, 2, 3, 4], loc='center right', title='Seeds')

# Saving figure or printing depending on --save param.
if save:
    filename = f'results/sprint-{sprint}/charts/{data}/' \
               f'{data}_locally={locally}.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=1200)
else:
    plt.show()