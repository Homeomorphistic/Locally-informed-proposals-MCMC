"""TODO description of the script"""

import argparse
from tsp_mcmc import TravelingSalesmenMCMC

import numpy as np

def parse_arguments():
    """Parse arguments from terminal.

    Returns
    -------
    X: Tuple[str, bool, float, int, int]
        Tuple of arguments parsed from terminal.
    """
    parser = argparse.ArgumentParser(description="Locally informed MCMC")
    parser.add_argument('--data', default='berlin52',
                        help='Traveling salesmen problem name in data folder, '
                             '(omit .csv): str = berlin52')
    parser.add_argument('--seed', default=1,
                        help='Seed for PRNG: int = 1')
    parser.add_argument('--locally', default=False,
                        help='Use local distribution: bool = False')
    parser.add_argument('--temperature', default=lambda n: 2,
                        help='Temperature parameter, function of step: '
                             'Callable[[int], float] = lambda n: 2')
    parser.add_argument('--cooling', default=lambda n: 1,
                        help='Cooling parameter, function of step: '
                             'Callable[[int], float] = lambda n: 1')
    parser.add_argument('--max_iter', default=1000,
                        help='Maximum iteration count: int = 1000')
    parser.add_argument('--save', default=False,
                        help='Save flag: bool = False')
    args = parser.parse_args()
    args.locally = True if args.locally == 'True' else False
    args.save = True if args.save == 'True' else False

    if type(args.temperature) is str:
        args.temperature = eval(args.temperature)
    if type(args.cooling) is str:
        args.cooling = eval(args.cooling)

    return (args.data, int(args.seed), args.locally, args.temperature,
            args.cooling, int(args.max_iter), args.save)


# Get arguments for running TSP solver.
data, seed, locally, temperature, cooling, max_iter, save = parse_arguments()
# Set seed.
np.random.seed(seed)
# Run TSP solver and find optimum.
tsp_solver = TravelingSalesmenMCMC(name=data,
                                   locally=locally,
                                   temperature=temperature,
                                   cooling=cooling)
print(f'Running TSP solver for {data} with parameters: \nlocally={locally}, '
      f'temperature(1)={temperature(1):0.2f}, cooling(1)={cooling(1):0.2f},'
      f' iter={max_iter} \n')

print('Solution found: \n')
optimum = tsp_solver.find_optimum(max_iter=max_iter,
                                  stay_count=max_iter,
                                  save=save)
print(f'Distance: {optimum._weight}')


