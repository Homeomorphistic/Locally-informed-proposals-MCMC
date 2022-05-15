"""TODO description of the script"""

import argparse
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
                        help='Traveling salesmen problem name in data folder, '
                             '(omit .csv): str = berlin52')
    parser.add_argument('--locally', default=False,
                        help='Use local distribution: bool = False')
    parser.add_argument('--tolerance', default=0.01,
                        help='How big the difference between steps can get: '
                             'float = 0.01')
    parser.add_argument('--max_iter', default=1000,
                        help='Maximum iteration count: int = 1000')
    parser.add_argument('--stay_count', default=100,
                        help='Maximum number of stays in the same state: int = 10')
    args = parser.parse_args()
    args.locally = True if args.locally == 'True' else False

    return (args.data, args.locally, float(args.tolerance),
            int(args.max_iter), int(args.stay_count))


# Get arguments for running TSP solver.
data, locally, tolerance, max_iter, stay_count = parse_arguments()
# Run TSP solver and find optimum.
tsp_solver = TravelingSalesmenMCMC(name=data, locally=locally)
print(f'Running TSP solver for {data} with parameters: \nlocally={locally},'
      f'tol={tolerance}, iter={max_iter}, stay_limit={stay_count} \n')

print('Solution found: \n')
optimum = tsp_solver.find_optimum(tolerance=tolerance,
                                  max_iter=max_iter,
                                  stay_count=stay_count)
print(optimum, '\n')

