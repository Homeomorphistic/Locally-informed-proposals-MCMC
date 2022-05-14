"""TODO description of the script"""

import argparse
from tsp_mcmc import TravelingSalesmenMCMC

def parse_arguments():
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
    return (args.data, bool(args.locally), float(args.tolerance),
            int(args.max_iter), int(args.stay_count))


data, locally, tolerance, max_iter, stay_count = parse_arguments()
print(type(max_iter))

tsp_solver = TravelingSalesmenMCMC(name=data, locally=locally)
optimum = tsp_solver.find_optimum(tolerance=tolerance,
                                  max_iter=max_iter,
                                  stay_count=stay_count)
print(optimum)

