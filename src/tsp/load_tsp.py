import tsplib95

problem = tsplib95.load('data/berlin52.tsp')
solution = tsplib95.load('data/berlin52.opt.tour')

nodes = list(problem.get_nodes())
edges = list(problem.get_edges())

print(problem._wfunc(2,1))
print(problem.trace_tours(solution.tours))
