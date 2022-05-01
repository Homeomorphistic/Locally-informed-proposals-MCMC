"""
mcmc
=====

A package for Markov Chain Monte Carlo methods.

mcmc provides simple classes for stochastic processes and markov chains (similar to 'stochastic' package) and
Metropolis-Hastings sampling. The main focus is to quickly generate markov chains on different objects with
defined probability distributions to find minima/maxima of cost functions.

Modules
-------
stochastic_process
markov_chain
metropolis_hastings
"""

#from stochastic_process import StochasticProcess
#from markov_chain import MarkovChain, HomogeneousMarkovChain

__all__ = ['stochastic_process', 'markov_chain', 'metropolis_hastings']


