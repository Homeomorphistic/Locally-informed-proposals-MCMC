"""
From the same dir as setup.py run:
$> python3 setup.py develop
"""

from setuptools import find_packages, setup

setup(name='mcmc',
      version='0.0.1-dev',
      description='Monte Carlo Markov Chains',
      install_requires=['numpy', 'tsplib95', 'nptyping', 'scipy'],
      packages=find_packages(),
      zip_safe=False)
