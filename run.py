"""
A simple run script to run the simulation a single time.
"""
import numpy
from simulation import Simulation
import sys
import time


def run_simulation():
    start = time.time()

    simulation = Simulation(sys.argv[1], sys.argv[2], sys.argv[3])
    simulation.run(int(sys.argv[4]))

    end = time.time()
    print("TIME ELAPSED:", end - start)


if __name__ == '__main__':
    run_simulation()
