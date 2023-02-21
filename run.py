"""
A simple run script to run the simulation a single time
"""

from simulation import Simulation
import argparse


def run_simulation(operations, peers, network, repetition):
    simulation = Simulation(operations, peers, network)
    simulation.run(int(repetition))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--operations', default=1, help="Number of operations", metavar='')
    parser.add_argument('-p', '--peers', required=True, help="Number of peers", metavar='')
    parser.add_argument('-n', '--network', required=True, help="Network file", metavar='')
    parser.add_argument('-r', '--repetition', default=1, help="Number of repetitions", metavar='')

    args = parser.parse_args()
    parser.parse_args()

    run_simulation(args.operations, args.peers, args.network, args.repetition)




