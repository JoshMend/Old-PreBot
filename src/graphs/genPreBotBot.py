#!/usr/bin/env python

import respirnet
import numpy as np
import networkx as nx
import sys
import argparse 


def main(argv = None):
    if argv is None:
        argv = sys.argv
    parser = argparse.ArgumentParser(prog="genPreBotBot",
            description = 'Generates Graph based on Block Model')
    parser.add_argument('n', type = int, help='number of nodes')
    parser.add_argument('idegree', type = float, help='probabilty of inhibtory connection degree')
    parser.add_argument('output', help='output filename')
    parser.add_argument('-pI', type = float,
                        help = 'probability of inhibitory neuron',
                        default = .2)
    parser.add_argument('-gE', type=float,
                        help='conductance of excitatory (E) synapses, nS',
                        default = 2.5)
    parser.add_argument('-gI', type=float,
                        help='conductance of inhibitory (I) synapses, nS',
                        default = 2.5)

    args = parser.parse_args(argv[1:])
    n = args.n
    idegree = float(args.idegree)
    output = args.output
    gE = args.gE
    gI = args.gI
    pI = args.pI
    

    pMatE = np.array([ (3.0/(n-1), 0.05/(n-1)), 
                       (0.05/(n-1), 3.0/(n-1)) ])

    ###Idegree normally all the same, however changing to sweep block structure 	
    pMatI = np.array([ (idegree/(n-1), (3.0-idegree)/(n-1)), 
                       ((3.0-idegree)/(n-1), idegree/(n-1)) ])


    pTypes = [0, 0.25, 0.45, 0.3]

    g = respirnet.er_prebot_bot(n, pMatI, pMatE, pTypes, pI, gE, gI)
    nx.write_gml(g, output)

if __name__ == '__main__':
    status = main()
    sys.exit(status)
