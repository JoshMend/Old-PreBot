#!/user/bin/env python
import numpy as np
import scipy.io
import sys
import argparse
import matplotlib.pyplot as plt
import networkx as nx 


def parse_args(argv):
    parser = argparse.ArgumentParser(prog="connectionMatrixVisual",
                                     description = 'Visualizes the connectivity Matrix')

    #for the input we assume all files are valid(BAD)
    parser.add_argument('input', help = 'This is the graph file')
    parser.add_argument('output', help = 'Output Name (.png)')
    parser.add_argument('--inh', '-I', action = 'store_true',
                        help='gives you the connection of only inhibitory cells')
    parser.add_argument('--excit', '-E', action='store_true',
                        help ='gives you the connection of only excitatory cells')
    args = parser.parse_args(argv[1:])
    return args.input, args.output, args.inh, args.excit 

def main(argv=None):
    edges_x = []
    edges_y = []
    inh_x = []
    inh_y = []
    ex_x = []
    ex_y = []

    if argv is None:
        argv = sys.argv
    (inFn,outFn,inh,excit) = parse_args(argv)
    graph = nx.read_gml(inFn)
                   
    cells = nx.get_node_attributes(graph,'inh')                               
    edges = nx.edges(graph)
    
    for i in range(0,nx.number_of_edges(graph)-1):
        u = edges[i][0]
        v = edges[i][1]
        if inh:
            if cells[u] == 1:
                ex_x.append(u)
                ex_y.append(v)
        elif excit:
            if cells[u] == 0:
                ex_x.append(v)
                ex_y.append(u)
        else:
            if cells[u] == 1:
                inh_x.append(u)
                inh_y.append(v)
            else:    
                ex_x.append(u)
                ex_y.append(v)

         
    plt.scatter(ex_x,ex_y)
    plt.scatter(inh_x, inh_y, c = 'r')
    x = nx.number_of_nodes(graph)
    plt.axis([-.5,x,-.5,x])
    plt.savefig(outFn)

if __name__ == '__main__':
    status = main()
    sys.exit(status) 
