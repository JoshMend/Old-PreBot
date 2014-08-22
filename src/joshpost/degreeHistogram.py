import networkx as nx
import sys
import argparse
import matplotlib.pyplot as plt

def parse_args(argv):
        parser = argparse.ArgumentParser(prog="degreeHistogram",
                                     description = 'Visualize the degree histogram of a graph')
        parser.add_argument('input', help = 'This is the graph file')
        parser.add_argument('output', help = 'Output Name (.png)')
        args = parser.parse_args(argv[1:])
        return args.input, args.output

def main(argv=None):
    x = []
    y = []

    if argv is None:
        argv = sys.argv

    (inFn,OutFn) = parse_args(argv)
    graph = nx.read_gml(inFn)
    h = nx.degree_histogram(graph)
    numberOfNodes = nx.number_of_nodes(graph)
    index = 0
    for i in h:
        x.append(index)
        y.append( i/float(numberOfNodes))
        index = index + 1

    print x
    print y
    print h
    plt.bar(x,y,align='center')
    plt.axis([-.5,len(h)-.5,0,1])
    plt.savefig(OutFn)
    
if __name__ == '__main__':
    status = main()
    sys.exit(status)
    

        
