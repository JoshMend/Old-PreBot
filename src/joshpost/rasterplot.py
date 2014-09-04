import numpy as np
import scipy.io
import sys
import argparse
import matplotlib
import pylab
import matplotlib.markers 


def parse_args(argv):

    parser = argparse.ArgumentParser(prog="rasterplot",
                                     description = 'Plots a individual Neuron')

    #for the input we are assuming it is a valid file (BAD) 
    parser.add_argument('input', help = 'input(.mat) file')
    parser.add_argument('output', help = 'output (.jpeg) file')
    args = parser.parse_args(argv[1:])
    return args.input, args.output
             
                        
                    
def main(argv=None):
    if argv is None:
        argv = sys.argv
    (inFn,outFn) = parse_args(argv)
    mat_contents = scipy.io.loadmat(inFn)
    spikes = mat_contents['Y']
    spikes_coo = scipy.sparse.coo_matrix(spikes)
    x = []
    y = []

    for i,j in zip(spikes_coo.row, spikes_coo.col):
        x.append(j)
        y.append(i)
    

    
    matplotlib.pyplot.scatter(x,y, marker = '|')
    matplotlib.pyplot.axis([0,30000,0,80])
    matplotlib.pyplot.savefig(outFn)

if __name__ == '__main__':
    status = main()
    sys.exit(status)
        
