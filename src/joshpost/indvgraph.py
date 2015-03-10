#!/user/bin/env python
import numpy as np
import scipy.io
import sys
import argparse
import matplotlib.pyplot as plt


def parse_args(argv):

    parser = argparse.ArgumentParser(prog="indvgraph",
                                     description = 'Plots a individual Neuron')

    #for the input we are assuming it is a valid file (BAD) 
    parser.add_argument('input', help = 'input(.mat) file')
    parser.add_argument('output', help = 'output (.jpeg) file')
    args = parser.parse_args(argv[1:])
    return args.input, args.output

def chop_transient(data, transient, dt):
    firstIdx = int(np.ceil(transient / dt) - 1)
    return data[:,firstIdx:]

def main(argv=None):
    if argv is None:
        argv = sys.argv
    (inFn, outFn) = parse_args(argv)
    mat_contents = scipy.io.loadmat(inFn)
    voltages = mat_contents['Y']
    choppedata = chop_transient(voltages,2000,1)
    time = np.arange(choppedata[1].size)
    print(len(choppedata[0]))
    print(len(choppedata[1]))
    print(len(choppedata[2]))
    print(len(choppedata[3]))
    print(len(time))
    index = 1
    for v in choppedata:
        plt.plot(time,v)
        #plt.axis([0,30000,-50,20])
        plt.savefig(outFn + str(index) + ".png")
        index = index + 1
        plt.clf()
        

if __name__ == '__main__':
    status = main()
    sys.exit(status)
