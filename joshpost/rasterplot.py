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

#This creates a list for the spike times and neuron that fired it
#ie (i,t) ith nueron and time, used if you need to actually sort through data  
def findspikes(voltages):
    spike_time  = []
    spike_index = []

    #What node you are at 
    index = 1

    #This shows whether you are on a current spike or not 
    spike_on = False 
    for v in voltages:

        for i in range(0,v.size-1):

            #-15 is the threshold for spiking
            if v[i] > -15:

                #gets rid of "spike" due to init cond
                if i == 0:
                    continue 
			
                #if you find a spike it sets to true so it can disregard the rest of the spike
                elif not spike_on:
                    spike_on = True
                    spike_time.append(i)
                    spike_index.append(index)
		#if the your counted spike already do nothing
		else:
		    continue
            if spike_on:
                if v[i] < -20:
                    spike_on = False
                    
        index = index +1
        spike_on = False
    return (spike_time,spike_index)                
                        
                    
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
        
