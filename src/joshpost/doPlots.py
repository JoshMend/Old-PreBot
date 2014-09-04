#!/usr/bin/env python

import scipy.io
import argparse
import sys
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import os.path

def parse_args(argv):


    #parsing
    parser = argparse.ArgumentParser(prog ="doPlots",
                                     description = 'plotting of processed data')
    parser.add_argument('input', help = 'file contaning processed data (.mat)')
    parser.add_argument('output', help='output filename .jpeg(graph type will be appended)')
    
    args = parser.parse_args(argv[1:])
    return args.input, args.output


##Arranges the spike data for the raster plot
def arrange_raster(spike_data):
    spike_times = []
    nueron_spike = []
    for j in range(0,len(spike_data)):
        for i in range(0,len(spike_data[j])):
            if spike_data[j][i] == 1:
                spike_times.append(i)
                nueron_spike.append(j)
    return spike_times,nueron_spike

##Arranges the spike data for the connection matrix 
def arrange_connection_matrix(connection_data,inhib_data):
    inh_x = []
    inh_y = []
    ex_x = []
    ex_y = []

    edges = connection_data
    inhib = inhib_data

    for i in range(0,len(edges)-1):
        u = edges[i][0]
        v = edges[i][1]
        
        if inhib[u][0] == 1:
            inh_x.append(u)
            inh_y.append(v)
        else:
            ex_x.append(u)
            ex_y.append(v)
    return inh_x,inh_y,ex_x,ex_y

def arrange_histogram(histo_data, number_of_nodes):
    connection = []
    percent_of = []
    index = 0
    for h in histo_data:
        connection.append(index)
        percent_of.append(h/float(number_of_nodes))
        index = index +1
    return connection,percent_of

def arrange_xcorr_peaks(pop_peak1,pop_peak2,butter1,butter2):
    value1 = []
    value2 = []

    for i in pop_peak1:
        value1.append(butter1[i])
    for i in pop_peak2:
        value2.append(butter2[i])

    yaxis = max(max(value1),max(value2))
    yaxis = np.ceil(yaxis)

    return value1,value2,yaxis
    



def main(argv=None):
     if argv is None:
        argv = sys.argv
     (inFn,outFn) = parse_args(argv)
     data_dict = scipy.io.loadmat(inFn)
     number_of_nodes = data_dict['number_of_nodes'][0]
     timespan = data_dict['time'][0]

     ##RASTER PLOT
     
     axes = plt.figure(1,figsize=(20,10)).add_subplot(111)
     spike_times,nueron_spike = arrange_raster(data_dict['spike_mat_bin'])
     plt.scatter(spike_times,nueron_spike,marker = '|')
     plt.xlabel('Time (S)')
     plt.ylabel('Nueron #')
     plt.title('Population Raster')
     plt.axis([0,timespan/20,0,number_of_nodes])
     a=axes.get_xticks().tolist()
     newticks = [int(x*20/1000) for x in a]
     axes.set_xticklabels(newticks)
     plt.savefig(outFn+"_raster.jpeg")

     ##CONNECTION MATRIX PLOT
     plt.figure(2)
     ax = plt.subplot(111)
     inhx,inhy,exx,exy = arrange_connection_matrix(data_dict['graph_edges'],data_dict['cells_inhib'])
     plt.scatter(exx,exy,label = 'Excitatory')
     plt.scatter(inhx,inhy,c='r', label  = 'Inhibitory')
     plt.axis([-.5,number_of_nodes,-.5,number_of_nodes])
     plt.xlabel('Nueron # (connector)')
     plt.ylabel('Nueron # (conectee)')
     plt.title('Population Connection Matrix')
     ax.legend(bbox_to_anchor = (1,.5))
     plt.savefig(outFn+"_connectionMatrix.jpeg")

     ##HISTOGRAM PLOT
     plt.figure(3)
     connection,percent_of = arrange_histogram(data_dict['degree_histogram'],number_of_nodes)
     plt.bar(connection,percent_of,align='center')
     plt.axis([-.5,len(data_dict['degree_histogram'])-.5,0,1])
     plt.xlabel('# of Connections')
     plt.ylabel('Percentage of Population with connection')
     plt.title('Population Histogram of Connection')
     plt.savefig(outFn+"_histogram.jpeg")

     ##Convolution PLOT
     plt.figure(4,figsize=(30,10))
     ax = plt.subplot(111)
     pop_peak1 = data_dict['pop_burst_peak_pop1']
     pop_peak2 = data_dict['pop_burst_peak_pop2']
     butter1 = data_dict['butter_int_bin']
     butter2 = data_dict['butter_int_bin2']
     bins = data_dict['bins']
     peak_point1, peak_point2,yaxis = arrange_xcorr_peaks(pop_peak1,pop_peak2,butter1,butter2)
     plt.plot(bins/1000.,butter1, label = 'Population 1',linewidth='2')
     plt.plot(bins/1000.,butter2, c = 'r', label = 'Population 2',linewidth='2')
     plt.scatter(bins[pop_peak1]/1000.,peak_point1,marker = 'o',c = 'k', facecolors = 'none', linewidth='2')
     plt.scatter(bins[pop_peak2]/1000.,peak_point2,marker = 'o',c = 'k', facecolors = 'none',linewidth='2')
     plt.axis([0,timespan/1000,-.5,yaxis])
     plt.xlabel('Time (S)')
     plt.ylabel('Arbitrary?') ##figure out what that means
     plt.title('Convolution of Two populations')
     ax.legend(bbox_to_anchor=(1.1,1.13))
     plt.savefig(outFn+"_convolution.jpeg")
     
     ##CROSS CORRELATION
     plt.figure(5, figsize=(15,5))
     ax  = plt.subplot(111)
     xcorr = data_dict['cross_correlation']
     autocorr1 = data_dict['auto_cross_correlation1']
     phase_lag = data_dict['phase_lag']
     plt.plot(xcorr, label = 'X correlation')
     plt.plot(autocorr1, c = 'r', label = 'Auto Correlation')
     plt.xlabel('Tao (Shift in graph)')
     plt.ylabel('Amplitude?') ##check this def
     plt.title('Cross Correlation vs Auto Correlation')
     ax.legend(bbox_to_anchor = (1.1,.5))
     x = plt.axis()
     plt.text(180,x[3]/3,'Phase lag: %.3f' %(phase_lag))
     
     
    ##NORMALIZED CROSS CORRELATION
     normXCorr = data_dict['normalized_cross_correlation']
     pop_corr = data_dict['pop_correlation']
     yaxis = data_dict['max_time_norm']
     plt.plot(normXCorr,c='g',ls = '--', label = 'Normalized X Correlation')
     plt.text(180,0,'Pop Correlation: %.3f' %(pop_corr))
     ax.legend(bbox_to_anchor = (1.1,1))
     plt.savefig(outFn+"_crossCorrelation_normalized.jpeg")

    ##Normalized Solo Graph
     plt.figure(6)
     normXCorr = data_dict['normalized_cross_correlation']
     pop_corr = data_dict['pop_correlation']
     yaxis = data_dict['max_time_norm']
     plt.plot(normXCorr,c='g',ls = '--', label = 'Normalized X Correlation')
     plt.xlabel('Tao (Shift in graph)')
     plt.ylabel('Coefficient of Correlation')
     plt.title('Normalized Cross Correlation')
     plt.text(130,.5,'Pop Correlation: %.3f' %(pop_corr))
     plt.savefig(outFn+"_popcorrelation.jpeg")

     

if __name__ == '__main__':
    status = main()
    sys.exit(status)     
