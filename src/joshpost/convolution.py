#!/usr/bin/env python

import sys
import numpy as np
import scipy.signal
import scipy.io
import argparse
import networkx as nx
#from matplotlib.pyplot import acorr, psd
import matplotlib.pyplot as plt
        
maxorder=20
eta_norm_pts = 10

def parse_args(argv):
    # defaults
    transient = 1000 # ms
    spike_thresh = -20 # mV
    f_sigma = 20 # ms
    butter_high = 10 # Hz
    butter_low = 1 # Hz
    bin_width = 20 # ms
    cutoff = 0.5
    # parsing
    parser = argparse.ArgumentParser(prog="convolution",
                                     description=('Postprocessing of' 
                                                  ' model output'))
    parser.add_argument('sim', help='model output (.mat) file')
    parser.add_argument('output', help='output (.jpg) filename')
    parser.add_argument('--transient', '-t', 
                        help='transient time, ms (default: %(default)s)', 
                        type=float, default=transient)
    parser.add_argument('--sec', '-s', action='store_true',
                        help='time units are in seconds (default: ms)')
    parser.add_argument('--volt', '-V', action='store_true',
                        help=('file contains voltage traces '
                              '(default: sparse spike trains)'))
    parser.add_argument('--thresh', 
                        help='spike threshold, mV (default: %(default)s)',
                        type=float, default=spike_thresh) 
    parser.add_argument('--fsig', '-f', 
                        help=('filter standard deviation, ms '
                        '(default: %(default)s)'),
                        type=float, default=f_sigma)
    parser.add_argument('--butter_high', 
                        help=('Butterworth filter upper cutoff frequency, Hz '
                              '(default: %(default)s)'),
                        type=float, default=butter_high)
    parser.add_argument('--butter_low', 
                        help=('Butterworth filter lower cutoff frequency, Hz '
                              '(default: %(default)s)'),
                        type=float, default=butter_low)
    parser.add_argument('--bin_width', '-b', 
                        help='bin width, ms (default: %(default)s)',
                        type=float, default=bin_width)
    parser.add_argument('--cut', '-c', 
                        help='burst cutoff parameter (default: %(default)s)',
                        type=float, default=cutoff)
    args = parser.parse_args(argv[1:])
    return args.sim, args.output, args.transient, args.sec, args.thresh, \
        args.fsig, args.butter_low, args.butter_high, args.bin_width,\
        args.cut, args.volt

def chop_transient(data, transient, dt):
    '''
    Remove a transient from the data
    '''
    firstIdx = int(np.ceil(transient / dt) - 1)
    return data[:,firstIdx:]

def find_spikes(data, threshold):
    '''
    Find spikes in voltage data by taking relative maxima
    '''
    indices = scipy.signal.argrelmax(data, axis=1) # 1st and 2nd coords of maxima
    mask = np.where(data[indices] > threshold)
    new_indices = (indices[0][mask],
                   indices[1][mask])
    spike_mat = np.zeros(np.shape(data), dtype=np.int) # dense format
    spike_mat[new_indices] = 1
    return new_indices, spike_mat

def spikes_of_neuron(spikes, neuron):
    '''
    Return time indices of spiking of a given neuron
    '''
    return spikes[1][np.where(spikes[0] == neuron)]

def spikes_filt(spike_mat, samp_freq, f_sigma, butter_freq):
    '''
    Filter the spike timeseries. Returns both neuron-by-neuron timeseries
    filtered with a gaussian kernel and the population data filtered
    with a butterworth filter.

    Parameters
    ==========
    spike_mat: the numneuron x time matrix of spikes
    samp_freq: sample frequency
    f_sigma:   variance of gaussian
    butter_freq: butterworth filter cutoff frequency(s)

    Returns
    =======
    spike_fil: gaussian filtered matrix, same shape as spike_mat
    int_signal: butterworth filtered population timeseries
    spike_fil_butter: butterworth filtered matrix, same shape as spike_mat
    '''
    def filt_window_gauss(samp_freq, std = 20, width = None, normalize = 1):
        if width is None:
            width = std*4+1
        width /= samp_freq
        std /= samp_freq
        w = scipy.signal.gaussian(width, std)
        if not normalize == 0:
            w = normalize * w / sum(w)
        return w
    def filt_gauss(spike_mat, samp_freq, f_sigma=20):
        w = filt_window_gauss(samp_freq, std=f_sigma, normalize=1)
        spike_fil = scipy.signal.fftconvolve(spike_mat, w[ np.newaxis, : ], 
                                             mode='same')
        #spike_fil = scipy.signal.convolve(spike_mat, w[ np.newaxis, : ], 
        #                                  mode='same')
        return spike_fil
    def filt_butter(data, samp_freq, butter_freq, axis=-1):
        samp_freq *= 1e-3 # ms -> s, filter defn in terms of Hz
        order = 2
        ny = 0.5 / samp_freq # Nyquist frequency
        cof = butter_freq # cutoff freq, in Hz
        cof1 = cof / ny
        b, a = scipy.signal.butter(order, cof1, btype='low')
        filtNs = scipy.signal.filtfilt(b, a, data, axis=axis)
        filtNs /= samp_freq
        return filtNs
    spike_fil = filt_gauss(spike_mat, samp_freq, f_sigma=f_sigma) 
    int_signal = filt_butter(np.mean(spike_mat, axis=0), 
                             samp_freq, butter_freq)
    spike_fil_butter = filt_butter(spike_fil, samp_freq, butter_freq, axis=1)
    return spike_fil, int_signal, spike_fil_butter

def bin_spikes(spike_mat, bin_width, dt):
    '''
    Bin spikes

    Parameters
    ==========
      spike_mat: matrix of spikes, (num_neuron x num_time)
      bin_width: bin width in time units
      dt: sampling frequency in spike mat

    Returns
    =======
      bins: an array of the bin locations in time units
      binned_spikes: a new matrix (num_neuron x num_bins)
    '''
    num_neurons= np.shape(spike_mat)[0]
    num_times = np.shape(spike_mat)[1]
    stride = int(np.ceil(bin_width / dt))
    bins = np.arange(0, num_times, stride, dtype=np.float)
    which_bins = np.digitize(range(0, num_times), bins)
    num_bins = len(bins)
    binned_spikes = np.zeros((num_neurons, num_bins), dtype=np.int)
    for i in range(num_bins):
        bin_mask = np.where(which_bins == i)[0] # mask data in bin i, tuple
        bin_data = spike_mat[:,bin_mask]
        binned_spikes[:,i] = np.sum(bin_data, axis=1).flatten()
    return bins, binned_spikes


##Implementation of the mean to make sure all code is running correctly
def find_mean(data):
    sum = 0;
    for x in data:
        sum = sum + x
    return sum/len(data)

'''
This is computes the cross correlation for two signals

paramters:
    signal_1: first signal you want to use
    signal_2: second signal you want to use
    taolen: this number determines how much of the tao to use
returns:
    values of the cross correlation 
'''
def xcorr(signal_1,signal_2,taolen):
    xcorrl = np.empty(len(signal_1)/taolen)
    index = 0
    

    ## m1,m2 are the means of signals 1 and 2, respectivly
    m1 = find_mean(signal_1)
    m2 = find_mean(signal_2)
    
    #loops through all values of tao 
    for tao in range(0,(len(signal_1)/taolen)):
        sum = 0
        for t in range(0,(len(signal_1) - tao)):
            sum = sum + ((signal_1[t]-m1)*(signal_2[t+tao]-m2))
        average = sum / (len(signal_1) - tao)
        xcorrl[index] = average
        index = index + 1

    return xcorrl   
        
        


def main(argv = None):
    if argv is None:
        argv = sys.argv
    (simFn, outFn, trans, sec_flag, spike_thresh, f_sigma, butter_low, 
     butter_high, bin_width, cutoff, are_volts) = parse_args(argv)
    butter_freq = np.array([butter_low, butter_high])
    if sec_flag:
        scalet = 1e3
    else:
        scalet = 1
    
    ## load simulation output
    ## assumes no --save_full
    sim_output = scipy.io.loadmat(simFn)
    ## begin postprocessing
    graph_fn = str(sim_output['graphFn'][0])
    dt = float(sim_output['dt']) * scalet
    data = chop_transient(sim_output['Y'], trans, dt)
    num_neurons = np.shape(data)[0]
    tMax = np.shape(data)[1]
    ## spike trains
    if are_volts:
        spikes, spike_mat = find_spikes(data, spike_thresh)
    else:
        spike_mat = data.todense()
        spikes = data.nonzero()
    bins, spike_mat_bin = bin_spikes(spike_mat, bin_width, dt)
    ## filtering
    # spike_fil, butter_int = spikes_filt(spike_mat, dt, f_sigma, butter_freq)
    spike_fil_bin, butter_int_bin, spike_fil_butter = spikes_filt(spike_mat_bin[0:41], 
                                                                  dt*bin_width, 
                                                                  f_sigma, 
                                                                  butter_freq)
    spike_fil_bin2, butter_int_bin2, spike_fil_butter2 = spikes_filt(spike_mat_bin[41:], 
                                                                  dt*bin_width, 
                                                                  f_sigma, 
                                                                  butter_freq)

    ##Plots the convolutions based on the butterworth filter 
    

    plt.plot(bins/1000.,butter_int_bin)
    plt.plot(bins/1000.,butter_int_bin2,c ='r')

        
    
    
    #Burst Peaks need 
    pop_burst_peak = scipy.signal.argrelmax(butter_int_bin, order=maxorder)[0]
    pop_burst_peak2 = scipy.signal.argrelmax(butter_int_bin2, order=maxorder)[0]

    

    value1 = []
    value2 = []

    for i in pop_burst_peak:
        value1.append(butter_int_bin[i])

    for i in pop_burst_peak2:
        value2.append(butter_int_bin2[i])

        
    ##plots peaks of butterworth convolution
    plt.scatter(bins[pop_burst_peak]/1000.,value1 , marker ="o", c = 'k',facecolors= 'none')
    plt.scatter(bins[pop_burst_peak2]/1000.,value2 , marker ="o", c = 'k',facecolors='none')
    plt.savefig(outFn)

    plt.show()

    ##graphs the cross correlation
    cross_correlation = xcorr(butter_int_bin,butter_int_bin2,10)
    auto_cross_correlation1 = xcorr(butter_int_bin,butter_int_bin,10)
    auto_cross_correlation2 = xcorr(butter_int_bin2,butter_int_bin2,10)

    normalized_cross_correlation = cross_correlation/np.sqrt(auto_cross_correlation1[0]*auto_cross_correlation2[0])
    
    print float(scipy.signal.argrelmax(cross_correlation)[0])/float(scipy.signal.argrelmax(auto_cross_correlation1)[0])
    plt.plot(cross_correlation)
    plt.plot(auto_cross_correlation1,c='r')
    plt.show()

    plt.close()

    max_time = scipy.signal.argrelmax(normalized_cross_correlation)[0]
    print(normalized_cross_correlation[max_time])

    plt.plot(normalized_cross_correlation,lw=1.5)
    plt.show()
    plt.close()


    spike_times = []
    nueron_spike = []
    for j in range(0,len(spike_mat_bin)):
        for i in range(0,len(spike_mat_bin[j])):
            if spike_mat_bin[j][i] == 1:
                spike_times.append(i)
                nueron_spike.append(j)

    plt.scatter(spike_times,nueron_spike, marker = '|')
    plt.show()
    

if __name__ == '__main__':
    status = main()
    sys.exit(status) 
    



