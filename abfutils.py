import pyabf
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from signal_classes import ProcessedSignal, AnalyzedSignal
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from detect_peaks import detect_peaks

MAX_RECORDING_THRESHOLD = 77000
MIN_RECORDING_THRESHOLD = -77000
SAMPLING_FREQ = 5000
WINDOW = 20
STIM_PULSE_STD = 9000
SIGNAL_STD = 50

def process_channel(channel, w=WINDOW, max_thresh=MAX_RECORDING_THRESHOLD,
                    min_thresh=MIN_RECORDING_THRESHOLD, stim_std=STIM_PULSE_STD,
                    width_thresh="auto", discard_peaks="auto", peak_thresh = 21):
    # channel is a .abf file for a single channel
    # width_thresh is for detect_peaks
    # others are input to detect_stim_pulse
    # returns (b, output) where b is a bool for whether processing was successful
    # and output is either original x,y data(raw) for unsuccessful process or
    # a ProcessedSignal object if process was successful
    # discard_peaks determines the offset to apply, can be a number or "auto"
    # to apply the automatic algorithm
    abf_channel = pyabf.ABF(channel)
    abf_channel.setSweep(0)
    #  initialize for ProcessedSignal object
    raw = [abf_channel.sweepX, abf_channel.sweepY]
    f = abf_channel.dataRate
    # detect stim
    (b, stim) = is_interesting(abf_channel.sweepY, max_thresh,
                                  min_thresh, f, w, stim_std, signal_std=10)
    if not b:
        return (b, raw)
    else:
        intra_idx = stim["intra"][0]
        extra_idx = stim["extra"]
        intra_data = abf_channel.sweepY[intra_idx[0]:intra_idx[1]]
        extra_data = abf_channel.sweepY[extra_idx[0]:extra_idx[1]]
        intra_time= abf_channel.sweepX[intra_idx[0]:intra_idx[1]]
        extra_time= abf_channel.sweepX[extra_idx[0]:extra_idx[1]]
        intra_smooth = smooth_data(intra_data, w)
        extra_smooth = smooth_data(extra_data, w)
        intra_std = rolling_std(intra_smooth, w)
        extra_std = rolling_std(extra_smooth, w)
        # determine thresholds for stdev by percentile
        intra_std_thresh = std_threshold(intra_std)
        # print(intra_std_thresh)
        extra_std_thresh = std_threshold(extra_std, percentile=99)
        # print(extra_std_thresh)
        if width_thresh == "auto":
            intra_pkloc_temp = detect_peaks(intra_std, mph=intra_std_thresh, mpd=f*0.5)
            if len(intra_pkloc_temp) < peak_thresh:
                return(False, raw)
            first = intra_pkloc_temp[0]
            last = intra_pkloc_temp[peak_thresh-1]
            width_t = (intra_time[last] - intra_time[first])/(peak_thresh*1.3)
            
        else:
            width_t = width_thresh
            
        intra_pkloc = detect_peaks(intra_std, mph = intra_std_thresh, mpd=f*width_t)
        extra_pkloc = detect_peaks(extra_std, mph = extra_std_thresh, mpd=f*width_t)       
        if len(intra_pkloc) < peak_thresh:
        # if you can't detect a reasonable number of peaks, signal is probably too small
            return (False, raw)
            
        # discard peaks for intra
        if discard_peaks == "auto":
            to_discard = get_discard_peaks(intra_smooth, intra_pkloc)
            idx_peaks = intra_pkloc[to_discard]
            intra_smooth = intra_smooth[idx_peaks:]
            intra_time = intra_time[idx_peaks:]
            intra_std = intra_std[idx_peaks:]
            intra_pkloc = intra_pkloc[1+to_discard:]-idx_peaks
        else:
            idx_peaks = intra_pkloc[discard_peaks]
            intra_smooth = intra_smooth[idx_peaks:]
            intra_time = intra_time[idx_peaks:]
            intra_std = intra_std[idx_peaks:]
            intra_pkloc = intra_pkloc[1+discard_peaks:]-idx_peaks
        
        extra_pkloc = extra_pkloc[1:]
        
    intra = [intra_time, intra_smooth, intra_std, intra_pkloc]
    extra = [extra_time, extra_smooth, extra_std, extra_pkloc]
    return (b, ProcessedSignal(raw, intra, extra, f, width_t))

def get_discard_peaks(data, std_pkloc,val_thresh=12000, thresh=3000):
    # data fields in np arrays
    # given std data, pks picked for std and data, determine
    # how many peaks to discard.
    # thresh = delta change in mV to continue discarding
    discard = 0
    prev_data = data[std_pkloc[discard]]
    next_data = data[std_pkloc[discard+1]]
    diff = abs(next_data-prev_data)
    while diff >= thresh or abs(prev_data) > val_thresh:
        discard+=1
        prev_data = data[std_pkloc[discard]]
        next_data = data[std_pkloc[discard+1]]
        diff = abs(next_data-prev_data)
        
    return discard

def std_threshold(std, method='MAD', percentile=99.5):
    if method == 'MAD':
        noise_std = np.nanmedian(np.abs(std))/0.6745
    elif method == 'STD':
        noise_std = np.nanstd(std)
    return min(noise_std * 20, np.nanpercentile(std, percentile))

def get_interesting_channels(pyabf_file):
    # takes as input a pyabf format object, and outputs the channel/channel
    # names that look like they might have interesting data.
    pyabf_file.setSweep(0)
    # list of names for the channels that have interesting data
    good_channels_names = []
    # list of indices for the channels that have interesting data
    good_channels = []
    # list of dicts
    good_channels_stim = []
    for i in range(pyabf_file.channelCount):
        test_file = np.array(pyabf_file.data[i])
        test_file_name = pyabf_file.adcNames[i]
        b,d = is_interesting(test_file, f=pyabf_file.dataRate)
        if b:
            good_channels.append(i)
            good_channels_names.append(test_file_name)
            good_channels_stim.append(d)
    return (good_channels, good_channels_names, good_channels_stim)

def smooth_data(dataset, window):
    # takes in dataset as a 1D np array, returns moving time avg
    return np.convolve(dataset, np.ones((window,))/float(window), mode='same')

def detect_stim_pulse(dataset, max_thresh = MAX_RECORDING_THRESHOLD,
                      min_thresh = MIN_RECORDING_THRESHOLD, f = SAMPLING_FREQ,
                      w = WINDOW, stim_std = STIM_PULSE_STD):
    # dataset: 1D np array
    # takes in dataset, returns (bool, ndarray) of whether there was a stim
    # + locations of stim if it exists
    if np.max(dataset) > max_thresh:
        a = np.argwhere(dataset == np.max(dataset)).flatten()
        stim = _splitArray(a, f*5)
        return (True, stim)
    elif np.min(dataset) < min_thresh:
        a = np.argwhere(dataset == np.min(dataset)).flatten()
        stim = _splitArray(a, f*5)
        return (True, stim)
    else:
        smoothed = smooth_data(dataset, w)
        std = rolling_std(smoothed, w)
        max_std = np.nanmax(std)
        if max_std < stim_std:
            return (False, [])
        else:
            not_nan = std[np.invert(np.isnan(std))]
            not_nan_stim = not_nan[np.argwhere(not_nan > stim_std)]
            i = np.in1d(std, not_nan_stim)
            s = np.argwhere(i).flatten()
            stim = _splitArray(s, f*5)
            return (True, stim)

def is_interesting(sample, max_thresh = MAX_RECORDING_THRESHOLD,
                   min_thresh = MIN_RECORDING_THRESHOLD,
                   f = SAMPLING_FREQ, w = WINDOW, stim_std = STIM_PULSE_STD,
                   signal_std = SIGNAL_STD):
    # sample: 1D np array
    # returns (bool, dict) for if a sample potentially is interesting.
    # if it is interesting, dict is a pointer to (start,end)
    # of indices where the data is likely to be interesting. should have
    # a max of 2 entries, "extra" and "intra".
    # tests if data is interesting by the std of the first sample of intra
    d = {}
    b, stim = detect_stim_pulse(sample, max_thresh, min_thresh, f, w, stim_std)
    if b == False:
        return (b, d)
    else:
        smoothed = smooth_data(sample, w)
        std = rolling_std(smoothed, w)
        num_pulses = len(stim)
        
        first = stim[0][0]
        # handles if signal saturation happens at the very beginning
        if first == 0:
            first = stim[1][0]
            num_pulses -=1
            stim = stim[1:]
        extra = (0, first)
        d["extra"] = extra
        intra = []
        for i in range(num_pulses):
            first_elem = stim[i][len(stim[i])-1]
            if i+1 < num_pulses:
                last_elem = stim[i+1][0]
                intra.append((first_elem,last_elem))
            else:
                intra.append((first_elem, len(sample)))
        d["intra"] = intra
        start, stop = (int(intra[0][0] + 5*f), int(intra[0][1] - 5*f)) # buffer
        if np.nanmax(std[start:stop]) < signal_std:
            return (False, d)
        return (b, d)

def rolling_std(a, window):
    pd_a = pd.Series(a)
    pd_std = pd_a.rolling(window, min_periods=1).std()
    return np.array(pd_std.values)
    
def _splitArray(to_split, noise):
    # helper function to split 1D integer array into consecutive sequences.
    # assumes array is sorted.
    # returns a 2d array containing all sublists of consecutive lists.
    if len(to_split) == 0:
        return
    if len(to_split) == 1:
        return np.array([to_split])
    master = []
    sz = len(to_split)
    prev = to_split[0]
    temp = [to_split[0]]
    for i in range(1,sz):
        cur = to_split[i]
        if cur == prev+1:
            temp.append(cur)
            prev = cur
        elif cur <= (prev + noise): #account for noise
            temp.append(cur)
            prev = cur
        else:
            master.append(temp)
            temp = [cur]
            prev = cur
    master.append(temp)
    return np.array(master)
