import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from signal_classes import ProcessedSignal, AnalyzedSignal
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from detect_peaks import detect_peaks
#MCS PyData
import McsPy.McsData
from McsPy import ureg, Q_

MAX_RECORDING_THRESHOLD = 60000 #uV
MIN_RECORDING_THRESHOLD = -60000
WINDOW = 20
STIM_PULSE_STD = 3000
BUFFER_SECONDS = 10

def load_h5(file_path):
    # takes in a path to a .h5 file
    # all data of interest is in recordings[0].analog_streams[0]
    # returns (data, labels, timestamps, frequency)
    # where data, labels are lists containing n values
    # where each value is an np array of data
    # time is an np array of timestamps in s
    # frequency is the sampling rate
    file = McsPy.McsData.RawData(file_path)
    analog_stream = file.recordings[0].analog_streams[0]
    info_shared = analog_stream.channel_infos[0]
    frequency = info_shared.sampling_frequency.magnitude
    last_idx = analog_stream.channel_data.shape[1]
    time_raw = analog_stream.get_channel_sample_timestamps(0, 0, last_idx)
    # scale data to uV
    unit_scale = Q_(1,info_shared.info['Unit']).to(ureg.uV).magnitude
    scale_factor = info_shared.info['ConversionFactor']*(10**info_shared.info['Exponent'].astype(np.float64))
    scale_factor_uV = scale_factor*unit_scale
    data = np.asarray(analog_stream.channel_data)*scale_factor_uV
    # scale time to s
    scale_factor_s = Q_(1, time_raw[1]).to(ureg.s).magnitude
    time = (time_raw[0]*scale_factor_s)
    
    labels = [channel_info.info['Label'] for channel_info in analog_stream.channel_infos.values()]
    print("Data loaded!")
    return (data, labels, time, frequency)

def process_single_channel_crossing(time, data, f, dead_time=BUFFER_SECONDS,
                                    max_thresh = MAX_RECORDING_THRESHOLD, w = WINDOW,
                                    width_thresh="auto", discard_peaks="auto", peak_thresh = 30):
    # used to process single channel of analogstream recording
    # width_thresh is for detect_peaks
    # others are input to detect_stim_pulse
    # returns (b, output) where b is a bool for whether processing was successful
    # and output is either original x,y data(raw) for unsuccessful process or
    # a ProcessedSignal object if process was successful
    # discard_peaks determines the offset to apply, can be a number or "auto"
    # to apply the automatic algorithm
    #  initialize for ProcessedSignal object
    raw = [time, data]
    # detect stim
    (b, stim, smoothed, std) = is_interesting_crossing(data, f, dead_time,
                                                       max_thresh, w)
    if not b:
        return (b, raw)
    else:
        intra_idx = stim["intra"][0]
        extra_idx = stim["extra"]
        intra_time= time[intra_idx[0]:intra_idx[1]]
        extra_time= time[extra_idx[0]:extra_idx[1]]
        intra_smooth = smoothed[intra_idx[0]:intra_idx[1]]
        extra_smooth = smoothed[extra_idx[0]:extra_idx[1]]
        intra_std = std[intra_idx[0]:intra_idx[1]]
        extra_std = std[extra_idx[0]:extra_idx[1]]
        # determine thresholds for stdev by percentile
        intra_std_thresh = std_threshold(intra_std)
        extra_std_thresh = std_threshold(extra_std, percentile=99.5)
        if width_thresh == "auto":
            intra_pkloc_temp = detect_peaks(intra_std, mph=intra_std_thresh, mpd=f*0.5)
            if len(intra_pkloc_temp) < peak_thresh:
                return(False, raw)
            first = intra_pkloc_temp[0]
            last = intra_pkloc_temp[peak_thresh-1]
            width_t = (intra_time[last] - intra_time[first])/(peak_thresh*1.5)
            
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
            if not to_discard:
                return (False, raw)
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

def get_discard_peaks(data, std_pkloc,val_thresh=12000, thresh=1000):
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
        if discard+1 >= len(std_pkloc):
            return False
        prev_data = data[std_pkloc[discard]]
        next_data = data[std_pkloc[discard+1]]
        diff = abs(next_data-prev_data)
        
    return discard

def std_threshold(std, method='MAD', percentile=99):
    if method == 'MAD':
        noise_std = np.nanmedian(np.abs(std))/0.6745
    elif method == 'STD':
        noise_std = np.nanstd(std)
    return min(noise_std * 15, np.nanpercentile(std, percentile))

def get_interesting_channels(h5_file):
    # takes as input a h5 file name
    # returns (channels, channel_idx, signals)
    # which are lists with the channel names, indices as well
    # as the signals in ProcessedSignal form
    (data, labels, time, f) = load_h5(h5_file)
    # list of names for the channels that have interesting data
    good_channels = []
    channel_idx = []
    # list of ProcessedSignals
    signals = []
    for i in range(len(data)):
        b,sig = process_single_channel_crossing(time, data[i], f)
        if b:
            good_channels.append(labels[i])
            channel_idx.append(i)
            signals.append(sig)
    return (good_channels, channel_idx, signals)

def smooth_data(dataset, window):
    # takes in dataset as a 1D np array, returns moving time avg
    return np.convolve(dataset, np.ones((window,))/float(window), mode='same')
        
def detect_stim_pulse_crossing(dataset, f, dead_time=BUFFER_SECONDS, max_thresh = MAX_RECORDING_THRESHOLD):
    # dataset: 1D np array
    # takes in dataset, returns (bool, ndarray) of whether there was a stim
    # + locations of stim if it exists
    dead_time_idx = dead_time * f
    if np.max(np.abs(dataset)) > max_thresh:
        crossings = np.argwhere(np.abs(dataset) == np.max(np.abs(dataset))).flatten()
        distance_sufficient = np.insert(np.diff(crossings) >= dead_time_idx, 0, True)
        while not np.all(distance_sufficient):
        # repeatedly remove all threshold crossings that violate the dead_time
            crossings = crossings[distance_sufficient]
            distance_sufficient = np.insert(np.diff(crossings) >= dead_time_idx, 0, True)
        return (True, crossings)
    else:
        return (False, [])

def is_interesting_crossing(sample, f, dead_time=BUFFER_SECONDS,
                            max_thresh = MAX_RECORDING_THRESHOLD, w = WINDOW):
    # sample: 1D np array
    # returns (bool, dict, data, std) for if a sample potentially is interesting.
    # data is raw data if it's not interesting, and smoothed if it is
    # std is an empty array if it's not interesting, otherwise is the std
    # if it is interesting, dict is a pointer to (start,end)
    # of indices where the data is likely to be interesting. should have
    # a max of 2 entries, "extra" and "intra".
    # tests if data is interesting by the std of the first sample of intra
    d = {}
    b, crossing = detect_stim_pulse_crossing(sample, f, dead_time, max_thresh)
    if b == False:
        return (b, d, sample, [])
    else:
        smoothed = smooth_data(sample, w)
        std = rolling_std(smoothed, w)
        num_pulses = len(crossing)
        
        first = crossing[0]
        # handles if signal saturation happens at the very beginning
        if first == 0:
            if num_pulses == 1:
                return (False, d, sample, [])
            first = crossing[1]
            num_pulses -=1
            crossing = crossing[1:]
        extra = (0, first)
        d["extra"] = extra
        intra = []
        for i in range(num_pulses):
            first_elem = crossing[i]
            if i+1 < num_pulses:
                last_elem = crossing[i+1]
                intra.append((first_elem,last_elem))
            else:
                intra.append((first_elem, len(sample)))
        d["intra"] = intra
        start = int(intra[0][0] + 5*f) # buffer
        stop = int(intra[0][0] + 25*f) # don't really have to look at more than 20 seconds
        #print(np.nanpercentile(std[start:stop], 99))
        #print(5*np.nanmedian(np.abs(std))/0.6745)
        #print(np.nanmax(std[start:stop]))
        if (np.nanpercentile(std[start:stop], 99) < (5*np.nanmedian(std)/0.6745)) or (np.nanmax(std[start:stop]) < 10):
            return (False, d, sample, [])
        return (b, d, smoothed, std)

def rolling_std(a, window):
    pd_a = pd.Series(a)
    pd_std = pd_a.rolling(window, min_periods=1).std()
    return np.array(pd_std.values)


def plot_channel_by_idx(h5_file, channel_idx, from_in_s=0, to_in_s=None, show=True):
    """
    Plots data from a single AnalogStream channel from an h5 file by label.
    
    :param h5_file: path to an h5 file
    :param channel_idx: A scalar channel index (0 <= channel_idx < # channels in the AnalogStream)
    :param from_in_s: The start timestamp of the plot (0 <= from_in_s < to_in_s). Default: 0
    :param to_in_s: The end timestamp of the plot (from_in_s < to_in_s <= duration). Default: None (= recording duration)
    :param show: If True (default), the plot is directly created. For further plotting, use show=False
    """
    # extract basic information
    file = McsPy.McsData.RawData(h5_file)
    analog_stream = file.recordings[0].analog_streams[0]
    ids = [c.channel_id for c in analog_stream.channel_infos.values()]
    channel_id = ids[channel_idx]
    channel_info = analog_stream.channel_infos[channel_id]
    sampling_frequency = channel_info.sampling_frequency.magnitude
   
    # get start and end index
    from_idx = max(0, int(from_in_s * sampling_frequency))
    if to_in_s is None:
        to_idx = analog_stream.channel_data.shape[1]
    else:
        to_idx = min(analog_stream.channel_data.shape[1], int(to_in_s * sampling_frequency))
        
    # get the timestamps for each sample
    time = analog_stream.get_channel_sample_timestamps(channel_id, from_idx, to_idx)

    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second
    
    # get the signal
    signal = analog_stream.get_channel_in_range(channel_id, from_idx, to_idx)

    # scale signal to µV:
    scale_factor_for_uV = Q_(1,signal[1]).to(ureg.uV).magnitude
    signal_in_uV = signal[0] * scale_factor_for_uV

    # construct the plot
    _ = plt.figure(figsize=(8,5))
    _ = plt.plot(time_in_sec, signal_in_uV)
    _ = plt.xlabel('Time (%s)' % ureg.s)
    _ = plt.ylabel('Voltage (%s)' % ureg.uV)
    _ = plt.title('Channel %s' % channel_info.info['Label'])
    if show:
        plt.show()

####################################### DEPRECATED #####################################
def process_single_channel(time, data, f, w=WINDOW, max_thresh=MAX_RECORDING_THRESHOLD,
                    min_thresh=MIN_RECORDING_THRESHOLD, stim_std=STIM_PULSE_STD,
                    width_thresh="auto", discard_peaks="auto", peak_thresh = 30):
    # used to process single channel of analogstream recording
    # width_thresh is for detect_peaks
    # others are input to detect_stim_pulse
    # returns (b, output) where b is a bool for whether processing was successful
    # and output is either original x,y data(raw) for unsuccessful process or
    # a ProcessedSignal object if process was successful
    # discard_peaks determines the offset to apply, can be a number or "auto"
    # to apply the automatic algorithm
    #  initialize for ProcessedSignal object
    raw = [time, data]
    # detect stim
    (b, stim, smoothed, std) = is_interesting(data, f, max_thresh,
                                              min_thresh, w, stim_std)
    if not b:
        return (b, raw)
    else:
        intra_idx = stim["intra"][0]
        extra_idx = stim["extra"]
        intra_time= time[intra_idx[0]:intra_idx[1]]
        extra_time= time[extra_idx[0]:extra_idx[1]]
        intra_smooth = smoothed[intra_idx[0]:intra_idx[1]]
        extra_smooth = smoothed[extra_idx[0]:extra_idx[1]]
        intra_std = std[intra_idx[0]:intra_idx[1]]
        extra_std = std[extra_idx[0]:extra_idx[1]]
        # determine thresholds for stdev by percentile
        intra_std_thresh = std_threshold(intra_std)
        extra_std_thresh = std_threshold(extra_std, percentile=99.5)
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
            if not to_discard:
                return (False, raw)
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

def detect_stim_pulse(dataset, f, max_thresh = MAX_RECORDING_THRESHOLD,
                      min_thresh = MIN_RECORDING_THRESHOLD,
                      w = WINDOW, stim_std = STIM_PULSE_STD):
    # dataset: 1D np array
    # takes in dataset, returns (bool, ndarray) of whether there was a stim
    # + locations of stim if it exists
    if np.max(dataset) > max_thresh:
        a = np.argwhere(dataset == np.max(dataset)).flatten()
        stim = _splitArray(a, f*10)
        return (True, stim)
    elif np.min(dataset) < min_thresh:
        a = np.argwhere(dataset == np.min(dataset)).flatten()
        stim = _splitArray(a, f*10)
        return (True, stim)
    else:
        smoothed = smooth_data(dataset, w)
        std = rolling_std(smoothed, w)
        max_std = np.nanmax(std[40:]) #rolling std might make for some weird stuff
        if max_std < stim_std:
            return (False, [])
        else:
            not_nan = std[np.invert(np.isnan(std))]
            not_nan_stim = not_nan[np.argwhere(not_nan > stim_std)]
            i = np.in1d(std, not_nan_stim)
            s = np.argwhere(i).flatten()
            stim = _splitArray(s, f*10)
            return (True, stim)

def is_interesting(sample, f, max_thresh = MAX_RECORDING_THRESHOLD,
                   min_thresh = MIN_RECORDING_THRESHOLD,
                   w = WINDOW, stim_std = STIM_PULSE_STD):
    # sample: 1D np array
    # returns (bool, dict, data, std) for if a sample potentially is interesting.
    # data is raw data if it's not interesting, and smoothed if it is
    # std is an empty array if it's not interesting, otherwise is the std
    # if it is interesting, dict is a pointer to (start,end)
    # of indices where the data is likely to be interesting. should have
    # a max of 2 entries, "extra" and "intra".
    # tests if data is interesting by the std of the first sample of intra
    d = {}
    b, stim = detect_stim_pulse(sample, f, max_thresh, min_thresh, w, stim_std)
    if b == False:
        return (b, d, sample, [])
    else:
        smoothed = smooth_data(sample, w)
        std = rolling_std(smoothed, w)
        num_pulses = len(stim)
        
        first = stim[0][0]
        # handles if signal saturation happens at the very beginning
        if first == 0:
            if len(stim) == 1:
                return (False, d, sample, [])
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
        start = int(intra[0][0] + 5*f) # buffer
        stop = int(intra[0][0] + 25*f) # don't really have to look at more than 20 seconds
        if (np.nanpercentile(std[start:stop], 99) < (5*np.nanmedian(std)/0.6745)) or (np.nanmax(std[start:stop]) < 10):
            #print(5*np.nanmedian(np.abs(std))/0.6745)
            return (False, d, sample, [])
        return (b, d, smoothed, std)

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
