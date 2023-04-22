import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

class ProcessedSignal:
    def __init__(self, raw, intra, extra, f, width_t):
        # raw is an np array of raw channel data 
        # f is sampling frequency
        # intra and extra are size 4 arrays containing
        # np arrays of time, smoothed data, std data and
        # peak locs in that order
        self.time = raw[0]
        self.rawdata = raw[1]
        self.sampling_f = f
        self.intra_time = intra[0]
        self.intra_data = intra[1]
        self.intra_std = intra[2]
        self.intra_pkloc = intra[3]
        self.extra_time = extra[0]
        self.extra_data = extra[1]
        self.extra_std = extra[2]
        self.extra_pkloc = extra[3]
        self.width_t = width_t
        
    def analyze_intra(self, num_peaks=30):
        # returns an AnalyzedSignal object for the first n
        # num_peaks specified. If greater than number of peaks
        # in data, returns None.
        if len(self.intra_pkloc) < num_peaks+1:
            return
        # define data range
        data_select = self.intra_data[0:self.intra_pkloc[num_peaks]+1]
        baseline_values = baseline_als(data_select)
        zeroed_data = data_select - baseline_values
        pkloc_select = self.intra_pkloc[0:num_peaks+1]
        time_select = self.intra_time[0:pkloc_select[num_peaks]+1]
        std_select = self.intra_std[0:pkloc_select[num_peaks]+1]
        # calculate some stdevs
        std_med = np.nanmedian(std_select)
        std_select_sorted = np.sort(std_select)
        stdofstd = np.std(std_select_sorted[0:int(len(std_select_sorted)*0.6)])
        # initialize some arrays
        max_locs = np.zeros(num_peaks, dtype=int)
        min_locs = np.zeros(num_peaks, dtype=int)
        start_locs = np.zeros(num_peaks, dtype=int)
        for i in range(num_peaks):
            cur_data = zeroed_data[pkloc_select[i]:pkloc_select[i+1]]
            max_locs[i] = np.argmax(cur_data) + pkloc_select[i] - 1
            min_locs[i] = np.argmin(cur_data) + pkloc_select[i] - 1
            temp = pkloc_select[i]
            while std_select[temp]>(std_med + 5*stdofstd):
                if temp == 0:
                    break
                temp -= 1
            start_locs[i] = temp
        output = (time_select, data_select, std_select, pkloc_select,
                  max_locs, min_locs, start_locs, zeroed_data)
        return AnalyzedSignal(self.sampling_f, num_peaks, output)

    """
    def analyze_intra_zeroed(self, num_peaks=20, peak_thresh=85):
        # returns an AnalyzedSignal object for the first n
        # num_peaks specified. If greater than number of peaks
        # in data, returns none.
        # define data range
        if len(self.intra_pkloc) < num_peaks+2:
            return
        if len(self.intra_pkloc) < num_peaks*2:
            data_temp = self.intra_data[0:self.intra_pkloc[-1]]
        else:
            data_temp = self.intra_data[0:self.intra_pkloc[num_peaks*2-1]]
        baseline_values = baseline_als(data_temp)
        zeroed_data = data_temp - baseline_values
        #re-find peaks with zeroed data on just the trace
        pk_height = np.nanpercentile(zeroed_data, peak_thresh)
        pk_thresh = abs(np.nanpercentile(zeroed_data, 10))
        pkloc_select = detect_peaks(zeroed_data, mph=pk_height,
                                    mpd=self.sampling_f*self.width_t/2)
        #pkloc_select = pkloc_select[1:]
        #num_peaks = min(len(pkloc_select)-1, num_peaks)
        data_select = self.intra_data[0:pkloc_select[num_peaks+1]+1]
        time_select = self.intra_time[0:pkloc_select[num_peaks+1]+1]
        std_select = self.intra_std[0:pkloc_select[num_peaks+1]+1]
        zeroed_data = zeroed_data[0:pkloc_select[num_peaks+1]+1]
        # initialize some arrays
        pkloc_select = pkloc_select[1:num_peaks+2]
        max_locs = pkloc_select[:num_peaks]
        min_locs = np.zeros(num_peaks, dtype=int)
        start_locs = np.zeros(num_peaks, dtype=int)
        prev_data = zeroed_data[0:pkloc_select[0]]
        for i in range(num_peaks):
            cur_data = zeroed_data[pkloc_select[i]:pkloc_select[i+1]]
            min_locs[i] = np.argmin(cur_data) + pkloc_select[i] - 1
            d = int(self.sampling_f*0.04)
            n = len(prev_data) - d
            delta = cur_data[0] - prev_data[n]
            while delta > pk_thresh:
                n -= d
                if n < 0:
                    n=0
                    break
                delta = prev_data[n+d] - prev_data[n]
            start_locs[i] = pkloc_select[i] - (len(prev_data) - n)
            #start_locs[i] = pkloc_select[i] - np.argmin(np.abs(prev_data[::-1]))
            prev_data = cur_data
        output = (time_select, data_select, std_select, pkloc_select,
                  max_locs, min_locs, start_locs, zeroed_data)
        return AnalyzedSignal(self.sampling_f, num_peaks, output)
    """

    def plot_data(self, datatype='intra', num_peaks=30):
        # plots the data. If num_peaks > number of peaks available, plots
        # all data available.
        # datatype can be 'intra' or 'extra'
        plt.figure(figsize=(8,5))
        plot_time = []
        plot_data = []
        plot_std = []
        plot_pkloc = []
        if datatype == 'intra':
            plot_time = self.intra_time
            plot_data = self.intra_data
            plot_std = self.intra_std
            plot_pkloc = self.intra_pkloc
        elif datatype == 'extra':
            plot_time = self.extra_time
            plot_data = self.extra_data
            plot_std = self.extra_std
            plot_pkloc = self.extra_pkloc
        else:
            print("Unrecognized datatype to plot (only accepts 'intra' or 'extra').")
            return
            
        if len(plot_pkloc) < num_peaks+1:
            plt.plot(plot_time, plot_data, 'b')
            plt.plot(plot_time, plot_std, 'r')
            plt.plot(plot_time[plot_pkloc], plot_std[plot_pkloc], 'm+')
            plt.show()
        else:
            pkloc_select = plot_pkloc[0:num_peaks+1]
            time_select = plot_time[0:pkloc_select[num_peaks]+1]
            data_select = plot_data[0:pkloc_select[num_peaks]+1]
            std_select = plot_std[0:pkloc_select[num_peaks]+1]
            plt.plot(time_select, data_select, 'b')
            plt.plot(time_select, std_select, 'r')
            plt.plot(time_select[pkloc_select], std_select[pkloc_select], 'm+')
            plt.show()
            
class AnalyzedSignal:
    # class for processed intracellular data
    def __init__(self, f, num_peaks, output):
        self.sampling_f = f
        self.num_peaks = num_peaks
        self.time = output[0]
        self.data = output[1]
        self.std = output[2]
        self.pkloc = output[3]
        self.max_locs = output[4]
        self.min_locs = output[5]
        self.start_locs = output[6]
        self.zeroed_data = output[7]
        # initialize some holders
        self.spike_velocity = None
        self.spike_interval = None
        self.start_max_amp = None
        self.min_max_amp = None
        self.amp_duration = None #90 to 10
        self.APD100 = None
        self.APD90 = None
        self.APD50 = None
        self.APD10 = None
        self.cell_type = None

    def calculate_all(self):
        # populate the holders. returns nothing.
        n = self.num_peaks
        f = self.sampling_f
        sp_vel = np.zeros(n)
        sp_int = np.zeros(n)
        SM_amp = np.zeros(n)
        MM_amp = np.zeros(n)
        amp_dur = np.zeros(n)
        apd100 = np.zeros(n)
        apd90 = np.zeros(n)
        apd50 = np.zeros(n)
        apd10 = np.zeros(n)
      
        for i in range(n):
            start = self.pkloc[i]-2
            stop = self.pkloc[i]+2
            temp_time = self.time[start:stop]
            temp_data = self.zeroed_data[start:stop]
            start_loc = self.zeroed_data[self.start_locs[i]]
            poly = np.polyfit(temp_time, temp_data, 1)
            sp_vel[i] = poly[0]
            sp_int[i] = self.time[self.pkloc[i+1]] - self.time[self.pkloc[i]]
            SM_amp[i] = self.zeroed_data[self.max_locs[i]] - self.zeroed_data[self.start_locs[i]]
            MM_amp[i] = self.zeroed_data[self.max_locs[i]] - self.zeroed_data[self.min_locs[i]]
            
            cur_data = self.zeroed_data[self.start_locs[i]:self.min_locs[i]]
            apd100[i] = len(cur_data[cur_data>start_loc])/f
            apd90[i] = len(cur_data[cur_data>(start_loc + SM_amp[i]*0.1)])/f
            apd50[i] = len(cur_data[cur_data>(start_loc + SM_amp[i]*0.5)])/f
            apd10[i] = len(cur_data[cur_data>(start_loc + SM_amp[i]*0.9)])/f

            cur_data = self.zeroed_data[self.start_locs[i]:self.max_locs[i]]
            amp_dur[i] = len(cur_data[(cur_data>(start_loc + SM_amp[i]*0.2)) &
                                        (cur_data<(start_loc + SM_amp[i]*0.8))])/f

        self.spike_velocity = sp_vel
        self.spike_interval = sp_int
        self.start_max_amp = SM_amp
        self.min_max_amp = MM_amp
        self.amp_duration = amp_dur
        self.APD100 = apd100
        self.APD90 = apd90
        self.APD50 = apd50
        self.APD10 = apd10

        # determine cell type by APD50/APD90 ratio (0.5 and above ventricular)
        if np.mean(apd50)/np.mean(apd90) > 0.5:
            self.cell_type = 'ventricular'
        else:
            self.cell_type = 'atrial'

    def get_cell_type(self):
        return self.cell_type

    def get_APD_ratio(self):
        # gets apd50/apd90 ratio, only works after calculate_all is called
        apd50 = self.APD50
        apd90 = self.APD90
        return np.mean(apd50)/np.mean(apd90)

    def detect_arrythmia(self, threshold=0.05):
        # detects if cells are beating at an irregular rate
        # within the number of peaks analyzed
        # threshold specifies how different consecutive beats have to be
        # to qualify as arrythmia (in s)
        # only works after calculate_all has been called
        # returns a boolean
        is_arrythmic = False
        for i in range(2,len(self.spike_interval)-1):
            if (self.spike_interval[i]-self.spike_interval[i-1]) > threshold:
                is_arrythmic = True
        return is_arrythmic
            

    def plot(self, info="APDs"):
        # plots whatever information specified by "info"
        plt.figure(figsize=(8,5))
        if info == "APDs":
            plt.plot(self.APD90, 'b', label="APD90")
            plt.plot(self.APD50, 'm', label="APD50")
            plt.plot(self.APD10, 'r', label="APD10")
            plt.legend()
            plt.xlabel('Peaks')
            plt.ylabel('APD values')
        elif info == "APD100":
            plt.plot(self.APD100, label="APD100")
            plt.xlabel('Peaks')
            plt.ylabel('APD100')
        elif info == "APD90":
            plt.plot(self.APD90, label="APD90")
            plt.xlabel('Peaks')
            plt.ylabel('APD90')
        elif info == "APD50":
            plt.plot(self.APD50, label="APD50")
            plt.xlabel('Peaks')
            plt.ylabel('APD50')
        elif info == "APD10":
            plt.plot(self.APD10, label="APD10")
            plt.xlabel('Peaks')
            plt.ylabel('APD10')
        elif info == "Spike Velocity":
            plt.plot(self.spike_velocity, label="spike velocity")
            plt.xlabel('Peaks')
            plt.ylabel('Spike Velocity')
        elif info == "Spike Interval":
            plt.plot(self.spike_interval, label="spike interval")
            plt.xlabel('Peaks')
            plt.ylabel('Spike Interval/s')
        elif info == "SM_amp":
            plt.plot(self.start_max_amp, label="start to max amplitude")
            plt.xlabel('Peaks')
            plt.ylabel('Amplitude/mV')
        elif info == "MM_amp":
            plt.plot(self.min_max_amp, label="min to max amplitude")
            plt.xlabel('Peaks')
            plt.ylabel('Amplitude/mV')
        elif info == "Amp_Duration":
            plt.plot(self.amp_duration, label="Amplitude Duration")
            plt.xlabel('Peaks')
            plt.ylabel('Duration/s')
        else:
            print('Unrecognized type to plot. Try again.')
                       
        return

    def plot_data(self, show_original=True):
        # plots the data for sanity checking
        plt.figure(figsize=(8,5))
        if show_original:
            plt.plot(self.time, self.data, "b")
        plt.plot(self.time, self.zeroed_data, "g")
        plt.plot(self.time[self.max_locs], self.zeroed_data[self.max_locs], "r+")
        plt.plot(self.time[self.start_locs], self.zeroed_data[self.start_locs], "m+")
        plt.plot(self.time[self.min_locs], self.zeroed_data[self.min_locs], "c+")

def baseline_als(y, lam=10000000000000, p=0.015, niter=3):
    L = len(y)
    D = sparse.diags([1,-2,1],[0,-1,-2], shape=(L,L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z
