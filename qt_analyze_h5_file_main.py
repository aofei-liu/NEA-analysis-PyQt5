from PyQt5.QtWidgets import *
from analyze_h5_file_ui import Ui_H5Analyzer

import numpy as np
import pandas as pd
import random
import sys
import h5utils
import os
from signal_classes import AnalyzedSignal
from detect_peaks import detect_peaks

class analyzeH5File(QMainWindow, Ui_H5Analyzer):
    def __init__(self):
        super(analyzeH5File, self).__init__()

        # setup ui from designer
        self.setupUi(self)
        self.setWindowTitle("H5 File Analyzer")

        # define buttons
        self.openFileButton.clicked.connect(self.open_file)
        self.getInterestingChannelsButton.clicked.connect(self.get_interesting_channels)
        self.plotChannelButton.clicked.connect(self.plot_channel)
        self.analyzeChannelButton.clicked.connect(self.analyze_channel)
        self.plotChoiceButton.clicked.connect(self.plot_analysis)
        self.saveDataButton.clicked.connect(self.save_data)
        self.saveAnalyzedDataButton.clicked.connect(self.save_analyzed_data)

        self.analysisProgressBar.setValue(0)

        # set up fields to contain data
        self.raw_data = None
        self.channel_labels = None
        self.channel_ids = None
        self.time = None
        self.sampling_f = None
        self.interesting_channels_std = {}

        self.current_channel = None
        self.current_analyzed = None
        

    def open_file(self):
        name, filetype = QFileDialog.getOpenFileName(self, "Select .h5 file", os.getcwd(), "H5 files (*.h5)")
        self.fileLoadDisplay.setText("Loading...")
        data, labels, time, f = h5utils.load_h5(name)
        
        self.sampling_f = f
        self.time = time
        self.raw_data = data
        self.channel_labels = labels
        self.channel_ids = np.array(range(len(labels)))
        self.interesting_channels_std = {}
        
        self.fileLoadDisplay.setText("File Loaded!")
        namesplit = name.split("/")
        self.FileNameDisplay.setText(namesplit[-1])

    def get_interesting_channels(self):
        window_size = int(self.windowInput.text())
        max_recording_thresh = int(self.maxRecordingThreshInput.text())
        self.completed = 0
        self.analysisProgressBar.setMaximum(len(self.channel_ids))
        for i in self.channel_ids:
            processed = h5utils.is_interesting_crossing(self.raw_data[i], self.sampling_f,
                                                        max_thresh = max_recording_thresh,
                                                        w = window_size)
            if processed[0] == True:
                # save the std values; everything else isn't really needed
                # in this application
                self.interesting_channels_std[i] = processed[3]
            self.completed += 1
            self.analysisProgressBar.setValue(self.completed)
        
        self.interestingChannelDisplay.setText(str([*self.interesting_channels_std.keys()]))

    def analyze_channel(self):
        #clear analyzed signal
        self.current_analyzed = None
        #get params
        window_size = int(self.windowInput.text())
        channel_id = self.current_channel
        if channel_id not in self.channel_ids:
            return
        min_pk_sep = float(self.minPeakSeparationInput.text())
        min_std_thresh = float(self.minStdThreshInput.text())

        # get area of interest and analyze
        self.plotAnalyzedWidget.canvas.figure.clear()
        ax = self.plotAnalyzedWidget.canvas.figure.gca()
        ax_selected = self.plotChannelWidget.canvas.figure.gca()
        xlim = ax_selected.get_xlim()
        start_array = max(1, int(xlim[0]*self.sampling_f))
        end_array = min(len(self.time), int(xlim[1]*self.sampling_f))

        time_selected = self.time[start_array:end_array]
        data_selected = h5utils.smooth_data(self.raw_data[channel_id][start_array:end_array], window_size)
        if channel_id not in self.interesting_channels_std.keys():
            std_selected = h5utils.rolling_std(data_selected, window_size)
        else:
            std_selected = self.interesting_channels_std[channel_id][start_array:end_array]
        analyzed = _analyzeSignal(time_selected, data_selected, std_selected,
                                  min_pk_sep, min_std_thresh, self.sampling_f)
        self.current_analyzed = analyzed

        # plot
        ax.plot(analyzed.time, analyzed.data, "b")
        ax.plot(analyzed.time[analyzed.max_locs], analyzed.data[analyzed.max_locs], "r+")
        ax.plot(analyzed.time[analyzed.start_locs], analyzed.data[analyzed.start_locs], "m+")
        ax.plot(analyzed.time[analyzed.min_locs], analyzed.data[analyzed.min_locs], "c+")
        self.plotAnalyzedWidget.canvas.draw()

        # determine APD50/90 ratio and predict cell type
        ratio = analyzed.get_APD_ratio()
        cell_type = str(analyzed.get_cell_type())
        toDisplay = 'Cell type: ' + cell_type + ', APD50/90: %.3f' % ratio
        self.cellTypeDisplay.setText(toDisplay)
    
    def plot_channel(self):
        self.current_channel = None
        channel_id = int(self.channelSelectInput.text())
        if channel_id not in self.channel_ids:
            return
        self.current_channel = channel_id
        self.plotChannelWidget.canvas.figure.clear()
        ax = self.plotChannelWidget.canvas.figure.gca()
        ax.plot(self.time, self.raw_data[channel_id])
        if channel_id in self.interesting_channels_std.keys():
            ax.plot(self.time, self.interesting_channels_std[channel_id])
        self.plotChannelWidget.canvas.draw()
            

    def plot_analysis(self):
        sel = self.plotChoiceBox.currentText()
        analyzed = self.current_analyzed
        if self.current_analyzed is None:
            return
        self.plotParametersWidget.canvas.figure.clear()
        ax = self.plotParametersWidget.canvas.figure.add_subplot(111)
        if sel == "Start to Max Amplitude":
            ax.plot(analyzed.start_max_amp, label="start to max amplitude")
            ax.set_xlabel('Peaks')
            ax.set_ylabel('Amplitude/uV')
        elif sel == "Min to Max Amplitude":
            ax.plot(analyzed.min_max_amp, label="min to max amplitude")
            ax.set_xlabel('Peaks')
            ax.set_ylabel('Amplitude/uV')
        elif sel == "APDs":
            ax.plot(analyzed.APD90, 'b', label='APD90')
            ax.plot(analyzed.APD50, 'm', label='APD50')
            ax.plot(analyzed.APD10, 'r', label='APD10')
            ax.legend()
            ax.set_xlabel('Peaks')
            ax.set_ylabel('APD values')
        elif sel == "Spike Velocity":
            ax.plot(analyzed.spike_velocity, label="spike_velocity")
            ax.set_xlabel('Peaks')
            ax.set_ylabel('Spike Velocity')
        elif sel == "Spike Duration":
            ax.plot(analyzed.amp_duration, label="amp_duration")
            ax.set_xlabel('Peaks')
            ax.set_ylabel('Spike Duration')
        elif sel == "Time Interval":
            ax.plot(analyzed.spike_interval, label="spike_interval")
            ax.set_xlabel('Peaks')
            ax.set_ylabel('Time Interval')
        else:
            return
        self.plotParametersWidget.canvas.draw()

    def save_data(self):
        channel_id = self.current_channel
        if channel_id not in self.channel_ids:
            return
        name, filetype = QFileDialog.getSaveFileName(self, "Select Save File Directory",
                                                     os.getcwd(), "txt file (*.txt)")
        ax_selected = self.plotChannelWidget.canvas.figure.gca()
        xlim = ax_selected.get_xlim()
        start_array = max(1, int(xlim[0]*self.sampling_f))
        end_array = min(len(self.time), int(xlim[1]*self.sampling_f))
        time_selected = self.time[start_array:end_array]
        data_selected = self.raw_data[channel_id][start_array:end_array]
        df = pd.DataFrame({'time':time_selected, 'data':data_selected})
        df.to_csv(name)

    def save_analyzed_data(self):
        analyzed = self.current_analyzed
        if analyzed is None:
            return
        

def _analyzeSignal(time_select, data_select, std_select, min_pk_sep, min_std_thresh, sampling_f):
    pkloc_select = detect_peaks(std_select, mph = min_std_thresh, mpd = min_pk_sep*sampling_f)
    num_peaks = len(pkloc_select)-1
    std_med = np.nanmedian(std_select)
    std_select_sorted = np.sort(std_select)
    stdofstd = np.std(std_select_sorted[0:int(len(std_select_sorted)*0.6)])
    # initialize some arrays
    max_locs = np.zeros(num_peaks, dtype=int)
    min_locs = np.zeros(num_peaks, dtype=int)
    start_locs = np.zeros(num_peaks, dtype=int)
    for i in range(num_peaks):
        cur_data = data_select[pkloc_select[i]:pkloc_select[i+1]]
        max_locs[i] = np.argmax(cur_data) + pkloc_select[i] - 1
        min_locs[i] = np.argmin(cur_data) + pkloc_select[i] - 1
        temp = pkloc_select[i]
        while std_select[temp]>(std_med + 5*stdofstd):
            if temp == 0:
                break
            temp -= 1
        start_locs[i] = temp
    output = (time_select, data_select, std_select, pkloc_select,
              max_locs, min_locs, start_locs, data_select)
    analyzed = AnalyzedSignal(sampling_f, num_peaks, output)
    analyzed.calculate_all()
    return analyzed       

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = analyzeH5File()
    window.show()
    sys.exit(app.exec_())
