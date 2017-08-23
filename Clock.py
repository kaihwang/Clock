import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mne as mne
import os.path as op


#### examples to consult
# MNE read events
# https://martinos.org/mne/stable/generated/mne.read_events.html#mne.read_events
# time frequency
# https://martinos.org/mne/stable/auto_examples/time_frequency/plot_time_frequency_simulated.html#sphx-glr-auto-examples-time-frequency-plot-time-frequency-simulated-py
# https://martinos.org/mne/stable/auto_tutorials/plot_stats_cluster_time_frequency.html
# https://martinos.org/mne/stable/auto_tutorials/plot_sensors_time_frequency.html
# Will’s timing script
# https://github.com/LabNeuroCogDevel/EmoClock.py/blob/master/timing.py

## read raw
raw = mne.io.read_raw_fif('/home/despoB/kaihwang/Clock/10772/MEG/10772_clock_run6_dn_ds_sss_raw.fif')


## examine raw
order = np.arange(raw.info['nchan'])
order[9] = 310  # We exchange the plotting order of two channels
order[310] = 9  # to show the trigger channel as the 10th channel.
raw.plot(n_channels=10, order=order, block=True)


##epoch
events = mne.read_events('/home/despoB/kaihwang/Clock/10772/MEG/MEG_10772_20140509_6_clock_ds4.eve')
raw.plot(events=events, n_channels=10, order=order)

tmin, tmax = -0.2, 0.5
event_id = {'Clock': 110}
# Only pick MEG and EOG channels.
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=True)
baseline = (None, 0.0)
reject = {'mag': 4e-12, 'eog': 200e-6}
epochs = mne.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, reject=reject, picks=picks)
epochs.plot(block=True)


##average
picks = mne.pick_types(epochs.info, meg=True, eog=True)
evoked_clock = epochs['Clock'].average(picks=picks)
evoked_clock.plot() 



##TFR
epochs.plot_psd_topomap(ch_type='grad', normalize=True)