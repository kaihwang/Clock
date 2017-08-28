import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
from collections import defaultdict
#matplotlib.use('Qt4Agg')
#rm ~/.ICEauthority

#### scripts to use MNE to analyze Clock MEG data

#### examples to consult

# from start to finish
# https://martinos.org/mne/stable/auto_tutorials/plot_introduction.html
	
# MNE read events
# https://martinos.org/mne/stable/generated/mne.read_events.html#mne.read_events

# time frequency
# https://martinos.org/mne/stable/auto_examples/time_frequency/plot_time_frequency_simulated.html#sphx-glr-auto-examples-time-frequency-plot-time-frequency-simulated-py
# https://martinos.org/mne/stable/auto_tutorials/plot_stats_cluster_time_frequency.html
# https://martinos.org/mne/stable/auto_tutorials/plot_sensors_time_frequency.html

# Will’s timing script
# https://github.com/LabNeuroCogDevel/EmoClock.py/blob/master/timing.py

# epoching
# https://martinos.org/mne/stable/generated/mne.concatenate_epochs.html?highlight=concatenate#mne.concatenate_epochs

# artifact rejection
# http://autoreject.github.io/

## read raw

def raw_to_epoch(subject):
	'''short hand to load raw fif across runs, and return a combined epoch object'''
	
	#setup variables
	datapath = '/home/despoB/kaihwang/Clock/'
	
	Event_types =['clock', 'feedback', 'ITI', 'RT']

	Event_codes = {
	'clock' : { 
	'DEV.fear.face'   : 47,
	'IEV.fear.face'   : 56,
	'DEV.happy.face'  : 83,
	'IEV.happy.face'  : 92,
	'CEV.scram.face'  : 101,
	'CEVR.scram.face' : 110,
	'DEV.scram.face'  : 119,
	'IEV.scram.face'  : 128,
	},
	'feedback' : {
	'DEV.fear.score'  : 154,
	'IEV.fear.score'  : 163,
	'DEV.happy.score' : 190,
	'IEV.happy.score' : 199,
	'CEV.scram.score' : 208,
	'CEVR.scram.score': 217,
	'DEV.scram.score' : 226,
	'IEV.scram.score' : 235,
	},
	'ITI' : {
	'DEV.fear.score'  : 154,
	'IEV.fear.score'  : 163,
	'DEV.happy.score' : 190,
	'IEV.happy.score' : 199,
	'CEV.scram.score' : 208,
	'CEVR.scram.score': 217,
	'DEV.scram.score' : 226,
	'IEV.scram.score' : 235,
	},
	'RT' : {
	'DEV.fear.score'  : 154,
	'IEV.fear.score'  : 163,
	'DEV.happy.score' : 190,
	'IEV.happy.score' : 199,
	'CEV.scram.score' : 208,
	'CEVR.scram.score': 217,
	'DEV.scram.score' : 226,
	'IEV.scram.score' : 235,
	}}



	Epoch_timings = {
	'clock'   : [-1,4],
	'feedback': [-1,.850],
	'ITI'     : [0,1],  #no baseline for ITI
	'RT': [-1,1.15],
	}

	epochs = dict.fromkeys(Event_types)

	for r in np.arange(1,9):
		
		fn = datapath + '%s/MEG/%s_clock_run%s_dn_ds_sss_raw.fif' %(subject, subject, r)
		raw = mne.io.read_raw_fif(fn)		
		picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
			
		for event in Event_types:

			#adjust evevnts:
			# RT can be calculated by 300ms prior to feedback onset
			# ITI is 850 ms after feedback
			if event == 'ITI':
				fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
				triggers = mne.read_events(glob.glob(fn)[0])
				triggers[1:,0] = triggers[1:,0] +213 #shift 850ms
				baseline = None

			elif event == 'RT':
				fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
				triggers = mne.read_events(glob.glob(fn)[0])
				triggers[1:,0] = triggers[1:,0] - 75 #shift 300ms
				baseline = (None, 0.0)

			else:
				fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, event)
				triggers = mne.read_events(glob.glob(fn)[0])
				baseline = (None, 0.0)

			if r == 1: #create epoch with first run	
				#epochs[event].update({r: mne.Epochs(raw, events=triggers, event_id=Event_codes[event], tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=picks, on_missing = 'ignore')})
				epochs[event] = mne.Epochs(raw, events=triggers, event_id=Event_codes[event], tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=picks, on_missing = 'ignore')
			else: #concat epochs
				epochs[event] = mne.concatenate_epochs((epochs[event], 
					mne.Epochs(raw, events=triggers, event_id=Event_codes[event], tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=picks, on_missing = 'ignore')))
	

	return epochs		
		
epochs = raw_to_epoch(11288)
		
		
			





## examine raw
# order = np.arange(raw.info['nchan'])
# order[9] = 310  # We exchange the plotting order of two channels
# order[310] = 9  # to show the trigger channel as the 10th channel.
# raw.plot(n_channels=10, order=order, block=True)

# Only pick MEG and EOG channels.
#reject = {'mag': 4e-12, 'eog': 200e-6}
#tmin, tmax = -0.2, 0.5
# epochs.plot(block=True)


##average
picks = mne.pick_types(epochs.info, meg=True, eog=True)
evoked_clock = epochs['CEV.scram.face'].average(picks=picks)
evoked_clock.plot() 



##TFR
epochs.plot_psd_topomap(ch_type='grad', normalize=True)








