import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
#from collections import defaultdict
import sys
#plt.ion()
#matplotlib.use('Qt4Agg')
#rm ~/.ICEauthority

#### scripts to use MNE to analyze Clock MEG data

def raw_to_epoch(subject, Event_types):
	'''short hand to load raw fif across runs, and return a combined epoch object
	input: subject, Event_types
	Event_types is a list of strings to indicate the event type you want to extract from raw. possible choices are:
	Event_types = ['clock', 'feedback', 'ITI', 'RT']

	The return of this function will be a dictionary, where key is the event_type, and the item is the mne epoch object.
	'''
	
	#setup variables
	datapath = '/home/despoB/kaihwang/Clock/'
	
	#Event_types = ['clock', 'feedback', 'ITI', 'RT']

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


def	epoch_to_evoke(epochs, Event_types, plot = False):
	#Event_types =['clock', 'feedback', 'ITI', 'RT']
	evoked = {}
	for event in Event_types:
		evoked[event] = epochs[event].average()

		if plot == True:
			evoked[event].plot()
			mne.viz.plot_evoked_topo(evoked[event])	
	return evoked		


def epoch_to_TFR(epochs, event, average = True):
	freqs = np.logspace(*np.log10([2, 50]), num=20)
	n_cycles = freqs / 2.
	if average == True:
		n_jobs = 3
	else:
		n_jobs = 1
	power = mne.time_frequency.tfr_morlet(epochs[event], freqs=freqs, n_cycles=n_cycles, return_itc=False, n_jobs=n_jobs, average = average)

	return power


if __name__ == "__main__":	
	
	##### variables
	subject = raw_input()
	Event_types =['clock', 'feedback', 'ITI', 'RT']
	datapath = '/home/despoB/kaihwang/Clock/'

	##### create epoch object
	epochs = raw_to_epoch(subject, Event_types)
	for event in Event_types:
		fn = datapath + '%s/MEG/%s_%s-epo.fif.gz' %(subject, subject, event)
		epochs[event].save(fn)

	##### plot examine and evoke responses 		
	#evoked = epoch_to_evoke(epochs,, Event_types, plot = False)

	#### do TFR
	power = {}
	for event in Event_types:
		# for now only output average to save space
		power[event] = epoch_to_TFR(epochs, event, average = True)
		fn = datapath + '%s/MEG/%s_%s-avepower-epo.fif.gz' %(subject, subject, event)
		power[event].save(fn)

	#### visualize	
	#power.plot_topomap(ch_type='grad', tmin=0.2, tmax=0.5, fmin=4, fmax=8,
    #               baseline=(-0.5, 0), mode='logratio',
    #                title='Beta', vmax=0.45, show=True)



## examine raw
# order = np.arange(raw.info['nchan'])
# order[9] = 310  # We exchange the plotting order of two channels
# order[310] = 9  # to show the trigger channel as the 10th channel.
# raw.plot(n_channels=10, order=order, block=True)

# Only pick MEG and EOG channels.
#reject = {'mag': 4e-12, 'eog': 200e-6}
#tmin, tmax = -0.2, 0.5
# epochs.plot(block=True)








