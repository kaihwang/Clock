import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
from autoreject import LocalAutoRejectCV
from autoreject import get_rejection_threshold
#from collections import defaultdict
import sys
#plt.ion()
#matplotlib.use('Qt4Agg')
#rm ~/.ICEauthority

#### scripts to use MNE to analyze Clock MEG data

def raw_to_epoch(subject, Event_types, autoreject = False):
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
				triggers = np.delete(triggers, -1, 0)  # delete the last row becuase for some runs final trial doesn't have a long enough ITI. UGH.
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
				epochs[event] = mne.Epochs(raw, events=triggers, event_id=Event_codes[event], 
					tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=picks, on_missing = 'ignore')
			else: #concat epochs
				epochs[event] = mne.concatenate_epochs((epochs[event], 
					mne.Epochs(raw, events=triggers, event_id=Event_codes[event], 
						tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=picks, on_missing = 'ignore')))	

	#run autoreject after compile			
	if autoreject:

		epochs_clean = dict.fromkeys(Event_types)
		for event in Event_types:
			ar = LocalAutoRejectCV()
			epochs_clean[event] = ar.fit_transform(epochs[event])
			reject = get_rejection_threshold(epochs[event])	
	
		return epochs_clean, reject	
	
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
	''' use morlet wavelet to cal trail by trial power
	for now can only return and save average across trials because of memory issues...
	'''
	freqs = np.logspace(*np.log10([2, 50]), num=20)
	n_cycles = freqs / 2.
	if average == True:
		n_jobs = 3
	else:
		n_jobs = 1
	power = mne.time_frequency.tfr_morlet(epochs[event], freqs=freqs, n_cycles=n_cycles, return_itc=False, n_jobs=n_jobs, average = average)

	return power


def indiv_subject_raw_to_tfr(subject):
	''' individual pipelines start to finish'''
	Event_types =['clock', 'feedback', 'ITI', 'RT']
	datapath = '/home/despoB/kaihwang/Clock/'

	##### create epoch object
	epochs = raw_to_epoch(subject, Event_types)
	for event in Event_types:
		fn = datapath + '%s/MEG/%s_%s-epo.fif' %(subject, subject, event)
		epochs[event].save(fn)

	##### plot examine and evoke responses 		
	#evoked = epoch_to_evoke(epochs,, Event_types, plot = False)

	#### do TFR
	for event in Event_types:
		# for now only output average to save space
		power = epoch_to_TFR(epochs, event, average = True)
		fn = datapath + '%s/MEG/%s_%s-avepower-tfr.h5' %(subject, subject, event)
		mne.time_frequency.write_tfrs(fn, power, overwrite = True)	


def group_average_evoke(subjects, event):
	'''average evoke response'''
	
	datapath = '/home/despoB/kaihwang/Clock/'

	subject = str(int(subjects[0]))
	fn = datapath + '%s/MEG/%s_%s-epo.fif' %(subject, subject, event)
	e = mne.read_epochs(fn)
	a = e.average()
	e_sum = np.zeros(a.data.shape) 

	for n, subject in enumerate(subjects):
		subject = str(int(subject))		
		fn = datapath + '%s/MEG/%s_%s-epo.fif' %(subject, subject, event)
		e = mne.read_epochs(fn)
		a = e.average() ### note, basline already applied during epoching
		e_sum = e_sum + a.data
	e_ave = a.copy()
	e_ave.nave = n+1
	e_ave.data = e_sum / (n+1)	

	return e_ave 	


def run_group_ave_evoke():
	save_path='/home/despoB/kaihwang/bin/Clock'
	Event_types =['clock', 'feedback', 'RT']
	subjects = np.loadtxt(save_path+'/TFR_subjects', dtype=int)

	for event in Event_types:
		aveEvoke = group_average_evoke(subjects, event)
		fn = save_path +'/Data/group_%s-evoke-ave.fif' %(event)
		mne.write_evokeds(fn, aveEvoke)	



def group_average_power(subjects, event, normalize_within_subject = False):
	''' load averaged TFR from subjects, average across subjects, and return group average TFR for plotting
	Event_types =['clock', 'feedback', 'ITI', 'RT']
	also need to set whether or not to normalize TFR within a subject
	'''
	datapath = '/home/despoB/kaihwang/Clock/'	
	
	#determine size
	subject = str(int(subjects[0]))
	fn = datapath + '%s/MEG/%s_%s-avepower-tfr.h5' %(subject, subject, event)
	pow = mne.time_frequency.read_tfrs(fn)
	pow_Sum = np.zeros(pow[0].data.shape) 

	
	#averaging
	for n, subject in enumerate(subjects):
		subject = str(int(subject))

		fn = datapath + '%s/MEG/%s_%s-avepower-tfr.h5' %(subject, subject, event)
		pow = mne.time_frequency.read_tfrs(fn)

		if normalize_within_subject:
			fn = datapath + '%s/MEG/%s_%s-avepower-tfr.h5' %(subject, subject, 'ITI')
			baseline = mne.time_frequency.read_tfrs(fn)
			baseline_mean = np.repeat(np.mean(baseline[0].data, axis = 2)[:, :, np.newaxis], pow[0].data.shape[2], axis=2) # mean across time
			baseline_sd = np.repeat(np.std(baseline[0].data, axis = 2)[:, :, np.newaxis], pow[0].data.shape[2], axis=2) # sd across time	
			pow_Sum = pow_Sum + ((pow[0].data - baseline_mean)/baseline_sd)
		else:
			pow_Sum = pow_Sum + pow[0].data

	pow_ave = pow[0].copy()
	pow_ave.nave = n+1
	pow_ave.data = pow_Sum / (n+1)
	
	return	pow_ave


def run_group_ave_power():
	save_path='/home/despoB/kaihwang/bin/Clock'
	Event_types =['clock', 'feedback', 'RT']
	subjects = np.loadtxt(save_path+'/TFR_subjects', dtype=int)

	for event in Event_types:
		power = group_average_power(subjects, event, normalize_within_subject = True)
		fn = save_path +'/Data/group_%s_power-tfr.h5' %(event)
		mne.time_frequency.write_tfrs(fn, power, overwrite = True)	




if __name__ == "__main__":	
	
	#### run indiv subject pipeline
	# subject = raw_input()
	# indiv_subject_raw_to_tfr(subject)

	#### test autorejct bad trials
	# datapath = '/home/despoB/kaihwang/Clock/'
	# subject = 10637
	# event='clock'
	# fn = datapath + '%s/MEG/%s_%s-epo.fif' %(subject, subject, event)
	# epochs = mne.read_epochs(fn)

	# from autoreject import LocalAutoRejectCV
	# ar = LocalAutoRejectCV()
	# epochs_clean = ar.fit_transform(epochs) 

	# from autoreject import get_rejection_threshold
	# reject = get_rejection_threshold(epochs)


	#### group average evoke response
	#run_group_ave_evoke()

	#### group averaged TFR power
	# run_group_ave_power()



#to access data: power[event].data
#to access time: power[event].time
#to access fre : power[event].freqs
#to accesss ave: power[event].nave








