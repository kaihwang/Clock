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
from scipy import io
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle

#plt.ion()
#matplotlib.use('Qt4Agg')
#rm ~/.ICEauthority

#### scripts to use MNE to analyze Clock MEG data

def raw_to_epoch(subject, Event_types, channels_list = None, autoreject = False):
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
		
		try:
			fn = datapath + '%s/MEG/%s_clock_run%s_dn_ds_sss_raw.fif' %(subject, subject, r)
			raw = mne.io.read_raw_fif(fn)		
			picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
		except:
			break # skip all if fif file does not exist

		
		for event in Event_types:

			#adjust evevnts:
			# RT can be calculated by 300ms prior to feedback onset
			# ITI is 850 ms after feedback
			if event == 'ITI':
				try:
					fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
					triggers = mne.read_events(glob.glob(fn)[0])
					triggers[1:,0] = triggers[1:,0] +213 #shift 850ms
					triggers = np.delete(triggers, -1, 0)  # delete the last row becuase for some runs final trial doesn't have a long enough ITI. UGH.
					baseline = None
				except:
					triggers = None

			elif event == 'RT':
				try:
					fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
					triggers = mne.read_events(glob.glob(fn)[0])
					triggers[1:,0] = triggers[1:,0] - 75 #shift 300ms
					baseline = (None, 0.0)
				except:
					triggers = None

			else:
				try:	
					fn = datapath + '%s/MEG/MEG_%s_*_%s_%s_ds4.eve' %(subject, subject, r, event)
					triggers = mne.read_events(glob.glob(fn)[0])
					baseline = (None, 0.0)
				except:
					triggers = None

			if r == 1: #create epoch with first run	
				if triggers is not None:

					try:
						epochs[event] = mne.Epochs(raw, events=triggers, event_id=Event_codes[event], 
							tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=channels_list, on_missing = 'ignore')
					except:
						pass #if fif file exist but fail for whatev reason
				else:
					pass
			
			else: #concat epochs
				if triggers is not None:

					try:
						epochs[event] = mne.concatenate_epochs((epochs[event], 
							mne.Epochs(raw, events=triggers, event_id=Event_codes[event], 
							tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = baseline, picks=channels_list, on_missing = 'ignore')))	
					except:
						pass
				else:
					pass

	#run autoreject after compile			
	if autoreject:

		epochs_clean = dict.fromkeys(Event_types)
		for event in Event_types:
			ar = LocalAutoRejectCV()
			epochs_clean[event] = ar.fit_transform(epochs[event])
			reject = get_rejection_threshold(epochs[event])	
	
		return epochs_clean, reject	
	
	return epochs	


def get_dropped_trials_list(epoch):
	'''mne_read_epoch will automatically drop trials that are too short without warning, so need to retrieve those tiral indx...'''

	drop_log = epoch[epoch.keys()[0]].drop_log
	trial_list = []

	# event lists start with "0 0 0", get rid of those
	for n in range(0,len(drop_log)):
		if drop_log[n] == ['IGNORED']:
			trial_list.append(n)

	for index in sorted(trial_list, reverse=True):		
		del drop_log[index]

	drop_list = []	
	for n in range(0,len(drop_log)):	
		if drop_log[n]!=[]:  #get list of trials dropped for whatever reason
			 drop_list.append(n)
	
	drop_list = np.array(drop_list)		 
	return drop_list	 


def	epoch_to_evoke(epochs, Event_types, plot = False):
	#Event_types =['clock', 'feedback', 'ITI', 'RT']
	evoked = {}
	for event in Event_types:
		evoked[event] = epochs[event].average()

		if plot == True:
			evoked[event].plot()
			mne.viz.plot_evoked_topo(evoked[event])	

	return evoked		


def epoch_to_TFR(epochs, event, freqs = None, average = True):
	''' use morlet wavelet to compute trail by trial power
	for now can only return and save average across trials because of memory restriction...
	'''
	
	if freqs == None: #do full spec if not specified
		freqs = np.logspace(*np.log10([2, 50]), num=20)
	
	n_cycles = freqs / 2.
	if average == True:
		n_jobs = 3
	else:
		n_jobs = 1
	try:	
		power = mne.time_frequency.tfr_morlet(epochs[event], freqs=freqs, n_cycles=n_cycles, return_itc=False, n_jobs=n_jobs, average = average)
	except:
		power = mne.time_frequency.tfr_morlet(epochs[event], freqs=[freqs], n_cycles=n_cycles, return_itc=False, n_jobs=n_jobs, average = average)
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


def save_object(obj, filename):
	''' Simple function to write out objects into a pickle file
	usage: save_object(obj, filename)
	'''
	with open(filename, 'wb') as output:
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
	#M = pickle.load(open(f, "rb"))	


def TFR_regression(chname, freqs, Event_types):

	subjects = [10637, 10638, 10662, 10711] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#Event_types='clock'
	channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	#chname = 'MEG0713'
	pick_ch = mne.pick_channels(channels_list.tolist(),[chname])
	#mne.set_log_level('WARNING')

	Data = dict()
	for s, subject in enumerate(subjects):
		
		# create epoch with one channel of data
		e = raw_to_epoch(subject, [Event_types], channels_list = pick_ch, autoreject = False)
		b = raw_to_epoch(subject, ['ITI'], channels_list = pick_ch, autoreject = False) #ITI baseline


		if e[Event_types]==None:
			break # skip subjects that have no fif files (need to check with Will on why?)

		#get list of trials dropped
		drops = get_dropped_trials_list(e)

		# get trial by trial TFR
		#event = 'clock'
		TFR = epoch_to_TFR(e, Event_types, freqs, average = False)
		BaselineTFR = epoch_to_TFR(b, 'ITI', freqs, average = False)

		#TFR.data dimension is trial x channel x freq x time

		#baseline correction
		#TFR.apply_baseline((-1,-.2), mode='zscore')
		baseline_power = np.broadcast_to(np.mean(np.mean(BaselineTFR.data,axis=3),axis=0),TFR.data.shape)
		TFR.data = 100*((TFR.data - baseline_power) / baseline_power) #convert to percent of signal change

		#get model parameters
		fn = "/home/despoB/kaihwang/Clock_behav/%s_pe.mat" %(subject)
		pe = io.loadmat(fn)
		fn = "/home/despoB/kaihwang/Clock_behav/%s_value.mat" %(subject)
		value = io.loadmat(fn)
		
		pe = np.delete(pe['pe'],drops, axis=0)
		pe = np.max(pe,axis=1) #for prediction error take the max
		value = np.delete(value['value'],drops, axis=0)


		#create dataframe
		for f, freq in enumerate(TFR.freqs):
			for t, time in enumerate(TFR.times[250:]):  #start from time 0, which is indx 250
				tidx = t+250

				pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Pow', 'Pe')) 
				pdf.loc[:,'Pow'] = TFR.data[:,:,f,tidx].squeeze()
				pdf.loc[:,'Trial'] = np.arange(TFR.data[:,:,f,tidx].shape[0])+1
				pdf.loc[:,'Subject'] = str(subject)
				pdf.loc[:,'Pe'] = pe
				pdf['Pe'].subtract(pdf['Pe'].mean()) #mean center PE
				
				pdf['Trial'] = pdf['Trial'].astype('category')
				pdf['Subject'] = pdf['Subject'].astype('category')

				if s ==0: #first subject
					Data[(freq,time)] = pdf
				else:
					Data[(freq,time)] = pd.concat([Data[(freq,time)], pdf])
	
	#fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_clock' + '_tfr'
	#save_object(Data, fn)		


	### plot to look at distribution
	#%matplotlib qt
	#g=sns.jointplot('Pow','Pe',data=D])   

	## regression
	RegStats = dict()
	for freq in TFR.freqs:
		for time in TFR.times[250:]:
			Data[(freq,time)] = Data[(freq,time)].dropna()
			
			#for some reason getting inf, get rid of outliers, and look into this later
			Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pow']<200] 
			
			md = smf.mixedlm("Pow ~ Pe + Trial", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Pe+Trial")
			RegStats[(chname, freq, time)] = md.fit()
			#print(RegStats[(chname, freq, time)].summary())

			## need to extract parameters 
			## save into mat

	fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'		
	save_object(RegStats, fn)		



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

	##### test if there are issues with fif and eve files
	subject = 10637#np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjlist', dtype=int)	 #[10637, 10638, 10662, 10711]
	Event_types=['clock']
	channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	# chname = 'MEG0713'
	# pick_ch = mne.pick_channels(channels_list.tolist(),[chname])
	# #mne.set_log_level('WARNING')

	# Data = dict()
	# for s, subject in enumerate(subjects):
	# 	# create epoch with one channel of data
	e = raw_to_epoch(subject, Event_types, autoreject = False)

	#### test single trial TFR conversion

	# fullfreqs = np.logspace(*np.log10([2, 50]), num=20)
	# freqs=fullfreqs[0]
	# chname = 'MEG0713'
	# Event_types = 'feedback'
	# TFR_regression(chname, freqs, Event_types)
	# Event_types = 'clock'
	# TFR_regression(chname, freqs, Event_types)


