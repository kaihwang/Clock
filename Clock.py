#### scripts to use MNE to analyze Clock MEG data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
from functools import partial 
from autoreject import (LocalAutoRejectCV, compute_thresholds, set_matplotlib_defaults)
from mne.utils import check_random_state 
#from collections import defaultdict
import sys
from scipy import io
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
from scipy import signal
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats.mstats import zscore

def raw_to_epoch(subject, Event_types, channels_list = None):
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
		if drop_log[n]!=[]:  #get list of trials dropped for whatever reason, note rejected bad trials will also be here.
			 drop_list.append(n)
	
	drop_list = np.array(drop_list)		 
	return drop_list	 


def get_epoch_trial_types(epoch):
	''' get the order of face conditions'''
	trig_codes = epoch[epoch.keys()[0]].events[:,2]
	event = epoch.keys()[0]

	Face_codes = {
	'clock' : { 
	'fear.face'   : np.array([47, 56]),
	'happy.face'  : np.array([83, 92]),
	'scram.face' : np.array([101, 110, 119, 128])
	},

	'feedback' : { 
	'fear.face'   : np.array([154, 163]),
	'happy.face'  : np.array([190, 199]),
	'scram.face' : np.array([208, 217, 226, 235])
	},

	'RT' : {
	'fear.face'   : np.array([154, 163]),
	'happy.face'  : np.array([190, 199]),
	'scram.face' : np.array([208, 217, 226, 235])
	}}

	out = ['NaN'] * len(trig_codes)
	for i, trig in enumerate(trig_codes):
		if any(trig == Face_codes[epoch.keys()[0]]['fear.face']):
			out[i] = 'Fear'
		elif any(trig == Face_codes[epoch.keys()[0]]['happy.face']):	
			out[i] = 'Happy'
		elif any(trig == Face_codes[epoch.keys()[0]]['scram.face']):	
			out[i] = 'ASramble'
		else:
			pass
			

	# Epoch_timings = {
	# 'clock'   : [-1,4],
	# 'feedback': [-1,.850],
	# 'ITI'     : [0,1],  #no baseline for ITI
	# 'RT': [-1,1.15],
	# }
	return out


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
	''' individual pipeline start to finish'''
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
	'''average evoked responses'''
	
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


def read_object(filename):
	''' short hand for reading object because I can never remember pickle syntax'''
	o = pickle.load(open(filename, "rb"))	
	return o


def get_epochs_for_TFR_regression(chname, Event_types):
	''' get epoch and baseline epoch for TFR_regression
	return trial epoch, baseline epoch, and list of drop trials
	trying to speed up TFR_regression loop without the need of keep reading in data from disk
	Will output trial epoch, baseline epoch, and list of bad trilas as dict (where subject ID is key) '''

	#subjects = [10637, 10638] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#subjects = [11253]
	subjects = np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)
	#subjects =[11253]
	channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	pick_ch = mne.pick_channels(channels_list.tolist(),[chname])

	Event_Epoch = {}
	Baseline_Epoch = {}
	BadTrial_Lists = {}

	for s, subject in enumerate(subjects):

		#get bad trial and bad channel info
		try:
			bad_channels, bad_trials = get_bad_channels_and_trials(subject, Event_types, 0.3) #reject if thirty percent of data segment is bad
		except: #no ar 
			bad_channels = np.array([], dtype='<U7')
			bad_trials = np.array([])

		if any(chname == bad_channels):
			continue #skip if bad channel 

		try:
			baseline_bad_channels, baseline_bad_trials = get_bad_channels_and_trials(subject, 'ITI', 0.3) #reject if thirty percent of data segment is bad
		except: #no ar 
			baseline_bad_channels = np.array([], dtype='<U7')
			baseline_bad_trials = np.array([])

		if any(chname == baseline_bad_channels):
			continue #skip if bad channel 	

		# create epoch with one channel of data
		e = raw_to_epoch(subject, [Event_types], channels_list = pick_ch)
		b = raw_to_epoch(subject, ['ITI'], channels_list = pick_ch) #ITI baseline

		if e[Event_types]==None:
			continue

		#drop bad trials
		e[Event_types].drop(bad_trials)	
		b['ITI'].drop(baseline_bad_trials)	
		#get list of trials dropped
		drops = get_dropped_trials_list(e)

		Event_Epoch[subject] = e
		Baseline_Epoch[subject] = b
		BadTrial_Lists[subject] = drops

	return Event_Epoch, Baseline_Epoch, BadTrial_Lists		



def TFR_regression(Event_Epoch, Baseline_Epoch, chname, freqs, Event_types, do_reg = True, parameters ='Pe'):
	''' compile TFR dataframe and model params for regression.
	Need output from get_epochs_for_TFR_regression() as input.
	Event_Epoch is a dict with each subjects trial epoch, Baseline_Epoch is the baseline epoch period.
	Will do one channel a time (chname), and read model parameters from Michael's matlab output
	'''

	#subjects = [11345, 11346, 11347] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#subjects = np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)
	
	subjects = Event_Epoch.keys()
	#Event_types='clock'
	#channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	#chname = 'MEG0713'
	#pick_ch = mne.pick_channels(channels_list.tolist(),[chname])
	
	#mne.set_log_level('WARNING')
	demographic = pd.read_table('/home/despoB/kaihwang/bin/Clock/subinfo_db')

	Data = dict()
	for s, subject in enumerate(subjects):
		
		#get subject's age
		try: 
			age = demographic[demographic['lunaid']==subject]['age'].values[0]
		except:
			age = np.nan # no age info...??? 

		## check if skip because of bad channels
		try:
			bad_channels, _ = get_bad_channels_and_trials(subject, Event_types, 0.3) #reject if thirty percent of data segment is bad
		except: #no ar 
			bad_channels = np.array([], dtype='<U7')
			#bad_trials = np.array([])

		if any(chname == bad_channels):
			continue #skip if bad channel 

		try:
			baseline_bad_channels, _ = get_bad_channels_and_trials(subject, 'ITI', 0.3) #reject if thirty percent of data segment is bad
		except: #no autoreject record 
			baseline_bad_channels = np.array([], dtype='<U7')
			#baseline_bad_trials = np.array([])

		if any(chname == baseline_bad_channels):
			continue #skip if bad channel 	

		# create epoch with one channel of data
		#e = raw_to_epoch(subject, [Event_types], channels_list = pick_ch)
		#b = raw_to_epoch(subject, ['ITI'], channels_list = pick_ch) #ITI baseline

		#if e[Event_types]==None:
		#	continue # skip subjects that have no fif files (need to check with Will on why?)

		#drop bad trials
		#e[Event_types].drop(bad_trials)	
		#b['ITI'].drop(baseline_bad_trials)	
		#get list of trials dropped
		#drops = get_dropped_trials_list(e)
		
		e = Event_Epoch[subject]
		b = Baseline_Epoch[subject]
		drops = get_dropped_trials_list(e)

		# get list of emo face conditions
		#faces = np.delete(get_epoch_trial_types(e), drops, axis = 0)
		faces = get_epoch_trial_types(e)

		# get trial by trial TFR
		#event = 'clock'
		TFR = epoch_to_TFR(e, Event_types, freqs, average = False)
		BaselineTFR = epoch_to_TFR(b, 'ITI', freqs, average = False)

		## baseline correction
		#TFR.apply_baseline((-1,-.2), mode='zscore')
		baseline_power = np.broadcast_to(np.mean(np.mean(BaselineTFR.data,axis=3),axis=0),TFR.data.shape) #ave across time and trial, then broadcast to the right dimension
		#baseline_power_sd = np.broadcast_to(np.std(BaselineTFR.data.flatten()), TFR.data.shape)
		TFR.data = 100*((TFR.data - baseline_power) / baseline_power) #convert to percent of signal change
		#TFR.data = (TFR.data - baseline_power) / baseline_power_sd #convert to zscore
		times = TFR.times

		## extract model parameters and freq poer into dataframe
		##in the case of testing for PE
		if (parameters =='Pe') & (Event_types == 'feedback'):
			#get PE model parameters
			fn = "/home/despoB/kaihwang/Clock_behav/%s_pe.mat" %(subject)
			pe = io.loadmat(fn)
			pe = np.delete(pe['pe'],drops, axis=0) #delete dropped trial entries
			pe = np.max(pe,axis=1) #for prediction error take the max across time per trial
		
			## create dataframe
			for f, freq in enumerate(TFR.freqs):
				for t, time in enumerate(TFR.times):  
					#tidx = t+250

					pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Pow', 'Pe', 'Age', 'Faces')) 
					pdf.loc[:,'Pow'] = TFR.data[:,:,f,t].squeeze() #TFR.data dimension is trial x channel x freq x time
					pdf.loc[:,'Trial'] = np.arange(TFR.data[:,:,f,t].shape[0])+1  
					pdf.loc[:,'Subject'] = str(subject)
					pdf.loc[:,'Pe'] = pe
					#pdf['Pe'].subtract(pdf['Pe'].mean()) #mean center PE
					
					#pdf['Trial'] = pdf['Trial'].astype('category')  #for testing linear trend of trial history can't set to category..?
					pdf['Subject'] = pdf['Subject'].astype('category')
					pdf['Age'] = age
					pdf['Faces'] = faces

					if s ==0: #first subject
						Data[(freq,time)] = pdf
					else:
						Data[(freq,time)] = pd.concat([Data[(freq,time)], pdf])


		## Response lock analysis				
		elif (parameters =='Value') & (Event_types == 'RT'):
			#get PE model parameters
			fn = "/home/despoB/kaihwang/Clock_behav/%s_value.mat" %(subject)
			value = io.loadmat(fn)
			value = np.delete(value['value'],drops, axis=0)	
			value = np.max(value, axis=1) # take the max

			## create dataframe
			for f, freq in enumerate(TFR.freqs):
				for t, time in enumerate(TFR.times):  
					#tidx = t+250

					pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Pow', 'Value', 'Age', 'Faces')) 
					pdf.loc[:,'Pow'] = TFR.data[:,:,f,t].squeeze() #TFR.data dimension is trial x channel x freq x time
					pdf.loc[:,'Trial'] = np.arange(TFR.data[:,:,f,t].shape[0])+1  
					pdf.loc[:,'Subject'] = str(subject)
					pdf.loc[:,'Value'] = value
					#pdf['Value'].subtract(pdf['Value'].mean()) #mean center PE
					
					#pdf['Trial'] = pdf['Trial'].astype('category')
					pdf['Subject'] = pdf['Subject'].astype('category')
					pdf['Age'] = age
					pdf['Faces'] = faces

					if s ==0: #first subject
						Data[(freq,time)] = pdf
					else:
						Data[(freq,time)] = pd.concat([Data[(freq,time)], pdf])				
		

		##in the case of testing for value function		
		elif (parameters =='Value') & (Event_types == 'clock'):
			# get value model parameters
			fn = "/home/despoB/kaihwang/Clock_behav/%s_value.mat" %(subject)
			value = io.loadmat(fn)
			value = np.delete(value['value'],drops, axis=0)				

			for f, freq in enumerate(TFR.freqs):
				for t in range(np.shape(TFR.data)[0]): #loop through trials
					pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Time', 'Pow', 'Value', 'Age', 'Faces')) 
					pdf.loc[:,'Pow'] = signal.decimate(TFR.data[t,:,f,251:].squeeze(),25) #TFR.data[t,:,f,251:].squeeze()
					pdf.loc[:,'Trial'] = t+1
					pdf.loc[:,'Subject'] = str(subject)
					pdf.loc[:,'Value'] = value[t,]  #.repeat(25)  # should we upsample value function, which was 100ms resolution to 4ms, or shouuld we downsample TFR?
					pdf.loc[:,'Time'] = np.arange(value[t,].shape[0])+1
					#pdf['Value'].subtract(pdf['Value'].mean())
					#pdf['Trial'] = pdf['Trial'].astype('category')
					pdf['Subject'] = pdf['Subject'].astype('category')
					pdf['Age'] = age
					pdf['Faces'] = faces[t]

					if s ==0: #first subject
						Data[(freq)] = pdf
					else:
						Data[(freq)] = pd.concat([Data[(freq)], pdf])	
		else:
			print('something wrong with parameter or event input, can only do clock if testing value funciton, feedback if testing Pe')
			return				
						

	## Regression
	if do_reg:
		if (parameters =='Pe') & (Event_types == 'feedback'):
			RegStats = dict()
			
			for freq in [freqs]:
				for time in times[250:]:
					Data[(freq,time)] = Data[(freq,time)].dropna()

					#for some reason getting inf, get rid of outliers, and look into this later
					#Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pow']<300] 
					Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pe']!=0] #remove first 3 trials whith no behav parameters (0)
					#Data[(freq,time)]['Pe'].subtract(Data[(freq,time)]['Pe'].mean()) #grand mean centering
					#Data[(freq,time)]['Age'].subtract(Data[(freq,time)]['Age'].mean())
					Data[(freq,time)]['Pe'] = zscore(Data[(freq,time)]['Pe'])
					Data[(freq,time)]['Age'] = zscore(Data[(freq,time)]['Age'])
					Data[(freq,time)]['Trial'] = zscore(Data[(freq,time)]['Trial'])

					md = smf.mixedlm("Pow ~ Trial + Pe + Age + Age*Pe + Faces + Faces*Age*Pe + Faces*Pe ", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Pe  ").fit(reml=False)
					# this is equivalent to this in R's lme4: Pow ~ 1 + Pe + Age + Age*pe + Faces + Faces*Pe + Faces*Age*Pe + (1+Pe | Subject)
					# note only full ML estimation will return AIC
					
					RegStats[(chname, freq, time, 'parameters')] = md.params.copy()
					RegStats[(chname, freq, time, 'zvalue')] = md.tvalues.copy()
					RegStats[(chname, freq, time, 'llf')] = md.llf.copy()
					RegStats[(chname, freq, time, 'pvalues')] = md.pvalues.copy()
					RegStats[(chname, freq, time, 'conf_int')] = md.conf_int().copy()
					RegStats[(chname, freq, time, 'aic')] = md.aic

				fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'		
				save_object(RegStats, fn)
		
		if (parameters =='Value') & (Event_types == 'clock'):
			RegStats = dict()

			for freq in [freqs]:
				Data[(freq)] = Data[(freq)].dropna()
				#Data[(freq)]=Data[(freq)][Data[(freq)]['Pow']<300] 
				Data[(freq)]=Data[(freq)][Data[(freq)]['Value']!=0]
				Data[(freq)]['Value'].subtract(Data[(freq)]['Value'].mean()) #grand mean centering
				Data[(freq)]['Age'].subtract(Data[(freq)]['Age'].mean())

				#vcf = {"Trial": "0+C(Trial)"} #fit nested random effect for trial, but takes FOREVER to run....
				#,vc_formula = vcf 
				md = smf.mixedlm("Pow ~ Trial + Value + Age + Age*Value + Faces + Faces*Age*Value + Faces*Value ", Data[(freq)], groups=Data[(freq)]["Subject"], re_formula="~Value ").fit(reml=False)

				RegStats[(chname, freq, 'parameters')] = md.params.copy()
				RegStats[(chname, freq, 'zvalue')] = md.tvalues.copy()
				RegStats[(chname, freq, 'llf')] = md.llf.copy()
				RegStats[(chname, freq, 'pvalues')] = md.pvalues.copy()
				RegStats[(chname, freq, 'conf_int')] = md.conf_int().copy()
				RegStats[(chname, freq, 'aic')] = md.aic

				fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'		
				save_object(RegStats, fn)


		if (parameters =='Value') & (Event_types == 'RT'):
			RegStats = dict()

			for freq in [freqs]:
				for time in times:
					Data[(freq,time)] = Data[(freq,time)].dropna()
					#Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pow']<300]
					Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Value']!=0]
					Data[(freq,time)]['Value'].subtract(Data[(freq,time)]['Value'].mean()) #grand mean centering
					Data[(freq,time)]['Age'].subtract(Data[(freq,time)]['Age'].mean())

					md = smf.mixedlm("Pow ~ Trial + Value + Age + Age*Value + Faces + Faces*Age*Value + Faces*Value ", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Value").fit(reml=False)

					RegStats[(chname, freq, time, 'parameters')] = md.params.copy()
					RegStats[(chname, freq, time, 'zvalue')] = md.tvalues.copy()
					RegStats[(chname, freq, time, 'llf')] = md.llf.copy()
					RegStats[(chname, freq, time, 'pvalues')] = md.pvalues.copy()
					RegStats[(chname, freq, time, 'conf_int')] = md.conf_int().copy()
					RegStats[(chname, freq, time, 'aic')] = md.aic

				fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'		
				save_object(RegStats, fn)

		return RegStats
	
	else:	
		return Data		
	
	#left over
	#fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_clock' + '_tfr'
	#save_object(Data, fn)		

	### plot to look at distribution
	#%matplotlib qt
	#g=sns.jointplot('Pow','Pe',data=D])   

		

def run_autoreject(subject):
	'''run autoreject through epochs, save autoreject object, will read bad data segment indices later'''

	#params
	check_random_state(42)
	n_interpolates = np.array([1, 4, 8, 16, 24, 32, 40])
	consensus_percs = np.linspace(0, 1.0, 11)

	Event_types = ['clock', 'feedback', 'ITI', 'RT']
	e = raw_to_epoch(subject, Event_types, autoreject = False)

	for event in Event_types:
		epochs = e[event]
		#fn = '/home/despoB/kaihwang/Clock/Group/' + 'beforearreject'
		#save_object(epochs, fn)

		#do grad 
		picks = mne.pick_types(epochs.info, meg='grad', eeg=False, stim=False, eog=False, include=[], exclude=[])
		thresh_func = partial(compute_thresholds, picks=picks, random_state=42)
		ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks, thresh_func=thresh_func)	
		egrad = ar.fit(epochs)
		fn = '/home/despoB/kaihwang/Clock/autoreject/' + str(subject) + '_ar_' + event + '_grad'
		save_object(egrad, fn)
		
		#do mag
		picks = mne.pick_types(epochs.info, meg='mag', eeg=False, stim=False, eog=False, include=[], exclude=[])
		thresh_func = partial(compute_thresholds, picks=picks, random_state=42)
		ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks, thresh_func=thresh_func)
		emag = ar.fit(epochs)	
		fn = '/home/despoB/kaihwang/Clock/autoreject/' + str(subject) + '_ar_' + event + '_mag'
		save_object(emag, fn)


def get_bad_channels_and_trials(subject, event, threshold):
	''' get list of bad channels and trails from autoreject procedure, need to give threshold (percentage of bad segments to be rejected)'''

	channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	fn = '/home/despoB/kaihwang/Clock/autoreject' + '/%s_ar_%s_grad' %(subject, event)
	grad = read_object(fn)
	fn = '/home/despoB/kaihwang/Clock/autoreject' + '/%s_ar_%s_mag' %(subject, event)
	mag = read_object(fn)

	#num_trial = grad.bad_segments.shape[0]
	b=mag.bad_segments.mean(axis=0)>threshold
	a=grad.bad_segments.mean(axis=0)>threshold
	bad_channels = channels_list[a[0:306]+b[0:306]]

	b=mag.bad_segments.mean(axis=1)>threshold
	a=grad.bad_segments.mean(axis=1)>threshold
	bad_trials = np.where(a+b)[0]

	return bad_channels, bad_trials


def run_TFR_regression(chname, hz):
	''' wrap for TFR then regression, channel by channel, freq by freq'''
	fb_Epoch, Baseline_Epoch, _ = get_epochs_for_TFR_regression(chname, 'feedback')
	#ck_Epoch, _, _ = get_epochs_for_TFR_regression(chname, 'clock')
	#rt_Epoch, _, _= get_epochs_for_TFR_regression(chname, 'RT')
	
	#for hz in fullfreqs:
	Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, parameters='Pe')
	#clockdata = TFR_regression(ck_Epoch, Baseline_Epoch, chname, hz, 'clock', do_reg = True, parameters='Value')
	#RTdata = TFR_regression(rt_Epoch, Baseline_Epoch, chname, hz, 'RT', do_reg = True, parameters='Value')


def compile_group_reg(trial_type = 'feedback', fdr_correction = True):
	''' assemble regression results freq by freq, channel by channel...'''

	### load gorup power ave epoch template
	# trials TFR time locked to clock onset
	#clock_power = mne.time_frequency.read_tfrs('Data/group_clock_power-tfr.h5')
	# trials TFR time locked to response onset
	#RT_power = mne.time_frequency.read_tfrs('Data/group_RT_power-tfr.h5')
	# trials TFR time locked to feedback onset
	regdatadir = '/home/despoB/kaihwang/Clock/Group/'
	freqs = np.loadtxt('/home/despoB/kaihwang/bin/Clock/fullfreqs')
	channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	metrics = ['zvalue', 'pvalues']
	parameters = ['Pe', 'Age', 'Age:Pe', 'Faces[T.Fear]', 'Faces[T.Happy]', 'Faces[T.Happy]:Pe', 'Faces[T.Fear]:Pe', 
	'Faces[T.Happy]:Age','Faces[T.Fear]:Age', 'Faces[T.Happy]:Age:Pe', 'Faces[T.Fear]:Age:Pe', 'Trial', 'Intercept'] #, 'Trial'

	if trial_type == 'feedback':
		template = mne.time_frequency.read_tfrs('/home/despoB/kaihwang/bin/Clock/Data/group_feedback_power-tfr.h5')[0]

		#setup var
		Output ={}
		for metric in metrics:
			Output[metric]={}
			for param in parameters:
				Output[metric][param] = np.zeros(template.data.shape)
				
		# Output = {
		# 'zvalue': {
		# 'Pe' : np.zeros(template.data.shape), 
		# 'Age' : np.zeros(template.data.shape),
		# 'Age:Pe' : np.zeros(template.data.shape),
		# 'Faces[T.Fear]': np.zeros(template.data.shape),
		# 'Faces[T.Happy]': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Age': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Age': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Age:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Age:Pe': np.zeros(template.data.shape),
		# },
		# 'pvalues': {
		# 'Pe' : np.zeros(template.data.shape), 
		# 'Age' : np.zeros(template.data.shape),
		# 'Age:Pe' : np.zeros(template.data.shape),
		# 'Faces[T.Fear]': np.zeros(template.data.shape),
		# 'Faces[T.Happy]': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Age': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Age': np.zeros(template.data.shape),
		# 'Faces[T.Happy]:Age:Pe': np.zeros(template.data.shape),
		# 'Faces[T.Fear]:Age:Pe': np.zeros(template.data.shape),
		# }}
		#pedata = np.zeros(template.data.shape)
		#agedata = np.zeros(template.data.shape)
		#agexpedata = np.zeros(template.data.shape)

		for ch in channels_list:	
			pick_ch = mne.pick_channels(channels_list.tolist(),[ch]) #has to be list, annoying
			
			
			for ih, hz in enumerate(freqs):
				try:
					fn = regdatadir + '%s_%shz_feedback_mlm.stats' %(ch, hz)
					ds = read_object(fn)
				except:	
					continue

				for it, t in enumerate(template.times[250:]): #no negative timepoint
					
					# Intercept as mean power
					template.data[pick_ch,ih,it+250] = ds[(str(ch), hz, t, 'parameters')]['Intercept']

					for metric in metrics:
						for param in parameters:

							Output[metric][param][pick_ch,ih,it+250] = ds[(str(ch), hz, t, metric)][param]
							#pedata[pick_ch,ih,it+250] = ds[(str(ch), hz, t, 'zvalue')]['Pe']
							#agedata[pick_ch,ih,it+250] = ds[(str(ch), hz, t, 'zvalue')]['Age']
							#agexpedata[pick_ch,ih,it+250] = ds[(str(ch), hz, t, 'zvalue')]['Age:Pe']

							
		# FRD correction to create significant mask	
		if fdr_correction:
			Sig_mask = {}

			for param in parameters:
				Sig_mask[param] = np.zeros(template.data.shape)==1
			
			# Sig_mask = {
			# 'Pe' : np.zeros(template.data.shape)==1, 
			# 'Age' : np.zeros(template.data.shape)==1,
			# 'Age:Pe' : np.zeros(template.data.shape)==1,
			# 'Faces[T.Fear]': np.zeros(template.data.shape)==1,
			# 'Faces[T.Happy]': np.zeros(template.data.shape)==1,
			# 'Faces[T.Happy]:Pe': np.zeros(template.data.shape)==1,
			# 'Faces[T.Fear]:Pe': np.zeros(template.data.shape)==1,
			# 'Faces[T.Happy]:Age': np.zeros(template.data.shape)==1,
			# 'Faces[T.Fear]:Age': np.zeros(template.data.shape)==1,
			# 'Faces[T.Happy]:Age:Pe': np.zeros(template.data.shape)==1,
			# 'Faces[T.Fear]:Age:Pe': np.zeros(template.data.shape)==1
			# }
			
			for param in parameters:
				ps = Output['pvalues'][param][:,:,250:]
				ps[ps==0]=np.inf #missing values 
				Sig_mask[param][:,:,250:] = np.reshape(multipletests(ps.flatten(),alpha=0.05,method='fdr_by')[0], ps.shape)

				#Sig_mask[param] = pmask
				#param : pmask}


		return Output, template, Sig_mask 

def get_exampledata():
	### get data to Michael to check Model fit
	files = glob.glob('*_data')

	
	for f in files:
		a= read_object(f)
		keys = a.keys()

		df = pd.DataFrame()
		for k in keys:
			d = a[k]
			d['Time'] = k[1]
			d['Freq'] = k[0]

			df = pd.concat([df, d])
		fn = 'MEG2232_%s.csv' %k[0]	
		df.to_csv(fn)



if __name__ == "__main__":	
	
	#### run indiv subject pipeline
	# subject = raw_input()
	# indiv_subject_raw_to_tfr(subject)

	#### group average evoke response
	#run_group_ave_evoke()

	#### group averaged TFR power
	# run_group_ave_power()

		#to access data: power[event].data
		#to access time: power[event].time
		#to access fre : power[event].freqs
		#to accesss ave: power[event].nave


	#### run autoreject pipeline to get bad data segment indices
	#subject = raw_input()
	#run_autoreject(subject)


	#### need to get bad trial statistics


	#### test single trial TFR conversion
	#chname = raw_input()
	chname='MEG2232'
	#hz = 2
	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')
	fullfreqs = np.logspace(*np.log10([2, 50]), num=20)
	for hz in fullfreqs:
		Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, parameters='Pe')
	#fullfreqs = np.logspace(*np.log10([2, 50]), num=20)

	#chname, hz = raw_input().split()
	#hz = np.float(hz)
	#run_TFR_regression(chname, hz)

	#### exaime distribution
	# chname='MEG2232'
	# freqs = np.logspace(*np.log10([2, 50]), num=20)#10.88
	
	# for hz in freqs:
	# 	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')
	# 	Feedbackdata = TFR_regression(fb_Epoch, BaselineTFRline_Epoch, chname, hz, 'feedback', do_reg = False, parameters='Pe')
	# 	fn = str(hz) +'_data'
	# 	save_object(Feedbackdata, fn)

	### complie freq by freq channel by channel results

	#feedback_reg, avepower, fb_sig_mask = compile_group_reg('feedback')
	#save_object(feedback_reg, 'feedback_reg')
	#save_object(fb_sig_mask, 'feedback_reg_sigmask')
	
	## Visualize
	#plt.ion()
	#matplotlib.use('Qt4Agg')
	#rm ~/.ICEauthority
	#%matplotlib qt

	# avepower = mne.time_frequency.read_tfrs('/home/despoB/kaihwang/bin/Clock/Data/group_feedback_power-tfr.h5')[0]
	# feedback_reg = read_object('/home/despoB/kaihwang/bin/Clock/npr')
	# fb_sig_mask = read_object('feedback_reg_sigmask')

	# avepowerzvalue = avepower.data.copy()


	def plottfrz(param):
		avepower.data = feedback_reg['zvalue'][param]
		avepower.plot_topo(mode=None, tmin=0, vmin=-2, vmax=2, show=True, title = param, yscale='auto') 












