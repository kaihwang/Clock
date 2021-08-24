#### scripts to use MNE to analyze Clock MEG data, UNC edition
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
from functools import partial
from autoreject import compute_thresholds
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
from pymer4.models import Lm, Lmer
from pymer4.utils import get_resource_path
import os

#make paths global
datapath =  '/proj/mnhallqlab/projects/Clock_MEG/fif_data/'
save_path = '/proj/mnhallqlab/projects/Clock_MEG/fif_data/'

#notes on signal scale:
# 'mag': Magnetometers (scaled by 1e+15 to plot in fT)
# 'grad': Gradiometers (scaled by 1e+13 to plot in fT/cm)

def raw_to_epoch(subject, Event_types, channels_list = None):
	'''short hand to load raw fif across runs, and return a combined epoch object
	input: subject, Event_types
	Event_types is a list of strings to indicate the event type you want to extract from raw. possible choices are:
	Event_types = ['clock', 'feedback', 'ITI', 'RT']
	The return of this function will be a dictionary, where key is the event_type, and the item is the mne epoch object.
	'''

	#setup variables

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
	'clock'   : [-2,4],
	'feedback': [-4,1],
	'ITI'     : [0,1],  #no baseline for ITI
	'RT': [-2,2],
	}

	print(Event_types)
	epochs = dict.fromkeys(Event_types)
	epo = []

	for event in Event_types:

		for r in np.arange(1,9):

			try:
				fn = datapath + '%s/MEG/%s_clock_run%s_dn_ds_sss_raw.fif' %(subject, subject, r)
				raw = mne.io.read_raw_fif(fn)
				picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False)
			except:
				st = 'cant read fif file for subject %s run number %s' %(subject, r)
				print(st)
				break # skip all if fif file does not exist

			#adjust evevnts:
			# RT can be calculated by 300ms prior to feedback onset
			# ITI is 850 ms after feedback
			if event == 'ITI':
				try:
					fn = datapath + '%s/MEG/MEG_%s_*%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
					triggers = mne.read_events(glob.glob(fn)[0])
					triggers[1:,0] = triggers[1:,0] +213 #shift 850ms
					triggers = np.delete(triggers, -1, 0)  # delete the last row becuase for some runs final trial doesn't have a long enough ITI. UGH.
					#baseline = None
				except:
					triggers = None

			elif event == 'RT':
				try:
					fn = datapath + '%s/MEG/MEG_%s_*%s_%s_ds4.eve' %(subject, subject, r, 'feedback')
					triggers = mne.read_events(glob.glob(fn)[0])
					triggers[1:,0] = triggers[1:,0] - 75 #shift 300ms
					#baseline = (None, 0.0)
				except:
					triggers = None

			else:
				try:
					fn = datapath + '%s/MEG/MEG_%s_*%s_%s_ds4.eve' %(subject, subject, r, event)
					triggers = mne.read_events(glob.glob(fn)[0])
					#baseline = (None, 0.0)
				except:
					triggers = None

			try:

				e = mne.Epochs(raw, events=triggers, event_id=Event_codes[event],
							tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = None, picks=channels_list, on_missing = 'ignore')

				if any(raw.times[-1]/.004 < triggers[:,0]): #raw.times[-1]/.004 < triggers[-1,0]: #bizzare that preproc  cut off the end...??
					e.drop(np.where(raw.times[-1]/.004 < triggers[:,0])[0]-1, reason='TOO SHORT') #e.drop(e.events.shape[0]-1, reason='TOO SHORT')

				epo.append(e)

			except:
				st = 'cant epoch for subject %s run number %s' %(subject, r)
				print(st)

				pass
			# if r == 1: #create epoch with first run
			# 	if triggers is not None:

			# 		try:
			# 			epochs[event] = mne.Epochs(raw, events=triggers, event_id=Event_codes[event],
			# 				tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = None, picks=channels_list, on_missing = 'ignore')
			# 		except:
			# 			st = 'cant epoch for subject %s run number %s' %(subject, r)
			# 			print(st)
			# 			pass #if fif file exist but fail for whatev reason
			# 	else:
			# 		pass

			# else: #concat epochs
			# 	if triggers is not None:

			# 		try:
			# 			epochs[event] = mne.concatenate_epochs((epochs[event],
			# 				mne.Epochs(raw, events=triggers, event_id=Event_codes[event],
			# 				tmin=Epoch_timings[event][0], tmax=Epoch_timings[event][1], reject=None, baseline = None, picks=channels_list, on_missing = 'ignore')))
			# 		except:
			# 			st = 'cant epoch for subject %s run number %s' %(subject, r)
			# 			print(st)
			# 			pass
			# 	else:
			# 		pass

		epochs[event] = mne.concatenate_epochs(epo)

	return epochs


def get_dropped_trials_list(epoch):
	'''mne_read_epoch will automatically drop trials that are too short without warning, so need to retrieve those tiral indx...'''
	try:
		drop_log = epoch[list(epoch.keys())[0]].drop_log
	except:
		drop_log =epoch.drop_log
	drop_log = list(drop_log)
	trial_list = []

	# event lists start with "0 0 0", get rid of those
	for n in range(0,len(drop_log)):
		if drop_log[n] == ('IGNORED',):
			trial_list.append(n)

	for index in sorted(trial_list, reverse=True):
		del drop_log[index]

	drop_list = []
	for n in range(0,len(drop_log)):
		if drop_log[n]!=():  #get list of trials dropped for whatever reason, note rejected bad trials will also be here.
			 drop_list.append(n)

	drop_list = np.array(drop_list)
	return drop_list


def get_epoch_trial_types(epoch):
	''' get the order of face conditions'''

	try:
		trig_codes = epoch[list(epoch.keys())[0]].events[:,2]
	except:
		trig_codes = epoch.events[:,2]
	try:
		event =list(epoch.keys())[0]
	except:
		event = 'feedback'

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
		if any(trig == Face_codes[event]['fear.face']):
			out[i] = 'Fear'
		elif any(trig == Face_codes[event]['happy.face']):
			out[i] = 'Happy'
		elif any(trig == Face_codes[event]['scram.face']):
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

	if freqs is None:
	#if not any(freqs): #do full spec if not specified
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

	#datapath = '/home/despoB/kaihwang/Clock/'

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
	#save_path='/data/backed_up/kahwang/Clock/'
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
	#datapath = '/home/despoB/kaihwang/Clock/'

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
	#save_path='/home/despoB/kaihwang/bin/Clock'
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


def get_epochs_for_TFR_regression(chname, subjects, channels_list, Event_types):
	''' get epoch and baseline epoch for TFR_regression
	return trial epoch, baseline epoch, and list of drop trials
	trying to speed up TFR_regression loop without the need of keep reading in data from disk
	Will output trial epoch, baseline epoch, and list of bad trilas as dict (where subject ID is key) '''

	#subjects = [10637, 10638] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#subjects = [11335]

	#subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
	#subjects =[10891]
	#channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
	pick_ch = mne.pick_channels(channels_list.tolist(), include=[chname])

	Event_Epoch = {}
	Baseline_Epoch = {}
	BadTrial_Lists = {}

	for s, subject in enumerate(subjects):

		#### for now not doing auto rejection
		# #get bad trial and bad channel info
		# try:
		# 	bad_channels, bad_trials = get_bad_channels_and_trials(subject, Event_types, 0.3) #reject if thirty percent of data segment is bad
		# except: #no ar
		# 	bad_channels = np.array([], dtype='<U7')
		# 	bad_trials = np.array([])
		#
		# if chname in bad_channels:
		# 	continue #skip if bad channel
		#
		# try:
		# 	baseline_bad_channels, baseline_bad_trials = get_bad_channels_and_trials(subject, 'ITI', 0.3) #reject if thirty percent of data segment is bad
		# except: #no ar
		# 	baseline_bad_channels = np.array([], dtype='<U7')
		# 	baseline_bad_trials = np.array([])
		#
		# if chname in baseline_bad_channels:
		# 	continue #skip if bad channel

		# create epoch with one channel of data
		e = raw_to_epoch(subject, [Event_types], channels_list = pick_ch)
		b = raw_to_epoch(subject, ['ITI'], channels_list = pick_ch) #ITI baseline

		if e[Event_types]==None:
			continue

		#drop bad trials
		#e[Event_types].drop(bad_trials)
		#b['ITI'].drop(baseline_bad_trials)

		#get list of trials dropped
		drops = get_dropped_trials_list(e)

		Event_Epoch[subject] = e
		Baseline_Epoch[subject] = b
		BadTrial_Lists[subject] = drops

	return Event_Epoch, Baseline_Epoch, BadTrial_Lists



def TFR_regression(Event_Epoch, Baseline_Epoch, chname, freqs, Event_types, do_reg = True, global_model = True, robust_baseline = True, parameters ='Pe'):
	''' compile TFR dataframe and model params for regression.
	Need output from get_epochs_for_TFR_regression() as input.
	Event_Epoch is a dict with each subjects trial epoch, Baseline_Epoch is the baseline epoch period.
	Will do one channel a time (chname), and read model parameters from Michael's matlab output
	'''

	#subjects = [11345, 11346, 11347] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#subjects = np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)

	subjects = list(Event_Epoch.keys())
	#Event_types='clock'
	#channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	#chname = 'MEG0713'
	#pick_ch = mne.pick_channels(channels_list.tolist(),[chname])

	#mne.set_log_level('WARNING')
	demographic = pd.read_csv('/proj/mnhallqlab/projects/Clock_MEG/code/subinfo_db', sep='\t')

	if global_model: #read global fit from Michael to get demo info

		global_model_df = pd.read_csv('/proj/mnhallqlab/projects/Clock_MEG/code/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics_reorganized.csv')
		# global_model_df = pd.read_csv('/proj/mnhallqlab/projects/Clock_MEG/code/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics.csv')

		# for i in range(len(global_model_df)):
		# 	global_model_df.loc[i,'id'] = global_model_df.loc[i,'id'][0:5]

		# global_model_df['id'] = global_model_df['id'].astype(int)
		# global_model_df['Rewarded'] = global_model_df['score_csv']>0

	Data = dict()
	for s, subject in enumerate(subjects):

		#get subject's age
		if not global_model:
			try:
				age = demographic[demographic['lunaid']==subject]['age'].values[0]
			except:
				age = np.nan # no age info...???

		if global_model:
			try:
				age = demographic[demographic['lunaid']==subject]['age'].values[0] ## no age in csv?? ask Michael
			except:
				age = np.nan

		## check if skip because of bad channels
		try:
			bad_channels, _ = get_bad_channels_and_trials(subject, Event_types, 0.3) #reject if thirty percent of data segment is bad
		except: #no ar
			bad_channels = np.array([], dtype='<U7')
			#bad_trials = np.array([])

		if chname in bad_channels:
			continue #skip if bad channel

		try:
			baseline_bad_channels, _ = get_bad_channels_and_trials(subject, 'ITI', 0.3) #reject if thirty percent of data segment is bad
		except: #no autoreject record
			baseline_bad_channels = np.array([], dtype='<U7')
			#baseline_bad_trials = np.array([])

		if chname in baseline_bad_channels:
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
		print(subject)
		## baseline correction
		#TFR.apply_baseline((-1,-.2), mode='zscore')
		baseline_power = np.tile(np.mean(np.mean(BaselineTFR.data,axis=3),axis=0),(TFR.data.shape[0],1,1,TFR.data.shape[3])) #ave across time and trial, then broadcast to the right dimension
		#baseline_power_sd = np.broadcast_to(np.std(BaselineTFR.data.flatten()), TFR.data.shape)

		## this is convert to percent signal change
		#TFR.data = 100*((TFR.data - baseline_power) / baseline_power) #convert to percent of signal change

		## Michale thinks it is better to convert to db, which is 10log10(singal/noise)
		TFR.data = 10 * np.log10(TFR.data / baseline_power)

		#TFR.data = (TFR.data - baseline_power) / baseline_power_sd #convert to zscore
		times = TFR.times

		## extract model parameters and freq poer into dataframe
		##in the case of testing for PE
		if (parameters =='Pe') & (Event_types == 'feedback'):

			#get PE model parameters from new csv fitted to group data (per Michael)


			#get PE model parameters from individual model fit

			if global_model:
				pe = global_model_df.loc[global_model_df['id'] == subject]['pe_max'].values
				pe = np.delete(pe, drops, axis=0)

				Rewarded = global_model_df.loc[global_model_df['id'] == subject]['Rewarded'].values
				Rewarded = np.delete(Rewarded, drops, axis=0)

				run = global_model_df.loc[global_model_df['id'] == subject]['run'].values
				run = np.delete(run, drops, axis=0)

			else:
				fn = "/data/backed_up/kahwang/Clock_behav/%s_pe.mat" %(subject)
				pe = io.loadmat(fn)
				#pe = np.delete(pe['pe'],drops, axis=0) #delete dropped trial entries
				pe = np.max(pe,axis=1) #for prediction error take the max across time per trial

			#### here insert code to check if number of trials for each block matches between epoch and model_df


			## create dataframe
			for f, freq in enumerate(TFR.freqs):
				for t, time in enumerate(TFR.times):
					#tidx = t+250

					pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Pow', 'Pe', 'Age', 'Faces'))
					pdf.loc[:,'Pow'] = TFR.data[:,:,f,t].squeeze() #TFR.data dimension is trial x channel x freq x time
					pdf.loc[:,'Trial'] = np.arange(TFR.data[:,:,f,t].shape[0])+1
					pdf.loc[:,'Subject'] = str(subject)
					try:
						pdf.loc[:,'Pe'] = pe
					except:
						msg = 'check number of trials in epoch and pe model for subject %s' %subject
						print(msg)
						pdf.loc[:,'Pe'] = np.nan

					try:
						pdf.loc[:,'Rewarded'] = Rewarded
					except:
						msg = 'check number of trials in epoch and rewarded model for subject %s' %subject
						print(msg)
						pdf.loc[:,'Rewarded'] = np.nan

					try:
						pdf.loc[:,'Run'] = run
					except:
						pdf.loc[:,'Run'] = np.nan
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
			fn = "/data/backed_up/kahwang/Clock_behav/%s_value.mat" %(subject)
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
			fn = "/data/backed_up/kahwang/Clock_behav/%s_value.mat" %(subject)
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
					Data[(freq,time)]['Rewarded'] = Data[(freq,time)]['Rewarded'].astype(int)
					#for some reason getting inf, get rid of outliers, and look into this later
					Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pe']!=0] #remove first 3 trials whith no behav parameters (0)
					#Data[(freq,time)]['Pe'].subtract(Data[(freq,time)]['Pe'].mean()) #grand mean centering
					#Data[(freq,time)]['Age'].subtract(Data[(freq,time)]['Age'].mean())
					Data[(freq,time)]['Pe'] = zscore(Data[(freq,time)]['Pe'])
					Data[(freq,time)]['Age'] = zscore(Data[(freq,time)]['Age'])
					Data[(freq,time)]['Trial'] = zscore(Data[(freq,time)]['Trial'])
					Data[(freq,time)] = Data[(freq,time)].loc[Data[(freq,time)]['Pow']!=-np.inf]  # account for 0 power
					Data[(freq,time)] = Data[(freq,time)].loc[Data[(freq,time)]['Pow']>-300]

					####----Model after discussion with Michael and Alex in Jan 2019----####

                                        # MH 24Jul2019: Remove random (linear) slope of trial because of pre-stimulus baseline normalization. If pre-stim baseline is used to normalize Power, then
                                        #   this should, as far as I can tell, substantially reduce or eliminate linear drifts in power across the run, let alone individual differences (random effect)
                                        #   in the drift. We are getting a lot of singular fit messages because there is a 0 variance component on the trial slope. Although this does not invalidate the
                                        #   parameter estimates, it does suggest we're fitting a model that is more complex than the data support. I've retained Trial as a fixed effect only to allow
                                        #   for a sample-level drift term, even though this will probably be ~0, on average.
                                        
					if robust_baseline:
						#formula = "Pow ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces + (0 + Trial | Subject) + (1 | Subject/Run)"  #"Pow ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces"
                                                formula = "Pow ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces + (1 | Subject/Run)"  #"Pow ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces"
					else:
						formula = "Pow ~ Faces  + Age + Faces*Age + Trial + Pe + Pe*Faces + (0 + Trial | Subject) + (1 | Subject/Run)" #"Pow ~ Faces  + Age + Faces*Age + Trial + Pe + Pe*Faces"

					#vcf = {"Run": "0+C(Run)"}
					#groups = Data[(freq,time)]["Subject"].values
					#ref = "~Trial"
					#md = sm.MixedLM.from_formula(formula = formula, data = Data[(freq,time)], vc_formula = vcf, groups = "Subject", re_formula = ref).fit(reml=False)

					md = Lmer(formula ,data=Data[(freq,time)])
					md_output = md.fit(REML=False)

					# model in lme4:
					# robust_baseline <- lmer(Pow_dB ~ 1 + Faces * Age_z + Trial_z + Rewarded * Faces + (1 + Trial_z | Subject/Run), dataset)
					# pe_basic <- lmer(Pow_dB ~ 1 + Faces * Age_z + Trial_z + Pe_z * Faces + (1 + Trial_z | Subject/Run), dataset)


					####----Model tested in 2018-----####
					#### md = smf.mixedlm("Pow ~ Trial + Pe + Age + Age*Pe + Faces + Faces*Age*Pe + Faces*Pe ", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Pe  ").fit(reml=False)
					#### this is equivalent to this in R's lme4: Pow ~ 1 + Pe + Age + Age*pe + Faces + Faces*Pe + Faces*Age*Pe + (1+Pe | Subject)
					#### note only full ML estimation will return AIC

					RegStats[(chname, freq, time, 'parameters')] = md_output['Estimate'] #md.params.copy()
					RegStats[(chname, freq, time, 'zvalue')] = md_output['T-stat']
					RegStats[(chname, freq, time, 'llf')] = md.logLike
					RegStats[(chname, freq, time, 'pvalues')] = md_output['P-val']
					RegStats[(chname, freq, time, '2.5_ci')] = md_output['2.5_ci']
					RegStats[(chname, freq, time, '97.5_ci')] = md_output['97.5_ci']
					RegStats[(chname, freq, time, 'aic')] = md.AIC

				if robust_baseline:
					fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_RobustBaseline' + '_mlm.stats'
				if not robust_baseline:
					fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_fullmodel' + '_mlm.stats'
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

				fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'
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

				fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'
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

	Event_types = ['feedback'] #'clock',  'RT' 'ITI'
	e = raw_to_epoch(subject, Event_types)

	for event in Event_types:
		epochs = e[event]
		#fn = '/home/despoB/kaihwang/Clock/Group/' + 'beforearreject'
		#save_object(epochs, fn)

		#do grad
		picks = mne.pick_types(epochs.info, meg='grad', eeg=False, stim=False, eog=False, include=[], exclude=[])
		thresh_func = partial(compute_thresholds, picks=picks, random_state=42)
		ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks, thresh_func=thresh_func)
		egrad = ar.fit(epochs)
		fn = datapath +'autoreject/' + str(subject) + '_ar_' + event + '_grad'
		save_object(egrad, fn)

		#do mag
		picks = mne.pick_types(epochs.info, meg='mag', eeg=False, stim=False, eog=False, include=[], exclude=[])
		thresh_func = partial(compute_thresholds, picks=picks, random_state=42)
		ar = LocalAutoRejectCV(n_interpolates, consensus_percs, picks=picks, thresh_func=thresh_func)
		emag = ar.fit(epochs)
		fn = datapath +'autoreject/' + str(subject) + '_ar_' + event + '_mag'
		save_object(emag, fn)


def get_bad_channels_and_trials(subject, event, threshold):
	''' get list of bad channels and trails from autoreject procedure, need to give threshold (percentage of bad segments to be rejected)'''

	channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
	fn = datapath +'autoreject/' + '/%s_ar_%s_grad' %(subject, event)
	grad = read_object(fn)
	fn = datapath +'autoreject/' + '/%s_ar_%s_mag' %(subject, event)
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
	Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, global_model = True, parameters='Pe')
	#clockdata = TFR_regression(ck_Epoch, Baseline_Epoch, chname, hz, 'clock', do_reg = True, parameters='Value')
	#RTdata = TFR_regression(rt_Epoch, Baseline_Epoch, chname, hz, 'RT', do_reg = True, parameters='Value')


def compile_group_reg(trial_type = 'feedback', model= 'fullmodel', fdr_correction = True):
	''' assemble regression results freq by freq, channel by channel...'''

	### load gorup power ave epoch template
	# trials TFR time locked to clock onset
	#clock_power = mne.time_frequency.read_tfrs('Data/group_clock_power-tfr.h5')
	# trials TFR time locked to response onset
	#RT_power = mne.time_frequency.read_tfrs('Data/group_RT_power-tfr.h5')
	# trials TFR time locked to feedback onset
	regdatadir = '/data/backed_up/kahwang/Group' #datapath + 'Group/'
	freqs = np.arange(2,62,2)#np.loadtxt('/home/kahwang/bin/Clock/fullfreqs')
	channels_list = np.load('/data/backed_up/kahwang/bin/Clock/channel_list.npy')
	metrics = ['zvalue', 'pvalues']
	parameters = ['Pe', 'Age', 'Age:Pe', 'Faces[T.Fear]', 'Faces[T.Happy]', 'Faces[T.Happy]:Pe', 'Faces[T.Fear]:Pe',
	'Faces[T.Happy]:Age','Faces[T.Fear]:Age', 'Faces[T.Happy]:Age:Pe', 'Faces[T.Fear]:Age:Pe', 'Trial', 'Intercept'] #, 'Trial'

	if trial_type == 'feedback':
		template = mne.time_frequency.read_tfrs('/data/backed_up/kahwang/Clock/Group/group_feedback_power-tfr.h5')[0]
		template.freqs = freqs
		template.times = template.times[250:-1]
		template.data = np.zeros((306, len(freqs), len(template.times)))
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
					fn = regdatadir + '%s_%shz_feedback_%s_mlm.stats' %(ch , hz, model)
					ds = read_object(fn)
				except:
					continue

				for it, t in enumerate(template.times[250:]): #no negative timepoint

					# Intercept as mean power
					template.data[pick_ch,ih,it] = ds[(str(ch), hz, t, 'parameters')]['(Intercept)'] 

					for metric in metrics:
						for param in parameters:

							Output[metric][param][pick_ch,ih,it] = ds[((str(ch), hz, t, metric))][param]
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
		else:
			return Output, template

def get_exampledata(data):
	### get data to Michael to check Model fit
	#fn = "*%s*_data" %chname
	#files = glob.glob(fn)

	df = pd.DataFrame()

	#for f in files:
	#	a = read_object(f)
	keys = list(data.keys())

	for k in keys:
		d = data[k]
		d['Time'] = k[1]
		d['Freq'] = k[0]

		df = pd.concat([df, d])
	#fn = 'MEG2232_%s.csv' %k[0]
	#df.to_csv(fn)

	return df


def plottfrz(param):
		avepower.data = feedback_reg['zvalue'][param]
		avepower.plot_topo(mode=None, tmin=0, vmin=-2, vmax=2, show=True, title = param, yscale='auto')



def get_cluster():

	# vectorize time frequency 3d mat into 2d, chnl by data
	from scipy.cluster.hierarchy import dendrogram, linkage
	from collections import defaultdict
	from scipy.cluster.hierarchy import fcluster

	avepower = mne.time_frequency.read_tfrs('/data/backed_up/kahwang/Clock/Group/group_feedback_power-tfr.h5')[0]
	datavec = np.zeros((306, 20*404))

	for ch in np.arange(306):
		datavec[ch,:] = avepower.data[ch,:,30:-30].flatten()

	# generate adj matrices
	R = np.corrcoef(datavec)

	#linkage then cluster
	Z = linkage(datavec, 'ward')
	#dn = dendrogram(Z)

	#7 seems reasonable
	ci=fcluster(Z, 7, criterion='maxclust')

	# plot clustered sensors

	#for i in np.unique(ci):
	#	avepower.plot_topo(picks=np.where(ci==i)[0], vmin=-3, vmax=3)

	### get example sensor data from hier clusters to Michael

	channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')

	for i in np.unique(ci):

		chname = channels_list[ci==i][0]

		freqs = np.logspace(*np.log10([2, 50]), num=20)
		fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')

		for hz in freqs:
			Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = False, parameters='Pe')
			#fn = 'cluster' + str(i) + '_' + chname + '_' + str(hz) +'_data'
			#save_object(Feedbackdata, fn)

			df = get_exampledata(Feedbackdata)
			fn = 'cluster' + str(i) + '_' + chname + '_' + str(hz) +'_data.csv'
			df.to_csv(fn)


def get_mosaic_mask():
	from scipy.cluster.hierarchy import dendrogram, linkage
	from collections import defaultdict
	from scipy.cluster.hierarchy import fcluster
	avepower = mne.time_frequency.read_tfrs('/data/backed_up/kahwang/Clock/Group/group_feedback_power-tfr.h5')[0]
	datavec = np.zeros((306, 20*404))
	channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
	for ch in np.arange(306):
		datavec[ch,:] = avepower.data[ch,:,30:-30].flatten()
	# generate adj matrices
	R = np.corrcoef(datavec)
	#linkage then cluster
	Z = linkage(datavec, 'ward')
	#dn = dendrogram(Z)
	#7 seems reasonable
	ci=fcluster(Z, 7, criterion='maxclust')

	for i in np.unique(ci):
	#i=1
		chname = channels_list[ci==i][0]
		pos = avepower.data[ mne.pick_channels(channels_list, [chname])][0,:,:] > 2
		neg = avepower.data[ mne.pick_channels(channels_list, [chname])][0,:,:] < -2
		fn = 'mask'+chname
		np.savetxt(fn,neg+pos, fmt='%1d')


def Evoke_regression(Event_Epoch, Baseline_Epoch, chname, demographic, global_model_df, Event_types, do_reg = True, global_model = True, robust_baseline = True, parameters ='Pe'):
	''' compile evoke dataframe and model params for regression.
	Need output from get_epochs_for_TFR_regression() as input.
	Event_Epoch is a dict with each subjects trial epoch, Baseline_Epoch is the baseline epoch period.
	Will do one channel a time (chname), and read model parameters from Michael's matlab output
	'''

	#subjects = [11345, 11346, 11347] #np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)	 #[10637, 10638, 10662, 10711]
	#subjects = np.loadtxt('/home/despoB/kaihwang/bin/Clock/subjects', dtype=int)

	subjects = list(Event_Epoch.keys())
	#Event_types='clock'
	#channels_list = np.load('/home/despoB/kaihwang/Clock/channel_list.npy')
	#chname = 'MEG0713'
	#pick_ch = mne.pick_channels(channels_list.tolist(),[chname])

	#mne.set_log_level('WARNING')
	#demographic = pd.read_csv('/home/kahwang/bin/Clock/subinfo_db', sep='\t')

	#if global_model: #read global fit from Michael to get demo info

	#	global_model_df = pd.read_csv('/home/kahwang/bin/Clock/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics_reorganized.csv')
		# global_model_df = pd.read_csv('/home/kahwang/bin/Clock/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics.csv')

		# for i in range(len(global_model_df)):
		# 	global_model_df.loc[i,'id'] = global_model_df.loc[i,'id'][0:5]

		# global_model_df['id'] = global_model_df['id'].astype(int)
		# global_model_df['Rewarded'] = global_model_df['score_csv']>0

	Data = dict()
	for s, subject in enumerate(subjects):

		#get subject's age
		if global_model is not None:
			try:
				age = demographic[demographic['lunaid']==subject]['age'].values[0]
			except:
				age = np.nan # no age info...???

		if global_model is not None:
			try:
				age = demographic[demographic['lunaid']==subject]['age'].values[0] ## no age in csv?? ask Michael
			except:
				age = np.nan


		e = Event_Epoch[subject][Event_types]
		b = Baseline_Epoch[subject]['ITI']
		drops = get_dropped_trials_list(e)

		# get list of emo face conditions
		#faces = np.delete(get_epoch_trial_types(e), drops, axis = 0)
		faces = get_epoch_trial_types(e)

		# get trial by trial Evoked data
		#event = 'clock'
		Evoke_data = e.get_data()
		Baseline_data = b.get_data()
		times = e.times

		## baseline correction
		Evoke_data = Evoke_data - np.mean(Baseline_data)
		print(subject)

		## extract model parameters and freq poer into dataframe
		##in the case of testing for PE
		if (parameters =='Pe') & (Event_types == 'feedback'):

			#get PE model parameters from new csv fitted to group data (per Michael)


			#get PE model parameters from individual model fit

			if global_model is not None:
				pe = global_model_df.loc[global_model_df['id'] == subject]['pe_max'].values
				pe = np.delete(pe, drops, axis=0)

				Rewarded = global_model_df.loc[global_model_df['id'] == subject]['Rewarded'].values
				Rewarded = np.delete(Rewarded, drops, axis=0)

				run = global_model_df.loc[global_model_df['id'] == subject]['run'].values
				run = np.delete(run, drops, axis=0)

			else:
				fn = "/data/backed_up/kahwang/Clock_behav/%s_pe.mat" %(subject)
				pe = io.loadmat(fn)
				#pe = np.delete(pe['pe'],drops, axis=0) #delete dropped trial entries
				pe = np.max(pe,axis=1) #for prediction error take the max across time per trial



			## create dataframe
			for t, time in enumerate(times):
				#tidx = t+250

				pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Vol', 'Pe', 'Rewarded', 'Age', 'Faces'))
				pdf.loc[:,'Vol'] = Evoke_data[:,:,t].squeeze() #TFR.data dimension is trial x channel x time
				pdf.loc[:,'Trial'] = np.arange(Evoke_data[:,:,t].shape[0])+1
				pdf.loc[:,'Subject'] = str(subject)
				try:
					pdf.loc[:,'Pe'] = pe
				except:
					msg = 'check number of trials in epoch and pe model for subject %s' %subject
					print(msg)
					pdf.loc[:,'Pe'] = np.nan

				try:
					pdf.loc[:,'Rewarded'] = Rewarded
				except:
					msg = 'check number of trials in epoch and rewarded model for subject %s' %subject
					print(msg)
					pdf.loc[:,'Rewarded'] = np.nan

				try:
					pdf.loc[:,'Run'] = run
				except:
					pdf.loc[:,'Run'] = np.nan
				#pdf['Pe'].subtract(pdf['Pe'].mean()) #mean center PE

				#pdf['Trial'] = pdf['Trial'].astype('category')  #for testing linear trend of trial history can't set to category..?
				pdf['Subject'] = pdf['Subject'].astype('category')
				pdf['Age'] = age
				pdf['Faces'] = faces

				if s ==0: #first subject
					Data[time] = pdf
				else:
					Data[time] = pd.concat([Data[time], pdf])


		# ## Response lock analysis
		# elif (parameters =='Value') & (Event_types == 'RT'):
		# 	#get PE model parameters
		# 	fn = "/data/backed_up/kahwang/Clock_behav/%s_value.mat" %(subject)
		# 	value = io.loadmat(fn)
		# 	value = np.delete(value['value'],drops, axis=0)
		# 	value = np.max(value, axis=1) # take the max

		# 	## create dataframe
		# 	for f, freq in enumerate(TFR.freqs):
		# 		for t, time in enumerate(TFR.times):
		# 			#tidx = t+250

		# 			pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Pow', 'Value', 'Age', 'Faces'))
		# 			pdf.loc[:,'Pow'] = TFR.data[:,:,f,t].squeeze() #TFR.data dimension is trial x channel x freq x time
		# 			pdf.loc[:,'Trial'] = np.arange(TFR.data[:,:,f,t].shape[0])+1
		# 			pdf.loc[:,'Subject'] = str(subject)
		# 			pdf.loc[:,'Value'] = value
		# 			#pdf['Value'].subtract(pdf['Value'].mean()) #mean center PE

		# 			#pdf['Trial'] = pdf['Trial'].astype('category')
		# 			pdf['Subject'] = pdf['Subject'].astype('category')
		# 			pdf['Age'] = age
		# 			pdf['Faces'] = faces

		# 			if s ==0: #first subject
		# 				Data[(freq,time)] = pdf
		# 			else:
		# 				Data[(freq,time)] = pd.concat([Data[(freq,time)], pdf])


		# ##in the case of testing for value function
		# elif (parameters =='Value') & (Event_types == 'clock'):
		# 	# get value model parameters
		# 	fn = "/data/backed_up/kahwang/Clock_behav/%s_value.mat" %(subject)
		# 	value = io.loadmat(fn)
		# 	value = np.delete(value['value'],drops, axis=0)

		# 	for f, freq in enumerate(TFR.freqs):
		# 		for t in range(np.shape(TFR.data)[0]): #loop through trials
		# 			pdf = pd.DataFrame(columns=('Subject', 'Trial', 'Time', 'Pow', 'Value', 'Age', 'Faces'))
		# 			pdf.loc[:,'Pow'] = signal.decimate(TFR.data[t,:,f,251:].squeeze(),25) #TFR.data[t,:,f,251:].squeeze()
		# 			pdf.loc[:,'Trial'] = t+1
		# 			pdf.loc[:,'Subject'] = str(subject)
		# 			pdf.loc[:,'Value'] = value[t,]  #.repeat(25)  # should we upsample value function, which was 100ms resolution to 4ms, or shouuld we downsample TFR?
		# 			pdf.loc[:,'Time'] = np.arange(value[t,].shape[0])+1
		# 			#pdf['Value'].subtract(pdf['Value'].mean())
		# 			#pdf['Trial'] = pdf['Trial'].astype('category')
		# 			pdf['Subject'] = pdf['Subject'].astype('category')
		# 			pdf['Age'] = age
		# 			pdf['Faces'] = faces[t]

		# 			if s ==0: #first subject
		# 				Data[(freq)] = pdf
		# 			else:
		# 				Data[(freq)] = pd.concat([Data[(freq)], pdf])
		else:
			print('something wrong with parameter or event input, can only do clock if testing value funciton, feedback if testing Pe')
			return


	## Regression
	if do_reg:
		if (parameters =='Pe') & (Event_types == 'feedback'):
			RegStats = dict()

			for time in times[250:]:  #skip baseline
				Data[time] = Data[time].dropna()
				Data[time]['Rewarded'] = Data[time]['Rewarded'].astype(int)
				#for some reason getting inf, get rid of outliers, and look into this later
				Data[time]=Data[time][Data[time]['Pe']!=0] #remove first 3 trials whith no behav parameters (0)
				#Data[time]['Pe'].subtract(Data[time]['Pe'].mean()) #grand mean centering
				#Data[time]['Age'].subtract(Data[time]['Age'].mean())
				Data[time]['Pe'] = zscore(Data[time]['Pe'])
				Data[time]['Age'] = zscore(Data[time]['Age'])
				Data[time]['Trial'] = zscore(Data[time]['Trial'])
				Data[time]['Vol'] = Data[time]['Vol'] *1e12 # otherwise data poorly scaled.
				# Data[time] = Data[time].loc[Data[time]['Vol']!=-np.inf]  # account for 0 power
				# Data[time] = Data[time].loc[Data[time]['Vol']>-300]

				####----Model after discussion with Michael and Alex in Jan 2019----####

				if robust_baseline:
					formula = "Vol ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces + (0 + Trial | Subject) + (1 | Subject/Run)"  #"Pow ~ Faces  + Age + Faces*Age + Trial + Rewarded + Rewarded*Faces"
				else:
					formula = "Vol ~ Faces  + Age + Faces*Age + Trial + Pe + Pe*Faces + (0 + Trial | Subject) + (1 | Subject/Run)" #"Pow ~ Faces  + Age + Faces*Age + Trial + Pe + Pe*Faces"

				#vcf = {"Run": "0+C(Run)"}
				#groups = Data[(freq,time)]["Subject"].values
				#ref = "~Trial"
				#md = sm.MixedLM.from_formula(formula = formula, data = Data[(freq,time)], vc_formula = vcf, groups = "Subject", re_formula = ref).fit(reml=False)

				md = Lmer(formula ,data=Data[time])
				md_output = md.fit(REML=False)

				# model in lme4:
				# robust_baseline <- lmer(Pow_dB ~ 1 + Faces * Age_z + Trial_z + Rewarded * Faces + (1 + Trial_z | Subject/Run), dataset)
				# pe_basic <- lmer(Pow_dB ~ 1 + Faces * Age_z + Trial_z + Pe_z * Faces + (1 + Trial_z | Subject/Run), dataset)


				####----Model tested in 2018-----####
				#### md = smf.mixedlm("Pow ~ Trial + Pe + Age + Age*Pe + Faces + Faces*Age*Pe + Faces*Pe ", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Pe  ").fit(reml=False)
				#### this is equivalent to this in R's lme4: Pow ~ 1 + Pe + Age + Age*pe + Faces + Faces*Pe + Faces*Age*Pe + (1+Pe | Subject)
				#### note only full ML estimation will return AIC

				RegStats[(chname, time, 'parameters')] = md_output['Estimate'] #md.params.copy()
				RegStats[(chname, time, 'zvalue')] = md_output['T-stat']
				RegStats[(chname, time, 'llf')] = md.logLike
				RegStats[(chname, time, 'pvalues')] = md_output['P-val']
				RegStats[(chname, time, '2.5_ci')] = md_output['2.5_ci']
				RegStats[(chname, time, '97.5_ci')] = md_output['97.5_ci']
				RegStats[(chname, time, 'aic')] = md.AIC

			if robust_baseline:
				fn = datapath + '/Group/' + chname + '_Evoke_' + Event_types + '_RobustBaseline' + '_mlm.stats'
			if not robust_baseline:
				fn = datapath + '/Group/' + chname + '_Evoke_' + Event_types + '_fullmodel' + '_mlm.stats'
			save_object(RegStats, fn)

		# if (parameters =='Value') & (Event_types == 'clock'):
		# 	RegStats = dict()

		# 	for freq in [freqs]:
		# 		Data[(freq)] = Data[(freq)].dropna()
		# 		#Data[(freq)]=Data[(freq)][Data[(freq)]['Pow']<300]
		# 		Data[(freq)]=Data[(freq)][Data[(freq)]['Value']!=0]
		# 		Data[(freq)]['Value'].subtract(Data[(freq)]['Value'].mean()) #grand mean centering
		# 		Data[(freq)]['Age'].subtract(Data[(freq)]['Age'].mean())

		# 		#vcf = {"Trial": "0+C(Trial)"} #fit nested random effect for trial, but takes FOREVER to run....
		# 		#,vc_formula = vcf
		# 		md = smf.mixedlm("Pow ~ Trial + Value + Age + Age*Value + Faces + Faces*Age*Value + Faces*Value ", Data[(freq)], groups=Data[(freq)]["Subject"], re_formula="~Value ").fit(reml=False)

		# 		RegStats[(chname, freq, 'parameters')] = md.params.copy()
		# 		RegStats[(chname, freq, 'zvalue')] = md.tvalues.copy()
		# 		RegStats[(chname, freq, 'llf')] = md.llf.copy()
		# 		RegStats[(chname, freq, 'pvalues')] = md.pvalues.copy()
		# 		RegStats[(chname, freq, 'conf_int')] = md.conf_int().copy()
		# 		RegStats[(chname, freq, 'aic')] = md.aic

		# 		fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'
		# 		save_object(RegStats, fn)


		# if (parameters =='Value') & (Event_types == 'RT'):
		# 	RegStats = dict()

		# 	for freq in [freqs]:
		# 		for time in times:
		# 			Data[(freq,time)] = Data[(freq,time)].dropna()
		# 			#Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Pow']<300]
		# 			Data[(freq,time)]=Data[(freq,time)][Data[(freq,time)]['Value']!=0]
		# 			Data[(freq,time)]['Value'].subtract(Data[(freq,time)]['Value'].mean()) #grand mean centering
		# 			Data[(freq,time)]['Age'].subtract(Data[(freq,time)]['Age'].mean())

		# 			md = smf.mixedlm("Pow ~ Trial + Value + Age + Age*Value + Faces + Faces*Age*Value + Faces*Value ", Data[(freq,time)], groups=Data[(freq,time)]["Subject"], re_formula="~Value").fit(reml=False)

		# 			RegStats[(chname, freq, time, 'parameters')] = md.params.copy()
		# 			RegStats[(chname, freq, time, 'zvalue')] = md.tvalues.copy()
		# 			RegStats[(chname, freq, time, 'llf')] = md.llf.copy()
		# 			RegStats[(chname, freq, time, 'pvalues')] = md.pvalues.copy()
		# 			RegStats[(chname, freq, time, 'conf_int')] = md.conf_int().copy()
		# 			RegStats[(chname, freq, time, 'aic')] = md.aic

		# 		fn = datapath + '/Group/' + chname + '_' + str(freqs) + 'hz_' + Event_types + '_mlm.stats'
		# 		save_object(RegStats, fn)

		return RegStats

	else:
		return Data

	#left over
	#fn = '/home/despoB/kaihwang/Clock/Group/' + chname + '_clock' + '_tfr'
	#save_object(Data, fn)

	### plot to look at distribution
	#%matplotlib qt
	#g=sns.jointplot('Pow','Pe',data=D])



def compile_evoke_reg(trial_type = 'feedback'):
	''' assemble regression results for evoke reg, channel by channel...'''

	### load gorup power ave epoch template
	# trials TFR time locked to clock onset
	#clock_power = mne.time_frequency.read_tfrs('Data/group_clock_power-tfr.h5')
	# trials TFR time locked to response onset
	#RT_power = mne.time_frequency.read_tfrs('Data/group_RT_power-tfr.h5')
	# trials TFR time locked to feedback onset
	regdatadir = '/data/backed_up/kahwang/Clock/Group/' #datapath + 'Group/'
	freqs = np.arange(2,62,2)#np.loadtxt('/home/kahwang/bin/Clock/fullfreqs')
	channels_list = np.load('/data/backed_up/kahwang/bin/Clock/channel_list.npy')
	metrics = ['zvalue', 'pvalues', 'parameters']
	parameters = ['(Intercept)','FacesFear', 'FacesHappy', 'Age', 'Trial', 'Rewarded', 'FacesFear:Age', 'FacesHappy:Age', 'FacesFear:Rewarded', 'FacesHappy:Rewarded']

	if trial_type == 'feedback':
		template = read_object('/home/kahwang/bin/Clock/ave_evoke_template')#[0]

		#setup var
		Output ={}
		for metric in metrics:
			Output[metric]={}
			for param in parameters:
				Output[metric][param] = np.zeros(template.data.shape)

		for ch in channels_list:
			pick_ch = mne.pick_channels(channels_list.tolist(),[ch]) #has to be list, annoying

			try:
				fn = regdatadir + '%s_Evoke_feedback_RobustBaseline_mlm.stats' %(ch)
				ds = read_object(fn)
			except:
				continue

			for it, t in enumerate(template.times[250:]): #no negative timepoint

				# Intercept as mean power
				for metric in metrics:
					for param in parameters:

						Output[metric][param][pick_ch,it+250] = ds[(str(ch), t, metric)][param]


		# FRD correction to create significant mask
		# if fdr_correction:
		# 	Sig_mask = {}

		# 	for param in parameters:
		# 		Sig_mask[param] = np.zeros(template.data.shape)==1

		# 	# Sig_mask = {
		# 	# 'Pe' : np.zeros(template.data.shape)==1,
		# 	# 'Age' : np.zeros(template.data.shape)==1,
		# 	# 'Age:Pe' : np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Fear]': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Happy]': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Happy]:Pe': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Fear]:Pe': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Happy]:Age': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Fear]:Age': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Happy]:Age:Pe': np.zeros(template.data.shape)==1,
		# 	# 'Faces[T.Fear]:Age:Pe': np.zeros(template.data.shape)==1
		# 	# }

		# 	for param in parameters:
		# 		ps = Output['pvalues'][param][:,:,250:]
		# 		ps[ps==0]=np.inf #missing values
		# 		Sig_mask[param][:,:,250:] = np.reshape(multipletests(ps.flatten(),alpha=0.05,method='fdr_by')[0], ps.shape)

		# 		#Sig_mask[param] = pmask
		# 		#param : pmask}
		# 	return Output, template, Sig_mask
		# else:
		return Output, template



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
	#chname='MEG0211'
	#hz=2

	##### To write out channel by channel epoch:
	# channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')

	# for chname in channels_list:
	# 	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')
	# 	fn = 'Data/fb_Epoch_ch%s' %chname
	# 	save_object(fb_Epoch, fn)
	# 	fn = 'Data/Baseline_Epoch_ch%s' %chname
	# 	save_object(Baseline_Epoch, fn)
	# 	fn = 'Data/dl_ch%s' %chname
	# 	save_object(dl, fn)

	#### read chn by chn epoch then do TFR regression:


	#fullfreqs = np.logspace(*np.log10([2, 50]), num=20)
	#for hz in np.arange(2,62,2):
	#hz=2
		#Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, global_model = True, robust_baseline = True, parameters='Pe')
	#	Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, global_model = True, robust_baseline = False, parameters='Pe')
	#save_object(Feedbackdata, 'Feedbackdata_exampchan_hz2_inDict')

	#Feedbackdata = read_object('Feedbackdata_exampchan_hz2_inDict')


	#RegFeedbackdata_base = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, global_model = True, robust_baseline = True, parameters='Pe')
	#RegFeedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = True, global_model = True, robust_baseline = False, parameters='Pe')
	#fullfreqs = np.logspace(*np.log10([2, 50]), num=20)

	#chname, hz = raw_input().split()
	#hz = np.float(hz)
	#run_TFR_regression(chname, hz)

	#### exaime distribution
	# chname='MEG2232'
	# freqs = np.logspace(*np.log10([2, 50]), num=20)#10.88

	# for hz in freqs:
	# 	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')
	# 	Feedbackdata = TFR_regression(fb_Epoch, Baseline_Epoch, chname, hz, 'feedback', do_reg = False, parameters='Pe')
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
	# ave across all sensors
	#avepower.plot(picks = None, mode=None, tmin=0, vmin=-3, vmax=3, show=True, title = 'All Sensors', yscale='auto', combine ='mean')
	# ave power topo plot
	# avepower.plot_topo(mode=None, tmin=0, vmin=-3, vmax=3, show=True, title = 'Ave Power', yscale='auto')


	### Get Mosaic Mask
	#get_mosaic_mask()

	### Hiearchical clustering
	#get_cluster()




	#### Restart in Nov 2019, try to compile evoke response. Run model on evoke data first before TFR. The purpose is to get a sense if there is any signal varying with model parm. Specifically rewarded vs unrewarded.
	# channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')

	# for chname in channels_list:
	# 	#chname = channels_list[0]
	# 	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')

	# 	demographic = pd.read_csv('/data/backed_up/kahwang/bin/Clock/subinfo_db', sep='\t')
	# 	global_model_df = pd.read_csv('/data/backed_up/kahwang/bin/Clock/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics_reorganized.csv')

	# 	Feedbackdata = Evoke_regression(fb_Epoch, Baseline_Epoch, chname, demographic, global_model_df, 'feedback', do_reg = True, global_model = True, robust_baseline = True, parameters='Pe')

	#save_object(Feedbackdata, 'Feedbackdata_exampchan_evoke_inDict')


	### plot ave evoke
	# subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
	# Event_types = 'feedback'
	# for s, subject in enumerate(subjects):
	#
	# 	if s == 0:
	# 		e = raw_to_epoch(subject, [Event_types])
	# 		a = e[Event_types].apply_baseline((-1,-0)).average()
	# 		#e_sum = np.zeros(a.data.shape)
	# 		e_sum = a.data
	# 	# create epoch with one channel of data
	# 	else:
	# 		e = raw_to_epoch(subject, [Event_types])
	# 		a = e[Event_types].apply_baseline((-1,-0)).average()
	# 		e_sum = e_sum + a.data
	#
	# e_ave = a.copy()
	# #e_ave.nave = s+1
	# e_ave.data = e_sum / (s+1)
	# e_ave.plot_topo()
	#
	# ### get group evoke
	#
	# evoke_reg, evoke_ave = compile_evoke_reg()
	# evoke_fit = evoke_ave.copy()
	# evoke_fit.data = evoke_reg['zvalue']['Rewarded']
	# evoke_fit.crop(tmin=0, tmax=1)
	# #evoke_fit.plot_topo(scalings={'grad':1, 'mag':1} )
	# evoke_fit.plot_topomap(scalings={'grad':1, 'mag':1}, times = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], vmax=5, vmin=-5 )
	# evoke_fit.plot_topomap(scalings={'grad':1, 'mag':1}, times = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6], vmax=5, vmin=-5, ch_type = 'grad' )
	# evoke_fit.plot_topomap(scalings={'grad':1, 'mag':1}, times = [0, 0.05, 0.1], vmax=5, vmin=-5, ch_type = 'grad' )
	# # template.plot_topomap(times = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
	# # template.plot_topomap(times = [-0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.5, 0])


	#### Notes from Dec 2020
	# Will be looking into doing source porjection. All sensor space data are properlly epoched and preprocessed.
	# Data should be at /data/backed_up/kahwang/Clock
	# subjects and channels in /home/kahwang/bin/Clock/subjects
	# Examples to consult:
	# https://mne.tools/stable/auto_examples/time_frequency/plot_compute_source_psd_epochs.html#sphx-glr-auto-examples-time-frequency-plot-compute-source-psd-epochs-py
	# foward model: https://mne.tools/stable/overview/implementation.html#ch-forward
	# defining landmark registration https://mne.tools/stable/auto_tutorials/source-modeling/plot_source_alignment.html#id7
	# compute inverse: https://mne.tools/stable/auto_tutorials/source-modeling/plot_mne_dspm_source_localization.html#sphx-glr-auto-tutorials-source-modeling-plot-mne-dspm-source-localization-py
	# use freesurfer label https://mne.tools/stable/auto_examples/time_frequency/plot_compute_source_psd_epochs.html#sphx-glr-auto-examples-time-frequency-plot-compute-source-psd-epochs-py
	# walk through https://mne.tools/stable/overview/cookbook.html

	# ### Try to compile datframe CSV for Michael. Only loading evoke data
	# channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
	# outputpath = '/data/backed_up/kahwang/Clock/csv_data/'
	#
	# for chname in [channels_list[0]]:
	# 	print(chname)
	# 	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, 'feedback')
	#
	# 	times = fb_Epoch[list(fb_Epoch.keys())[0]]['feedback'].times
	#
	# 	for itime, time in enumerate(times):
	#
	# 		fn = '/data/backed_up/kahwang/Clock/csv_data/ch_%s/time_%s/' %(chname, time)
	#
	# 		if not os.path.exists(fn):
	# 			os.makedirs(fn)
	#
	# 		df = pd.DataFrame()
	# 		for s in fb_Epoch.keys():
	#
	# 			Total_trianN = fb_Epoch[s]['feedback'].get_data().shape[0]
	# 			run = np.repeat(np.arange(1,9),63)
	# 			trials = np.arange(1,505)
	#
	# 			# reject trials
	# 			run = np.delete(run, dl[s], axis=0)
	# 			trials = np.delete(trials, dl[s], axis=0)
	#
	# 			try:
	# 				tdf = pd.DataFrame()
	# 				#for iTrial in np.arange(0,Total_trianN):
	# 				tdf.loc[:, 'Signal'] = fb_Epoch[s]['feedback'].get_data()[:,:,itime].squeeze()
	# 				tdf.loc[:, 'Subject'] = s
	# 				tdf.loc[:, 'Channel'] = chname
	# 				tdf.loc[:, 'Run'] = run
	# 				tdf.loc[:, 'Trial'] = trials
	# 				tdf.loc[:, 'Event'] = 'feedback'
	# 				tdf.loc[:, 'Time'] = time
	#
	# 				df = pd.concat([df, tdf])
	#
	# 			except:
	# 				fn = "check %s's trial number" %s
	# 				print(fn)
	# 				continue
	#
	# 		fn = '/data/backed_up/kahwang/Clock/csv_data/ch_%s/time_%s/ch-%s_time-%s.csv' %(chname, time, chname, time)
	# 		df.to_csv(fn)
	#
	#
	# 	#here "dl" are the bad trials list that we need to remove from analyses

	## Start work on doing inverse calculations
	#print('so hard')


	### topoplot for Alex's model fit
	# csv from r
	#load("/Volumes/rdss_kahwang/tmp/plots/rt_rt/meg_medusa_rt_predict_output_all.Rdata")
	#write.csv(rdf,file="/Volumes/rdss_kahwang/tmp/meg_medusa_rt_predict_output_all.csv")
	#load("/Volumes/rdss_kahwang/tmp/plots/rt_rt/meg_medusa_rt_predict_output_all.Rdata")
	#write.csv(ddf,file="/Volumes/rdss_kahwang/tmp/meg_medusa_rt_decode_output_all.csv")

	rdf = pd.read_csv('~/RDSS/tmp/meg_medusa_rt_predict_output_all.csv')
	ddf = pd.read_csv('~/RDSS/tmp/meg_medusa_rt_decode_output_all.csv')

	channels_list = np.load('/data/backed_up/kahwang/bin/Clock/channel_list.npy')
	template = read_object('/home/kahwang/bin/Clock/ave_evoke_template')

	dataset = [rdf, ddf]
	dfilename = ['predict', 'decode']
	#sig = ['NS', '0']
	#threshold = ['threshold', 'unthreshold']

	for id, dat in enumerate(dataset):
		terms = dat.term.unique()
		times = dat.t.unique()

		# get rid of () and : from alex's dataframe, otherwise files cant be generated"
		#dat.term = dat.term.str.replace('(', '_')
		#dat.term = dat.term.str.replace(')', '_')
		#dat.term = dat.term.str.replace(':', '_')

		for term in terms:
			df = dat.loc[dat.term==term]

			#create data matrix
			n_ch = np.unique(dat['sensor']).shape[0] #306 channels
			n_timepoint = np.unique(dat['t']).shape[0] #84 time points
			data = np.zeros((n_ch,n_timepoint)) #channel by time
			data_mask = np.zeros((n_ch,n_timepoint))

			for index, row in df.iterrows():
				#print('yes')

				# find channel
				ch = "MEG{:0>4d}".format(row['sensor'])
				pick_ch = mne.pick_channels(template.ch_names,[ch])[0]

				# find time
				tidx = np.where(times==row.t)[0][0]
				data[pick_ch, tidx] = row.estimate

				# mask of significant data points
				if row['p, FDR-corrected'] != 'NS':
					data_mask[pick_ch, tidx] = 1

			# mask out insig data
			vmax = np.percentile(data, 98)
			data_mask = np.array(data_mask, dtype=bool)
			data[data_mask==0] = 0

			# create evoke object for plotting
			info = mne.create_info(template.ch_names, ch_types=template.get_channel_types(), sfreq=20.8333333)
			info['chs'] = template.info['chs']
			evoked_array = mne.EvokedArray(data, info, tmin=min(dat.t))

			#evoked_array.plot_joint(picks='grad', times = evoked_array.times[np.arange(0,84,4)])
			#sc=dict(eeg=1, grad=1, mag=1)
			#sc=dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=7)
			fn = term + '-2.97_to_-1.72s'
			f=evoked_array.plot_topomap(times = evoked_array.times[np.arange(0,28,2)],ch_type='grad', sensors = False, cmap = 'RdBu_r', vmin=-1*vmax, vmax=vmax,
				colorbar = True, scalings = 1, units='statistic', title= fn, show = False, contours = 0)
			fn = 'AlexTopoPlots/'+ dfilename[id]+'_' + term + '-2.97_to_-1.72s' + '.png'
			f.savefig(fn)
			plt.close('all')
			del f
			fn = term + '-1.63_to_-.38s'
			f=evoked_array.plot_topomap(times = evoked_array.times[np.arange(28,56,2)],ch_type='grad', sensors = False, cmap = 'RdBu_r', vmin=-1*vmax, vmax=vmax,
				colorbar = True, scalings = 1, units='statistic', title= fn, show = False, contours = 0)
			fn = 'AlexTopoPlots/'+ dfilename[id]+'_' + term + '-1.63_to_-.38s' + '.png'
			f.savefig(fn)
			plt.close('all')
			del f
			fn = term + '-0.38_to_.96s'
			f=evoked_array.plot_topomap(times = evoked_array.times[np.arange(56,84,2)],ch_type='grad', sensors = False, cmap = 'RdBu_r', vmin=-1*vmax, vmax=vmax,
				colorbar = True, scalings = 1, units='statistic', title= fn, show = False, contours = 0)
			fn = 'AlexTopoPlots/'+ dfilename[id]+'_' + term + '-0.38_to_.96s' + '.png'
			f.savefig(fn)
			plt.close('all')
			del f



	#for ch in







	## End of script
