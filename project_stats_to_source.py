# test if we can project sensor level stats to source space
#### script to plot model outputs from Alex and Michael
from Clock import raw_to_epoch
import numpy as np
import pandas as pd
import mne as mne
import pyreadr
import matplotlib.pyplot as plt
from nilearn import datasets, plotting
plt.ion()
import os
from nilearn.image import math_img, new_img_like
from mne.datasets import sample, fetch_fsaverage
from mne.minimum_norm import apply_inverse, read_inverse_operator
from nilearn.plotting import plot_glass_brain

#mkae paths global
datapath = '/data/backed_up/kahwang/Clock/'
save_path='/data/backed_up/kahwang/Clock/'


def create_param_tfr(sdf, fdf, term, threshold = False, threshold_se = True, se=4, no_threshold = False):
	''' function to create mne objet for ploting from R data frame
	two inputs, sdf: the dtaframe
	term, the variable for plotting
	You also have the option to threshold the data, if True, then only data with fdr_p<0.05  will be saved
	'''

	# creat TFR epoch object for plotting. Use the "info" in this file for measurement info
	template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
	# the data array in this template is 306 ch by 20 freq by 464 time

	# create custom tfr data array
	try:
		tdf=sdf.loc[sdf.term==term]
	except:
		tdf = df

	time = np.sort(tdf.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
	freq = np.sort(tdf.Freq.unique())
	new_data = np.zeros((306, len(freq), len(time)))

	if threshold:
		fdf=fdf.loc[fdf.term==term]
	if threshold_se:
		fdf=fdf.loc[fdf.term==term]

	# now plut in real stats into the dataframe
	for index, row in tdf.iterrows():
		t = row.Time
		f = row.Freq
		try:
			ch = row.level
			ch = 'MEG'+ '{:0>4}'.format(ch)
		except:
			ch = row.Sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		#print(ch)
		ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])

		try:
			if threshold & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['p_fdr'].values<0.05):
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
			elif threshold & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['p_fdr'].values>0.05):
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = 0
			elif threshold_se & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['std.error'].values*se<abs(row.estimate)):
				#print('yes')
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
			# elif threshold_se & (row.estimate<0) & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['std.error'].values*se*-1<row.estimate):
			# 	#print('yes')
			# 	new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
			else:
				#print('did not survive threshold')
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = 0
			if no_threshold:
				#print('no threshold')
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
		except:
			print('zero')
			#new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate

	new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

	return new_tfr


def create_SE_tfr(sdf, fdf, term):

	# creat TFR epoch object for plotting. Use the "info" in this file for measurement info
	template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
	# the data array in this template is 306 ch by 20 freq by 464 time

	# create custom tfr data array
	tdf=sdf.loc[sdf.term==term]
	time = np.sort(tdf.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
	freq = np.sort(tdf.Freq.unique())
	new_data = np.zeros((306, len(freq), len(time)))

	fdf=fdf.loc[fdf.term==term]

	# now plut in real stats into the dataframe
	for index, row in tdf.iterrows():
		t = row.Time
		f = row.Freq
		try:
			ch = row.level
			ch = 'MEG'+ '{:0>4}'.format(ch)
		except:
			ch = row.Sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		#print(ch)
		ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])

		try:
				new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['std.error'].values
		except:
			print('zero')
			#new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate

	new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

	return new_tfr


def extract_sensor_random_effect(rdata, alignment):
	''' take r data frame, extract sensor level random effect, remeber to specity the alignment 'rt' or 'clock' '''
	df = rdata[None]

	# save a different df that contains the fix effect so we can get the p values
	fdf = df.loc[df.effect=='fixed']
	fdf = fdf.loc[fdf.alignment==alignment]
	fdf.Freq = fdf.Freq.astype('float')

	# cut it down to only include sensor data
	df = df.loc[df.group=='Sensor']
	#select random effects
	df = df.loc[df.effect=='ran_coefs']
	#which alignment?
	df = df.loc[df.alignment==alignment]
	df.Freq = df.Freq.astype('float')

	return df, fdf


def create_tfr(df):
	''' function to create mne objet for ploting from R data frame from Michael
	'''
	df['Freq'] = df.Freq.str.lstrip('f_')
	df['Freq'] = df.Freq.astype('float')

	# creat TFR epoch object for plotting. Use the "info" in this file for measurement info
	template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
	# the data array in this template is 306 ch by 20 freq by 464 time

	# create custom tfr data array
	time = np.sort(df.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
	freq = np.sort(df.Freq.unique())
	new_data = np.zeros((306, len(freq), len(time)))
	# now plut in real stats into the dataframe
	for index, row in df.iterrows():
		t = row.Time
		f = row.Freq
		try:
			ch = row.level
			ch = 'MEG'+ '{:0>4}'.format(ch)
		except:
			ch = row.Sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		#print(ch)
		ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
		new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.Estimate

	new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

	return new_tfr


def create_fixed_effect_tfr(inputdf, reward = 'Omission', regressor = 'RT_t', effect = 'zhigh', threshold = True, se = 4):
	''' function to create mne objet for ploting from fixed effect data frame from Alex
	'''
	#df['Freq'] = df.Freq.str.lstrip('f_')
	inputdf['Freq'] = inputdf.Freq.astype('float')
	df = inputdf.loc[(inputdf['reward_t']==reward) & (inputdf['regressor']==regressor)]
	# creat TFR epoch object for plotting. Use the "info" in this file for measurement info
	template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
	# the data array in this template is 306 ch by 20 freq by 464 time

	# create custom tfr data array
	time = np.sort(df.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
	freq = np.sort(df.Freq.unique())
	new_data = np.zeros((306, len(freq), len(time)))
	# now plut in real stats into the dataframe
	for index, row in df.iterrows():
		t = row.Time
		f = row.Freq
		try:
			ch = row.level
			ch = 'MEG'+ '{:0>4}'.format(ch)
		except:
			ch = row.Sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		#print(ch)
		ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
		new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row[effect]

	new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

	return new_tfr


subjects_dir = '/data/backed_up/kahwang/Clock/'
subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
#channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
fname_src_fsaverage = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'
src_fs = mne.read_source_spaces(fname_src_fsaverage)

def project_stats_to_source(stats_evoke, outputname):
	ave_img_data = np.zeros((82,97,85,1))
	i=0
	for sub in subjects:
		try:
			
			# volumn source space
			src_fn = subjects_dir +'%s/MEG/%s-vol-src.fif' %(sub, sub)
			vol_src = mne.read_source_spaces(src_fn)
			
			# volumn forward solution
			vol_fwd_fn = subjects_dir + '%s/MEG/%s-vol-fwd.fif' %(sub, sub)
			vol_fwd = mne.read_forward_solution(vol_fwd_fn)

			#noise cov
			Event_types = ['ITI']
			ITI_ep = raw_to_epoch(sub, Event_types)
			noise_cov = mne.compute_covariance(ITI_ep['ITI'], tmin = 0, tmax=1, method=['empirical'], rank = 'info', verbose=True)
			Event_types = ['feedback']
			fb_ep = raw_to_epoch(sub, Event_types)
			fb_ep = fb_ep['feedback'].apply_baseline((None,None))
			data_cov = mne.compute_covariance(fb_ep, tmin=0, tmax=0.8, method='empirical', rank = 'info')

			#make vol inverse
			raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(sub, sub)
			raw = mne.io.read_raw(raw_fname)
			raw.info['bads'] = np.array(raw.ch_names)[np.array(raw.get_channel_types())=='mag'].tolist()
			#inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, vol_fwd, noise_cov, loose="auto", depth=0.5)
			filter = mne.beamformer.make_lcmv(raw.info, vol_fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori=None, reduce_rank = False, rank=None, depth=0.2)
			#need to investigate the effect of reduced rank
			#filter_fn = subjects_dir + '%s/MEG/%s-vol-beamformer-lcmv.h5' %(sub, sub)
			#filter = mne.beamformer.read_beamformer(filter_fn)

			#stc = mne.minimum_norm.apply_inverse(stats_evoke, inverse_operator, method = "dSPM")
			stc = mne.beamformer.apply_lcmv(stats_evoke, filter)
			img = stc.as_volume(vol_src)
			# img_mean = img.get_fdata()[img.get_fdata()!=0].mean()
			# fns = "img1 - %s" %img_mean
			img = math_img("img1/1e10", img1 = img) 
			fn = "/data/backed_up/kahwang/Clock/Source/%s_%s.nii.gz" %(sub, outputname)
			img.to_filename(fn)

			morph = mne.compute_source_morph(vol_src, subject_from=str(sub), subjects_dir=subjects_dir, src_to=src_fs, verbose=True)
			img_fsaverage = morph.apply(stc, mri_resolution=2, output='nifti1')
			img_fsaverage = math_img("img1/1e10", img1 = img_fsaverage) 
			fn = "/data/backed_up/kahwang/Clock/Source/%s_%s_mni.nii.gz" %(sub, outputname)
			img_fsaverage.to_filename(fn)	

			ave_img_data = ave_img_data + img_fsaverage.get_fdata()
			i=i+1
		except:
			continue

	img_average = new_img_like(img_fsaverage, ave_img_data/i)
	fn = "/data/backed_up/kahwang/Clock/Source/ave_%s_mni.nii.gz" %outputname
	img_average.to_filename(fn)



### Prepare directly project to fsaverage.
raw_fname = subjects_dir + '11350/MEG/11350_clock_run1_dn_ds_sss_raw.fif'
raw = mne.io.read_raw(raw_fname)
src = os.path.join(subjects_dir, 'fsaverage/bem', 'fsaverage-vol-5-src.fif')
bem = os.path.join(subjects_dir, 'fsaverage/bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
fs_fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, meg=True, eeg=False, n_jobs=32, verbose=True)

#ave_img_data = np.zeros((82,97,85,1))
#i=0
# fp_eps = []
# iti_eps = []
# for sub in subjects:
# 	try:
# 		#noise cov
# 		Event_types = ['ITI']
# 		ITI_ep = raw_to_epoch(sub, Event_types)
# 		iti_eps.append(ITI_ep['ITI'])
# 		#noise_cov = mne.compute_covariance(ITI_ep['ITI'], tmin = 0, tmax=1, method=['empirical'], rank = 'info', verbose=True)
# 		Event_types = ['feedback']
# 		fb_ep = raw_to_epoch(sub, Event_types)
# 		fb_ep = fb_ep['feedback'].apply_baseline((None,None))
# 		fp_eps.append(fb_ep)
# 		#data_cov = mne.compute_covariance(fb_ep, tmin=0, tmax=0.8, method='empirical', rank = 'info')
# 	except:
# 		continue
# itiep = mne.concatenate_epochs(iti_eps, on_mismatch='ignore')	
# del iti_eps
# noise_cov = mne.compute_covariance(itiep, tmin = 0, tmax=1, method=['empirical'], rank = 'info', verbose=True)
# del itiep
# fbep = mne.concatenate_epochs(fp_eps, on_mismatch='ignore')	
# del fp_eps
# data_cov = mne.compute_covariance(fbep, tmin=0, tmax=0.8, method='empirical', rank = 'info')
# del fbep #save memory
# data_cov.save("/home/kahwang/bkh/Clock/Source/data_cov.fif")
# noise_cov.save("/home/kahwang/bkh/Clock/Source/noise_cov.fif")

data_cov = mne.read_cov("/home/kahwang/bkh/Clock/Source/data_cov.fif")
noise_cov = mne.read_cov("/home/kahwang/bkh/Clock/Source/noise_cov.fif")
raw.info['bads'] = np.array(raw.ch_names)[np.array(raw.get_channel_types())=='mag'].tolist() #get rid of mageometers
filter = mne.beamformer.make_lcmv(raw.info, fs_fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori=None, reduce_rank = False, rank=None, depth=0.2)



# 3/18/2022
# Early theta to reward omission: …/meg/plots/wholebrain/reward/meg_ddf_wholebrain_reward.rds
# term == "reward_t" & t >= 0.2 & t <= 0.4 & Freq >= "5" &  Freq <= "8.4". 
# Since we are looking at synchronization to omission, we can just get the negative effects of reward.
# Late beta to signed PEs: …/meg/plots/wholebrain/signed_pe_rs/meg_ddf_wholebrain_signed_pe_rs.rds
# term == "pe_max" & regressor == "signed_pe_rs" & t >= 0.4 & t <= 0.75 &  Freq >= "8.4" &  Freq <= "20"
# Here, we also just need negative effects of RPE.
 
signed_pe_rs_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_signed_pe_rs.rds') 
signed_pe_rs_df, signed_pe_rs_fdf = extract_sensor_random_effect(signed_pe_rs_rdata, 'rt')
signed_pe_rs_tfr = create_param_tfr(signed_pe_rs_df, signed_pe_rs_fdf, 'pe_max', se =0)
signed_pe_rs_tfr.plot_topo(yscale='log', picks='grad')
avdata = np.mean(signed_pe_rs_tfr.data[:,8:14,24:30], axis=(1,2))[:,np.newaxis]
avdata[avdata>0] = 0
signed_pe_rs_ave = mne.EvokedArray(data=avdata, info = signed_pe_rs_tfr.info)
signed_pe_rs_ave.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(signed_pe_rs_ave, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/signed_pe_rs_fsaverage.nii.gz")


reward_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_reward.rds') 
reward_df, reward_fdf = extract_sensor_random_effect(reward_rdata, 'rt')
reward_tfr = create_param_tfr(reward_df, reward_fdf, 'reward_t', se =0)
reward_tfr.plot_topo(yscale='log', picks='grad')
avdata = np.mean(reward_tfr.data[:,5:9,19:24], axis=(1,2))[:,np.newaxis]
avdata[avdata>0] = 0
reward_ave = mne.EvokedArray(data=avdata, info = reward_tfr.info)
#reward_ave.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(reward_ave, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/reward_fsaverage.nii.gz")



# before 2/28/
## Alex said:
## I guess (i) entropy change in the late low beta suppression as the model of interest (7-17 Hz band at 400-750 ms) and as a sanity check contrast, (ii) theta suppression to reward omission (5-7 Hz and 0.2-0.4 s). 
#entropy_change_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_entropy_change.rds') #whole brain data
#entropy_change_rt_df, entropy_change_rt_fdf = extract_sensor_random_effect(entropy_change_rdata, 'rt')
#entropy_change_t_rt_tfr = create_param_tfr(entropy_change_rt_df, entropy_change_rt_fdf, 'entropy_change_t', se =0)
#entropy_change_t_rt_tfr.plot_topo(yscale='log', picks='grad')
#avdata = np.mean(entropy_change_t_rt_tfr.data[:,7:13,:], axis=(1))
#entropy_change_ave = mne.EvokedArray(data=avdata, info = entropy_change_t_rt_tfr.info)
#entropy_change_ave.plot_topomap(ch_type='grad')

#wholebrain_zstats_random_slope = pyreadr.read_r(datapath + 'meg_rdf_wholebrain_zstats_random_slope.rds')
#zstats_df = wholebrain_zstats_random_slope[None]
#Omission_RT_t_zdiff = create_fixed_effect_tfr(zstats_df, 'Omission', 'RT_t' ,'zdiff')
#Omission_RT_t_zdiff.plot_topo(yscale='log', picks='grad')
#avdata = np.mean(Omission_RT_t_zdiff.data[:,5:8], axis=(1))
#omission_stats = mne.EvokedArray(data=avdata, info = Omission_RT_t_zdiff.info)
#omission_stats.plot_topomap(ch_type='grad')
#project_stats_to_source(entropy_change_ave, "entropy_change")
#project_stats_to_source(omission_stats, "reward_omission")

#stc = mne.beamformer.apply_lcmv(entropy_change_ave, filter)
#img = stc.as_volume(mne.read_source_spaces(src))
#img = math_img("img1/1e10", img1 = img) 
#img.to_filename("/data/backed_up/kahwang/Clock/Source/entropy_change_fsaverage.nii.gz")

#stc = mne.beamformer.apply_lcmv(omission_stats, filter)
#img = stc.as_volume(mne.read_source_spaces(src))
#img = math_img("img1/1e10", img1 = img) 
#img.to_filename("/data/backed_up/kahwang/Clock/Source/reward_omission_fsaverage.nii.gz")


### 2/28/2022
#separate positive and negatives
avdata = np.mean(entropy_change_t_rt_tfr.data[:,7:13,:], axis=(1))
pos_avdata = avdata.copy()
pos_avdata[pos_avdata<0] = 0
entropy_change_ave_pos = mne.EvokedArray(data=pos_avdata, info = entropy_change_t_rt_tfr.info)
entropy_change_ave_pos.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(entropy_change_ave_pos, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/entropy_change_pos_fsaverage.nii.gz")

avdata = np.mean(entropy_change_t_rt_tfr.data[:,7:13,:], axis=(1))
neg_avdata = avdata.copy()
neg_avdata[neg_avdata>0] = 0
entropy_change_ave_neg = mne.EvokedArray(data=neg_avdata, info = entropy_change_t_rt_tfr.info)
entropy_change_ave_neg.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(entropy_change_ave_neg, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/entropy_change_neg_fsaverage.nii.gz")


avdata = np.mean(Omission_RT_t_zdiff.data[:,5:8], axis=(1))
pos_avdata = avdata.copy()
pos_avdata[pos_avdata<0] = 0
omission_stats_pos = mne.EvokedArray(data=pos_avdata, info = Omission_RT_t_zdiff.info)
omission_stats_pos.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(omission_stats_pos, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/reward_omission_pos_fsaverage.nii.gz")

avdata = np.mean(Omission_RT_t_zdiff.data[:,5:8], axis=(1))
neg_avdata = avdata.copy()
neg_avdata[neg_avdata>0] = 0
omission_stats_neg = mne.EvokedArray(data=neg_avdata, info = Omission_RT_t_zdiff.info)
omission_stats_neg.plot_topomap(ch_type='grad')
stc = mne.beamformer.apply_lcmv(omission_stats_neg, filter)
img = stc.as_volume(mne.read_source_spaces(src))
img = math_img("img1/1e10", img1 = img) 
img.to_filename("/data/backed_up/kahwang/Clock/Source/reward_omission_neg_fsaverage.nii.gz")


##### 2/21/2022 #project subject level effect onto surface. STOP!
subjects_dir = '/data/backed_up/kahwang/Clock/'
subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
#channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')
fname_src_fsaverage = subjects_dir + '/fsaverage/bem/fsaverage-vol-5-src.fif'
src_fs = mne.read_source_spaces(fname_src_fsaverage)

entropy_change_rdata = pyreadr.read_r(datapath + 'meg_echange_sensor_subject_ranefs.rds') #whole brain data
for sub in subjects:
	sdf = extract_subject_random_effect(entropy_change_rdata, sub)
	try:
		stfr = create_subject_tfr(sdf)
		#stfr.plot_topomap(ch_type='grad')
		avdata = np.mean(stfr.data[:,7:13,:], axis=1)#chn by time, collaps on freq dimension
		s_evoke = mne.EvokedArray(data=avdata, info = stfr.info)
		#s_evoke.plot_topomap(ch_type='grad')
		project_sub_to_source(s_evoke, sub, 'entropy_change')
	except:
		continue

reward_rdata = pyreadr.read_r(datapath + 'meg_reward_sensor_subject_ranefs.rds') #whole brain data
for sub in subjects:
	sdf = extract_subject_random_effect(reward_rdata, sub)
	try:
		stfr = create_subject_tfr(sdf)
		#stfr.plot_topomap(ch_type='grad')
		avdata = np.mean(stfr.data[:,5:8,:], axis=1)#chn by time, collaps on freq dimension
		s_evoke = mne.EvokedArray(data=avdata, info = stfr.info)
		#s_evoke.plot_topomap(ch_type='grad')
		project_sub_to_source(s_evoke, sub, 'reward')
	except: 
		continue


def project_sub_to_source(stats_evoke, sub, outputname):
	# ave_img_data = np.zeros((82,97,85,1))
	# i=0
	# for sub in subjects:
	# 	try:
			
	# volumn source space
	src_fn = subjects_dir +'%s/MEG/%s-vol-src.fif' %(sub, sub)
	vol_src = mne.read_source_spaces(src_fn)
	
	# volumn forward solution
	vol_fwd_fn = subjects_dir + '%s/MEG/%s-vol-fwd.fif' %(sub, sub)
	vol_fwd = mne.read_forward_solution(vol_fwd_fn)

	#noise cov
	Event_types = ['ITI']
	ITI_ep = raw_to_epoch(sub, Event_types)
	noise_cov = mne.compute_covariance(ITI_ep['ITI'], tmin = 0, tmax=1, method=['empirical'], rank = 'info', verbose=True)
	Event_types = ['feedback']
	fb_ep = raw_to_epoch(sub, Event_types)
	fb_ep = fb_ep['feedback'].apply_baseline((None,None))
	data_cov = mne.compute_covariance(fb_ep, tmin=0, tmax=0.8, method='empirical', rank = 'info')

	#make vol inverse
	raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(sub, sub)
	raw = mne.io.read_raw(raw_fname)
	raw.info['bads'] = np.array(raw.ch_names)[np.array(raw.get_channel_types())=='mag'].tolist()
	#inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, vol_fwd, noise_cov, loose="auto", depth=0.5)
	filter = mne.beamformer.make_lcmv(raw.info, vol_fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori=None, reduce_rank = False, rank=None, depth=0.2)
	#need to investigate the effect of reduced rank
	#filter_fn = subjects_dir + '%s/MEG/%s-vol-beamformer-lcmv.h5' %(sub, sub)
	#filter = mne.beamformer.read_beamformer(filter_fn)

	#stc = mne.minimum_norm.apply_inverse(stats_evoke, inverse_operator, method = "dSPM")
	stc = mne.beamformer.apply_lcmv(stats_evoke, filter)
	img = stc.as_volume(vol_src)
	# img_mean = img.get_fdata()[img.get_fdata()!=0].mean()
	# fns = "img1 - %s" %img_mean
	img = math_img("img1/1e10", img1 = img) 
	fn = "/data/backed_up/kahwang/Clock/Source/%s_%s.nii.gz" %(sub, outputname)
	img.to_filename(fn)


def extract_subject_random_effect(rdata, subject):
	''' take r data frame, extract sensor level random effect, remeber to specity the alignment 'rt' or 'clock' '''
	df = rdata[None]
	sdf = df.loc[df['subject']==str(subject)]
	return sdf

def create_subject_tfr(sdf):
	''' create single subject tfr object
	'''
	sdf['Freq'] = sdf.Freq.astype('float')

	# creat TFR epoch object for plotting. Use the "info" in this file for measurement info
	template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
	# the data array in this template is 306 ch by 20 freq by 464 time

	# create custom tfr data array
	time = np.sort(sdf.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
	freq = np.sort(sdf.Freq.unique())
	new_data = np.zeros((306, len(freq), len(time)))
	# now plut in real stats into the dataframe
	for index, row in sdf.iterrows():
		t = row.Time
		f = row.Freq
		try:
			ch = row.sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		except:
			ch = row.Sensor
			ch = 'MEG'+ '{:0>4}'.format(ch)
		#print(ch)
		ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
		new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.combined_effect #from Michael's email

	new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

	return new_tfr


#check registration
# trans_file = subjects_dir + 'trans/%s-trans.fif' %sub

# mne.viz.plot_alignment(raw.info, trans=trans_file, subject=str(sub),
#                        src=vol_src, subjects_dir=subjects_dir, dig=True,
#                        surfaces=['head-dense', 'white'], coord_frame='meg')