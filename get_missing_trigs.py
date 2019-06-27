# script to regenerate trigger events
import numpy as np
import pandas as pd
import mne as mne
import glob
import os




def extract_trig(subj, run):

	dirp = '/home/kahwang/MEG_Raw/%s/' %subj
	os.chdir(dirp)

	cue_trigs = [47, 56, 83, 92, 101, 110, 119, 128]
	RT_trigs = [154, 163, 190, 199, 208, 217, 226, 235]
	#global_model_df = pd.read_csv('/home/kahwang/bin/Clock/mmclock_meg_decay_factorize_selective_psequate_fixedparams_meg_ffx_trial_statistics_reorganized.csv')
	
	try:
		fn = '%s_Clock_run%s_raw.fif' %(subj, run)
		f = mne.io.read_raw_fif(fn)

	except:
		try:
			fn = '%s_Clock_Run%s_Raw.fif' %(subj, run)
			f = mne.io.read_raw_fif(fn)
		except:
			fn = '%s_clock_run%s_raw_chpi_sss.fif' %(subj, run)	
			f = mne.io.read_raw_fif(fn)
	t = mne.find_stim_steps(f)


	# if run ==1:
	# 	try: 
	#fnstart = glob.glob("*ds4.eve")[0][0:19]
	# 	except:
	fnstart = 'MEG_%s_' %(subj)
	# else:
	# 	fnstart = glob.glob("*ds4.eve")[0][0:19]

	cue_t = mne.pick_events(t, include = cue_trigs) 
	fb_t = mne.pick_events(t, include = RT_trigs) 
	cue_t = np.vstack(([0,0,0],cue_t))
	fb_t = np.vstack(([0,0,0],fb_t ))

	clock = np.zeros((len(cue_t),4))  
	clock[1:,0] = np.round((cue_t[1:,0] - f.first_samp)/4)                                                                                                                                                                                             
	clock[:,1] = clock[:,0] * .004                                                                                                                                                                                           
	clock[:,3] = cue_t[:,2] 

	outf = '/data/backed_up/kahwang/Clock/%s/MEG/' %subj
	fn = outf + fnstart + str(run) + '_clock_ds4.eve'    
	np.savetxt(fn, clock, fmt='%2f')  

	fb = np.zeros((len(fb_t),4))  
	fb[1:,0] = np.round((fb_t[1:,0] - f.first_samp)/4)                                                                                                                                                                                                  
	fb[:,1] = fb[:,0] * .004                                                                                                                                                                                           
	fb[:,3] = fb_t[:,2] 

	fn = outf + fnstart + str(run) + '_feedback_ds4.eve'     
	np.savetxt(fn, fb, fmt='%2f')  

	return clock, fb


if __name__ == "__main__":	

	#subjs = [11335, 11320, 11313, 11281, 11252]
	subjs = [10891]
	runs = [1,2,3,4,5,6,7,8]

	for  sub, run in [(sub,run) for sub in subjs for run in runs]: #

	#for (sub, run) in zip(subjs, runs):
		print(sub, run)
		clock, fb = extract_trig(sub, run)



