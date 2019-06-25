# script to regenerate trigger events
import numpy as np
import pandas as pd
import mne as mne
import glob
import os




def extract_trig(subj, run):

	dirp = '/data/backed_up/kahwang/Clock/%s/MEG' %subj
	os.chdir(dirp)

	cue_trigs = [47,56, 83, 92, 101, 110, 119, 128]
	RT_trigs = [154, 163, 190, 199, 208, 217, 226, 235]

	fn = '%s_clock_run%s_dn_ds_sss_raw.fif' %(subj, run)
	f = mne.io.read_raw_fif(fn)
	t = mne.find_stim_steps(f)


	# if run ==1:
	# 	try: 
	fnstart = glob.glob("*ds4.eve")[0][0:19]
	# 	except:
	#fnstart = 'MEG_%s_' %(subj)
	# else:
	# 	fnstart = glob.glob("*ds4.eve")[0][0:19]

	cue_t = mne.pick_events(t, include = cue_trigs) 
	fb_t = mne.pick_events(t, include = RT_trigs) 
	cue_t = np.vstack(([0,0,0],cue_t))
	fb_t = np.vstack(([0,0,0],fb_t ))

	clock = np.zeros((len(cue_t),4))  
	clock[:,0] = cue_t[:,0]                                                                                                                                                                                                  
	clock[:,1] = cue_t[:,0] * .004                                                                                                                                                                                           
	clock[:,3] = cue_t[:,2] 

	fn = fnstart + str(run) + '_clock_ds4.eve'    
	np.savetxt(fn, clock, fmt='%2f')  

	fb = np.zeros((len(fb_t),4))  
	fb[:,0] = fb_t[:,0]                                                                                                                                                                                                  
	fb[:,1] = fb_t[:,0] * .004                                                                                                                                                                                           
	fb[:,3] = fb_t[:,2] 

	fn = fnstart + str(run) + '_feedback_ds4.eve'     
	np.savetxt(fn, fb, fmt='%2f')  


if __name__ == "__main__":	

	subjs = [11253, 11313,11324,11320,11322]
	runs = [4,2,5,6,5]

	#for  sub, run in [(sub,run) for sub in subjs for run in runs]: #

	for (sub, run) in zip(subjs, runs):
		print(sub, run)
		extract_trig(sub, run)