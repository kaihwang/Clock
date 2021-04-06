# save trial by trial, sensor space evoke data to dataframe and csv files
from Clock import raw_to_epoch, get_dropped_trials_list, get_epochs_for_TFR_regression
import numpy as np
import pandas as pd
import mne as mne
import os

env_channel=os.getenv('channel')


# global path to data
datapath = '/proj/mnhallqlab/projects/Clock_MEG/fif_data/'
save_path = '/proj/mnhallqlab/projects/Clock_MEG/fif_data/'

if __name__ == "__main__":

    # these two files should be in the github repo
    channels_list = np.load('/proj/mnhallqlab/projects/Clock_MEG/code/channel_list.npy')
    subjects = np.loadtxt('/proj/mnhallqlab/projects/Clock_MEG/code/subjects', dtype=int)

    if env_channel is None:
        to_extract = channels_list
    else:
        to_extract = [env_channel] #single channel extraction

    
    #where to save csv files
    outputpath = '/proj/mnhallqlab/projects/Clock_MEG/csv_data_update/'

    #loop through channels
    for chname in to_extract:

        # more efficient to load chn by chn
    	print(chname)
    	fb_Epoch, Baseline_Epoch, dl = get_epochs_for_TFR_regression(chname, subjects, channels_list, 'feedback')
        #right now the baseline epoch are not used, but can do baseline correction later

    	print(dl)
        
    	times = fb_Epoch[list(fb_Epoch.keys())[0]]['feedback'].times

        # different csv for each time pt
    	for itime, time in enumerate(times):

            fn = outputpath + 'ch_%s/time_%s/' %(chname, time)
            if not os.path.exists(fn):
                os.makedirs(fn)

            df = pd.DataFrame()

            for s in fb_Epoch.keys():

                Total_trianN = fb_Epoch[s]['feedback'].get_data().shape[0]
                run = np.repeat(np.arange(1,9),63)
                trials = np.arange(0,504)

    		# reject trials
                if dl[s].size > 0:
                    run = np.delete(run, dl[s], axis=0)
                    trials = np.delete(trials, dl[s], axis=0)

                try:
                    tdf = pd.DataFrame()
                    #for iTrial in np.arange(0,Total_trianN):
                    tdf.loc[:, 'Signal'] = fb_Epoch[s]['feedback'].get_data()[:,:,itime].squeeze()
                    tdf.loc[:, 'Subject'] = s
                    tdf.loc[:, 'Channel'] = chname
                    tdf.loc[:, 'Run'] = run
                    tdf.loc[:, 'Trial'] = trials + 1
                    tdf.loc[:, 'Event'] = 'feedback'
                    tdf.loc[:, 'Time'] = time

                    df = pd.concat([df, tdf])

                except Exception as e:
                    print(e)
                    fn = "check %s's trial number" %s
                    print(fn)
                    continue

            fn = outputpath + 'ch_%s/time_%s/ch-%s_time-%s.csv' %(chname, time, chname, time)
            df.to_csv(fn)
