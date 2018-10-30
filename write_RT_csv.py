from __future__ import division;
import mne;
import scipy.io; # for matlab read
import matplotlib.pyplot as plt;
import numpy as np; #, h5py;
import pandas as pd;
from itertools import groupby #, izip; # for runlenghencode like function
import argparse
import warnings
import glob



# only saving out RT from 

source_path='/Users/kaihwang/bin/clock_analysis/meg/data/tc_orig_files/'

matfiles = glob.glob(source_path+"*.mat")


for fn in matfiles:

  ### BEHAVIORAL
  # task file
  try:
    mat   = scipy.io.loadmat(fn,struct_as_record=True);
    subj  = mat.get('subject');
    order = subj['order'][0][0];

    #trial = np.ndarray((order.size,12))
    trial = [];
    
    for t in order:
      trial.append( [i[0] if isinstance(i[0],basestring) else  i[0][0] for i in t[0][0]]  ) ## collect trial info into 504 by 12 array. 504 trials, 12 data fields

    # convert trial info into pnadas dataframe
    df_trial = pd.DataFrame(trial)
    df_trial.columns = [ 'function','run','trial','block','null','starttime','mag','scoreinc','freq','ev','RT','emotion'] # here are the 12 data fields

    # two identical df with trial info...?
    #columns=['function','run','trial','block','NA','start','mag','inc','freq','ev','rsptime','emotion']
    #df = pd.DataFrame(trial,columns=columns) # it seems to me the response time unit is in ms?

    fn = fn[:-3]+'csv'
    df_trial.to_csv(fn)

  except:
    print fn + " has error"   