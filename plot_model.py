#### script to plot model outputs from Alex and Michael
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as matplotlib
import mne as mne
import os.path as op
import glob
import sys
from scipy import io
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import os
import pyreadr

#mkae paths global
datapath = '/data/backed_up/kahwang/Clock/'
save_path='/data/backed_up/kahwang/Clock/'

# read massive data
rdata = pyreadr.read_r(datapath + 'tf_entropy_random_slopes_encoding_models.rds')
df = rdata[None]
# cut it down to only include sensor data
sdf = df.loc[df.group=='sensor']
sdf.Freq = sdf.Freq.astype('float')

def create_param_tfr(sdf, term):
    # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
    template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
    # the data array in this template is 306 ch by 20 freq by 464 time

    # create custom tfr data array
    tdf=sdf.loc[sdf.term==term]
    time = tdf.Time.unique() #what is the diff between "Time" and "t" in the dataframe?
    freq = tdf.Freq.unique()
    new_data = np.zeros((306, len(freq), len(time)))

    # now plut in real stats into the dataframe
    for index, row in tdf.iterrows():
        t = row.Time
        f = row.Freq
        ch = 'MEG'+row.level
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
    new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

    return new_tfr

entropy_change_tfr = create_param_tfr(sdf, 'entropy_change_t')
entropy_wi_tfr = create_param_tfr(sdf, 'v_entropy_wi')

# a few different types of plots
entropy_change_tfr.plot_joint(mode='mean', timefreqs=[(-1, 10), (1, 20)], picks='grad')
entropy_change_tfr.plot_topomap(ch_type='grad', tmin=-1, tmax=-0.8, fmin=15, fmax=40, title='Beta at time -1 to -.8')





# end