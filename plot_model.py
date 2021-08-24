#### script to plot model outputs from Alex and Michael
import numpy as np
import pandas as pd
import mne as mne
import pyreadr
import matplotlib.pyplot as plt
plt.ion()

def create_param_tfr(sdf, term):
    ''' function to create mne objet for ploting from R data frame
    two inputs, sdf: the dtaframe
    term, the variable for plotting
    '''

    # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
    template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
    # the data array in this template is 306 ch by 20 freq by 464 time

    # create custom tfr data array
    tdf=sdf.loc[sdf.term==term]
    time = np.sort(tdf.Time.unique()) #what is the diff between "Time" and "t" in the dataframe?
    freq = np.sort(tdf.Freq.unique())
    new_data = np.zeros((306, len(freq), len(time)))

    # now plut in real stats into the dataframe
    for index, row in tdf.iterrows():
        t = row.Time
        f = row.Freq
        ch = row.level
        ch = 'MEG'+ '{:0>4}'.format(ch)
        #print(ch)
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
    new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)

    return new_tfr

def extract_sensor_random_effect(rdata, alignment):
    ''' take r data frame, extract sensor level random effect, remeber to specity the alignment 'rt' or 'clock' '''
    df = rdata[None]
    # cut it down to only include sensor data
    df = df.loc[df.group=='Sensor']
    #select random effects
    df = df.loc[df.effect=='ran_coefs']
    #which alignment?
    df = df.loc[df.alignment==alignment]
    df.Freq = df.Freq.astype('float')

    return df

####################
#### Read data!
####################

#mkae paths global
datapath = '/data/backed_up/kahwang/Clock/'
save_path='/data/backed_up/kahwang/Clock/'

# read massive data
entropy_rdata = pyreadr.read_r(datapath + 'entropy/meg_ddf_wholebrain_entropy.rds') #whole brain data
entropy_df = extract_sensor_random_effect(entropy_rdata, 'rt')

entropy_change_rdata = pyreadr.read_r(datapath + 'entropy_change/meg_ddf_wholebrain_entropy_change.rds') #whole brain data
entropy_change_df = extract_sensor_random_effect(entropy_change_rdata, 'rt')

kld_rdata = pyreadr.read_r(datapath + 'kld/meg_ddf_wholebrain_kld.rds') #whole brain data
kld_df = extract_sensor_random_effect(kld_rdata, 'rt')

# turn dataframe into mne object for plotting
v_entropy_wi_tfr = create_param_tfr(entropy_df, 'v_entropy_wi')
entropy_change_t_tfr = create_param_tfr(entropy_change_df, 'entropy_change_t')
kld_v_entropy_wi_tfr = create_param_tfr(kld_df, 'v_entropy_wi') ### I'm not sure what term to plot from kld.


####################
#### Plot!!
####################
# this function plots the sensor-wide TFR plot (the one you can click around with)
v_entropy_wi_tfr.plot_topo(yscale='log', picks='grad')
entropy_change_t_tfr.plot_topo(yscale='log', picks='grad')
kld_v_entropy_wi_tfr.plot_topo(yscale='log', picks='grad')
# this plots the topographic map with specific time-frequency interval
# under the 'timefreqs' flag, you can specifiy a list of (time, frequency) montage that you would like to plot
#v_entropy_wi_tfr.plot_joint(baseline=None, yscale='log', timefreqs=[(-1.5, 10), (-1, 10), (-0.5, 10), (0, 10), (0.5, 10), (1, 10)], picks='grad')

########
# another way to generate time-frequency montage
# Here let us plot the montage of alpha
# you would have to play around the colorbar scale (vmin and vmax).
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.5, 1.5, 0.2)
for n, time in enumerate(times):
    v_entropy_wi_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, vmin = -0.06, vmax = 0.06, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

# Here let us plot the montage of theta
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.5, 1.5, 0.2)
for n, time in enumerate(times):
    v_entropy_wi_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=7, vmin = -0.03, vmax = 0.03, ch_type ='grad', title = ('4 to 7 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

# Here let us plot the montage of delta
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.5, 1.5, 0.2)
for n, time in enumerate(times):
    v_entropy_wi_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=2.1, fmax=3, ch_type ='grad', title = ('2 to 3 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()



########################################################################
######### graveyard

#entropy_change_tfr = create_param_tfr(sdf, 'entropy_change_t')
#entropy_change_tfr.plot_topo(yscale='log', picks='grad')

#entropy_change_tfr = create_param_tfr(sdf, 'entropy_change_t')
# a few different types of plots
#entropy_change_tfr.plot_topo(yscale='log', picks='grad')
#entropy_change_tfr.plot_joint(mode='mean', yscale='log', timefreqs=[(-0.1, 20), (0.25, 12), (0.6, 5), (0.9, 10), (1, 5)], picks='grad')


#entropy_wi_tfr = create_param_tfr(sdf, 'v_entropy_wi')
#entropy_change_tfr.plot_joint(mode='mean', timefreqs=[(-1, 10), (1, 20)], picks='grad')
#entropy_change_tfr.plot_topomap(ch_type='grad', tmin=-1, tmax=-0.8, fmin=15, fmax=40, title='Beta at time -1 to -.8')
#
#
# reward_tfr = create_param_tfr(sdf, 'reward_t')
# reward_tfr.plot_joint(mode='mean', timefreqs=[(-1, 10),(0.6, 24), (1, 20)], picks='grad')
# reward_tfr.plot_topomap(ch_type='grad', tmin=0.4, tmax=0.8, fmin=15, fmax=30, title='Beta at time 0.4 to 0.8')
#
# entropy_wi_tfr = create_param_tfr(sdf, 'v_entropy_wi')
#
# fdf = df.loc[pd.isna(df.group)]
# fdf = fdf.loc[fdf.alignment=='rt']
# fdf.Freq = fdf.Freq.astype('float')
#
# def create_param_fixed_tfr(sdf, term):
#     # creat TFR epoch object for plotting. Use the "info" in this file for measurement info
#     template_TFR = mne.time_frequency.read_tfrs(datapath + 'Group/group_feedback_power-tfr.h5')[0]
#     # the data array in this template is 306 ch by 20 freq by 464 time
#
#     # create custom tfr data array
#     tdf=sdf.loc[sdf.term==term]
#     time = tdf.Time.unique() #what is the diff between "Time" and "t" in the dataframe?
#     freq = tdf.Freq.unique()
#     new_data = np.zeros((306, len(freq), len(time)))
#
#     # now plut in real stats into the dataframe
#     for index, row in tdf.iterrows():
#         t = row.Time
#         f = row.Freq
#         #ch = 'MEG'+row.level
#         #ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
#         new_data[:, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
#     new_tfr = mne.time_frequency.AverageTFR(template_TFR.info, new_data, time, freq, 1)
#
#     return new_tfr
#
#
#
# entropy_wi_fixed_tfr = create_param_fixed_tfr(fdf, 'v_entropy_wi')
#
#

# end
