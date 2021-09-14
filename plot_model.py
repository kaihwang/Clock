#### script to plot model outputs from Alex and Michael
import numpy as np
import pandas as pd
import mne as mne
import pyreadr
import matplotlib.pyplot as plt
plt.ion()

def create_param_tfr(sdf, fdf, term, threshold = True):
    ''' function to create mne objet for ploting from R data frame
    two inputs, sdf: the dtaframe
    term, the variable for plotting
    You also have the option to threshold the data, if True, then only data with fdr_p<0.05  will be saved
    '''

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
        ch = row.level
        ch = 'MEG'+ '{:0>4}'.format(ch)
        #print(ch)
        ch_idx = mne.pick_channels(template_TFR.ch_names, [ch])
        if threshold & (fdf.loc[(fdf.Time==t) & (fdf.Freq ==f)]['p_fdr'].values<0.05):
            new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate
        elif threshold:
            new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = 0
        else:
            new_data[ch_idx, np.where(freq==f)[0], np.where(time==t)[0]] = row.estimate

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

####################
#### Read data!
####################

#mkae paths global
datapath = '/data/backed_up/kahwang/Clock/'
save_path='/data/backed_up/kahwang/Clock/'

# read massive data
pe_data = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_abs_pe.rds')
pe_rt_df, pe_rt_fdf, = extract_sensor_random_effect(pe_data, 'rt')


entropy_rdata = pyreadr.read_r(datapath + 'entropy/meg_ddf_wholebrain_entropy.rds') #whole brain data
entropy_rt_df, entropy_rt_fdf, = extract_sensor_random_effect(entropy_rdata, 'rt')
entropy_clock_df, entropy_clock_fdf, = extract_sensor_random_effect(entropy_rdata, 'clock')

entropy_change_rdata = pyreadr.read_r(datapath + 'entropy_change/meg_ddf_wholebrain_entropy_change.rds') #whole brain data
entropy_change_rt_df, entropy_change_rt_fdf = extract_sensor_random_effect(entropy_change_rdata, 'rt')
entropy_change_clock_df, entropy_change_clock_fdf = extract_sensor_random_effect(entropy_change_rdata, 'clock')

kld_rdata = pyreadr.read_r(datapath + 'kld/meg_ddf_wholebrain_kld.rds') #whole brain data
kld_rt_df, kld_rt_fdf = extract_sensor_random_effect(kld_rdata, 'rt')
kld_clock_df, kld_clock_fdf = extract_sensor_random_effect(kld_rdata, 'clock')

reward_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_reward.rds') #whole brain data
reward_rt_df, reward_rt_fdf = extract_sensor_random_effect(reward_rdata, 'rt')
reward_clock_df, reward_clock_fdf = extract_sensor_random_effect(reward_rdata, 'clock')

## entropy change pos and neg
entropy_change_neg_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_entropy_change_neg.rds') #whole brain data
entropy_change_neg_rt_df, entropy_change_neg_rt_fdf = extract_sensor_random_effect(entropy_change_neg_rdata, 'rt')
entropy_change_neg_clock_df, entropy_change_neg_clock_fdf = extract_sensor_random_effect(entropy_change_neg_rdata, 'clock')

entropy_change_pos_rdata = pyreadr.read_r(datapath + 'meg_ddf_wholebrain_entropy_change_pos.rds') #whole brain data
entropy_change_pos_rt_df, entropy_change_pos_rt_fdf = extract_sensor_random_effect(entropy_change_pos_rdata, 'rt')
entropy_change_pos_clock_df, entropy_change_pos_clock_fdf = extract_sensor_random_effect(entropy_change_pos_rdata, 'clock')

# turn dataframe into mne object for plotting
v_entropy_wi_rt_tfr = create_param_tfr(entropy_rt_df, entropy_rt_fdf, 'v_entropy_wi')
entropy_change_t_rt_tfr = create_param_tfr(entropy_change_rt_df, entropy_change_rt_fdf, 'entropy_change_t')
kld_v_entropy_wi_rt_tfr = create_param_tfr(kld_rt_df, kld_rt_fdf, 'v_entropy_wi')
v_entropy_wi_clock_tfr = create_param_tfr(entropy_clock_df, entropy_clock_fdf, 'v_entropy_wi')
entropy_change_t_clock_tfr = create_param_tfr(entropy_change_clock_df, entropy_change_clock_fdf, 'entropy_change_t')
kld_v_entropy_wi_clock_tfr = create_param_tfr(kld_clock_df, kld_clock_fdf, 'v_entropy_wi')
entropy_change_neg_t_rt_tfr = create_param_tfr(entropy_change_neg_rt_df, entropy_change_neg_rt_fdf, 'entropy_change_neg_t')
entropy_change_pos_t_rt_tfr = create_param_tfr(entropy_change_pos_rt_df, entropy_change_pos_rt_fdf, 'entropy_change_pos_t')
entropy_change_neg_t_clock_tfr = create_param_tfr(entropy_change_neg_clock_df, entropy_change_neg_clock_fdf, 'entropy_change_neg_t')
entropy_change_pos_t_clock_tfr = create_param_tfr(entropy_change_pos_clock_df, entropy_change_pos_clock_fdf, 'entropy_change_pos_t')
reward_t_rt_tfr = create_param_tfr(reward_rt_df, reward_rt_fdf, 'reward_t')
reward_t_clock_tfr = create_param_tfr(reward_clock_df, reward_clock_fdf, 'reward_t')
pe_t_rt_tfr = create_param_tfr(pe_rt_df, pe_rt_fdf, 'scale(abs_pe)')


####################
#### Plot!!
####################
# this function plots the sensor-wide TFR plot (the one you can click around with)
v_entropy_wi_clock_tfr.plot_topo(yscale='log', picks='grad')

entropy_change_t_rt_tfr.plot_topo(yscale='log', picks='grad')
entropy_change_t_clock_tfr.plot_topo(yscale='log', picks='grad')


kld_v_entropy_wi_rt_tfr.plot_topo(yscale='log', picks='grad')
kld_v_entropy_wi_clock_tfr.plot_topo(yscale='log', picks='grad')


entropy_change_neg_t_rt_tfr.plot_topo(yscale='log', picks='grad')
entropy_change_neg_t_clock_tfr.plot_topo(yscale='log', picks='grad')

reward_t_rt_tfr.plot_topo(yscale='log', picks='grad')
reward_t_clock_tfr.plot_topo(yscale='log', picks='grad')

pe_t_rt_tfr.plot_topo(yscale='log', picks='grad')

# this plots the topographic map with specific time-frequency interval
# under the 'timefreqs' flag, you can specifiy a list of (time, frequency) montage that you would like to plot
#v_entropy_wi_tfr.plot_joint(baseline=None, yscale='log', timefreqs=[(-1.5, 10), (-1, 10), (-0.5, 10), (0, 10), (0.5, 10), (1, 10)], picks='grad')

########
# another way to generate time-frequency montage
# Here let us plot the montage of alpha
# you would have to play around the colorbar scale (vmin and vmax).

fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.5, 0.9, 0.05)
for n, time in enumerate(times):
    pe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=5, fmax=8, ch_type ='grad', title = ('5 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.5, 1.3, 0.05)
for n, time in enumerate(times):
    pe_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=12, ch_type ='grad', title = ('8 to 12 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()




fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0, 1.5, 0.1)
for n, time in enumerate(times):
    reward_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=9, ch_type ='grad', title = ('3 to 7 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()



##### entropy_change_neg_t_rt_tfr, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-0.5, 1.5, 0.1)
for n, time in enumerate(times):
    entropy_change_pos_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad',vmin = -0.4, vmax = 0.4, title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-0.5, 1.5, 0.1)
for n, time in enumerate(times):
    entropy_change_neg_t_rt_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=4, fmax=7, ch_type ='grad', title = ('3 to 7 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()



##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.7, -1.45, 0.05)
for n, time in enumerate(times):
    v_entropy_wi_clock_tfr.plot_topomap(baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### entropy_change_t_rt_tfr, alpha effect around 0 to 1sec
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.8, 1.2, 0.05)
for n, time in enumerate(times):
    entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### entropy_change_t_rt_tfr, low beta effect around 0 to 1sec
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.1, 0.5, 0.05)
for n, time in enumerate(times):
    entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=12, fmax=18, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('12 to 18 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### entropy_change_t_rt_tfr, alpha effect around 0 to 1sec
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.8, 1.2, 0.05)
for n, time in enumerate(times):
    entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=1, fmax=18, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### entropy_change_t_rt_tfr, theta effect around 1sec
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(0.8, 1.2, 0.05)
for n, time in enumerate(times):
    entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=3, fmax=5, vmin = -0.055, vmax = 0.055, ch_type ='grad', title = ('3 to 5 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()


##### entropy_change_t_rt_tfr, detla effect around 1sec
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(1, 1.5, 0.05)
for n, time in enumerate(times):
    entropy_change_t_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=2.1, fmax=4, vmin = -0.08, vmax = 0.08, ch_type ='grad', title = ('3 to 5 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()





##### entropy_change_t_clock_tfr, alpha effect
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.7, -1.3, 0.025)
for n, time in enumerate(times):
    entropy_change_t_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()


##### kld, alpha effect
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.7, -1.3, 0.025)
for n, time in enumerate(times):
    entropy_change_t_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=3, fmax=8, vmin = -0.15, vmax = 0.15, ch_type ='grad', title = ('3 to 8 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()



##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.5, 1.8, 0.3)
for n, time in enumerate(times):
    kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', vmin = -0.06, vmax = 0.06, title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(1.4, 2, 0.1)
for n, time in enumerate(times):
    kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=8, fmax=14, ch_type ='grad', title = ('8 to 14 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(1.4, 2, 0.1)
for n, time in enumerate(times):
    kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(1.4, 2, 0.1)
for n, time in enumerate(times):
    kld_v_entropy_wi_rt_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
plt.show()

##### v_entropy_wi, effect around -1.5 s (clock alignment)
fig, axis = plt.subplots(3, 5, squeeze = False, figsize=(25,10))
times = np.arange(-1.7, 0, 0.1)
for n, time in enumerate(times):
    kld_v_entropy_wi_clock_tfr.plot_topomap(cmap = 'bwr', baseline=None, tmin = time, tmax = time+0.05, fmin=13, fmax=20, ch_type ='grad', title = ('13 to 20 hz at time %s' %np.round(time, 2)), show=False, axes = axis[n//5, n%5])
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
