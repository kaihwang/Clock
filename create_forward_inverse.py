# create forward models and inverse operator for Clock MEG subjects
import mne
import pandas as pd
import numpy as np
from Clock import raw_to_epoch
import os.path as op
import nibabel as nib

subjects_dir = '/data/backed_up/kahwang/Clock/'
subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)
channels_list = np.load('/home/kahwang/bin/Clock/channel_list.npy')

# test subject
#subjects = [10637]

##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### MNE inverse solution, project to cortical surface, source space in surface space
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####

###################################################################################################
# Step 1. Coregister landmark with digitization
###################################################################################################

# The coregistration is the operation that allows to position the head and
# the sensors in a common coordinate system. In the MNE software the transformation
# to align the head and the sensors in stored in a so-called trans file. It is a FIF
# file that ends with -trans.fif. It can be obtained with mne.gui.coregistration()
# (or its convenient command line equivalent mne coreg), or mrilab if you’re using
# a Neuromag system. HAS TO BE DONE MANUALLY SUB by SUB
# See: https://www.slideshare.net/mne-python/mnepython-coregistration


for subject in subjects:

    # Only run subjects with freesurfer and trans.fif created:
    trans_file = subjects_dir + 'trans/%s-trans.fif' %subject

    if not op.isfile(trans_file):
        msg = 'trans file not there for subject %s' %subject
        print(msg)
        continue

    mri_folder = subjects_dir + '%s/mri/' %subject

    if not op.isdir(mri_folder):
        msg = 'freesurfer files not there for subject %s' %subject
        print(msg)
        continue

    ###################################################################################################
    # Step 2. Create head model, recreate because it was last done in 2014!
    ###################################################################################################
    ##########################
    # 2.1 create watershed bem
    ##########################
#     try:
#         mne.bem.make_watershed_bem(str(subject), subjects_dir=subjects_dir, overwrite=True)
#     except:
#         msg = 'no subject %s freesurfer files' %subject
#         print(msg)

    # visualize bem results
    # mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir,brain_surfaces='white', orientation='coronal')

    ##########################
    # Step 2.2 create source space. placing dipoles 5mm apart on the surface
    ##########################
    src = mne.setup_source_space(str(subject), spacing='oct6', add_dist='patch', subjects_dir=subjects_dir)
    src_fn = '/data/backed_up/kahwang/Clock/%s/MEG/%s-oct6-src.fif' %(subject, subject)
    mne.write_source_spaces(src_fn, src, overwrite=True)

    ##########################
    # 2.3 make foward solution
    ##########################
    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(str(subject), ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    # to calculate foward solution, need trans done!
    raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(subject, subject)
    raw = mne.io.read_raw(raw_fname)

    # need to copy trans file to meg folder: cp /mnt/nfs/lss/lss_kahwang_hpc/Clock/11328/MEG/11328-trans.fif .
    trans_file = subjects_dir + 'trans/%s-trans.fif' %subject
    src_fn = '/data/backed_up/kahwang/Clock/%s/MEG/%s-oct6-src.fif' %(subject, subject)
    src = mne.read_source_spaces(src_fn)

    fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=src, bem=bem, meg=True, eeg=False, mindist=5.0, n_jobs=16, verbose=True)
    fwd_fn = subjects_dir + '%s/MEG/%s-fwd.fif' %(subject, subject)
    mne.write_forward_solution(fwd_fn, fwd, overwrite=True, verbose=None)

    #check TRANS is appropriate
    # raw = mne.io.read_raw(raw_fname)
    # fig = mne.viz.plot_alignment(raw.info, trans=trans, subject=str(subject),
    #                        src=src, subjects_dir=subjects_dir, dig=True,
    #                        surfaces=['head-dense', 'white'], coord_frame='meg')

    ###################################################################################################
    # Step 3 calculate noise cov
    ###################################################################################################
    # compute noise covariance using the pre stim baseline data
    Event_types = ['ITI']
    ITI_ep = raw_to_epoch(subject, Event_types)
    ITI_ep['ITI'].apply_baseline((0,1))
    noise_cov = mne.compute_covariance(ITI_ep['ITI'], tmin = 0, tmax=1, method=['shrunk', 'empirical'],
        rank = 'info', verbose=True)

    raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(subject, subject)
    raw = mne.io.read_raw(raw_fname)
    #fig_cov, fig_spectra = mne.viz.plot_cov(noise_cov, raw.info)

    ###################################################################################################
    # Step 4. create inverse operator
    ###################################################################################################
    # read forward
    fwd_fn = subjects_dir + '%s/MEG/%s-fwd.fif' %(subject, subject)
    fwd = mne.read_forward_solution(fwd_fn)

    try:
        # make inverse using the ingredients from the above calculations
        inverse_operator = mne.minimum_norm.make_inverse_operator(raw.info, fwd, noise_cov, loose=0.2, depth=0.8)
        inv_fn = subjects_dir + '%s/MEG/%s-mne-inv.fif' %(subject, subject)
        mne.minimum_norm.write_inverse_operator(inv_fn, inverse_operator, verbose=None)

    except:
        msg = 'cant create inverse for subject %s' %s
        print(msg)
        continue


    ###################################################################################################
    # Step 5 apply inverse operator
    ###################################################################################################
    # load inverse
    # inv_fn = subjects_dir + '%s/MEG/%s-mne-inv.fif' %(subject, subject)
    # inverse_operator = mne.minimum_norm.read_inverse_operator(inv_fn)
    #
    # # apply inverse
    # # notes on single trial projection: https://mne.tools/stable/auto_examples/inverse/plot_compute_mne_inverse_epochs_in_label.html?highlight=apply%20inverse%20single%20trial
    # method = "MNE"
    # snr = 3.
    # lambda2 = 1. / snr ** 2
    #
    # # load evoke data to be projected
    # Event_types = ['feedback']
    # fb_ep = raw_to_epoch(subject, Event_types)
    #
    # stc = mne.minimum_norm.apply_inverse_epochs(fb_ep['feedback'], inverse_operator, lambda2,
    #     method=method, pick_ori=None, verbose=True)

    # stc will have source esitmates for each trial
    # need to figure out how to create source ROIs into freesurfer labels: https://surfer.nmr.mgh.harvard.edu/fswiki/mri_vol2label

    # visualize stc. Not working on thalamege.
    # mne.viz.plot_source_estimates(stc, subject = str(subject), subjects_dir = subjects_dir)

    # Notes on how to morph source estimate into a common reference
    # https://mne.tools/stable/auto_examples/inverse/plot_morph_volume_stc.html#sphx-glr-auto-examples-inverse-plot-morph-volume-stc-py




##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
##### Beamforming inverse solution, project to source space in volumne
##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### ##### #####
# see https://mne.tools/stable/auto_tutorials/source-modeling/plot_beamformer_lcmv.html#sphx-glr-auto-tutorials-source-modeling-plot-beamformer-lcmv-py

#subjects = [11329]
subjects = np.loadtxt('/home/kahwang/bin/Clock/subjects', dtype=int)

for subject in subjects:

    # Only run subjects with freesurfer and trans.fif created:
    trans_file = subjects_dir + 'trans/%s-trans.fif' %subject

    if not op.isfile(trans_file):
        msg = 'trans file not there for subject %s' %subject
        print(msg)
        continue

    mri_folder = subjects_dir + '%s/mri/' %subject

    if not op.isdir(mri_folder):
        msg = 'freesurfer files not there for subject %s' %subject
        print(msg)
        continue

    ###################################################################################################
    # Step 1. lcmv needs cov for both data and noise (prestim baseline)
    ###################################################################################################
    Event_types = ['ITI']
    ITI_ep = raw_to_epoch(subject, Event_types)
    ITI_ep['ITI'].apply_baseline((0,1))

    Event_types = ['feedback']
    fb_ep = raw_to_epoch(subject, Event_types)

    data_cov = mne.compute_covariance(fb_ep['feedback'], tmin=0, tmax=0.8, method='empirical', rank = 'info')
    noise_cov = mne.compute_covariance(ITI_ep['ITI'], tmin=0, tmax=1, method='empirical', rank = 'info')

    # plot
    #data_cov.plot(fb_ep['feedback'].info)
    #noise_cov.plot(ITI_ep['ITI'].info)

    ###################################################################################################
    # Step 2. setup volume source space
    ###################################################################################################

    surface = op.join(subjects_dir, str(subject), 'bem', 'inner_skull.surf')
    vol_src = mne.setup_volume_source_space(str(subject), subjects_dir=subjects_dir, surface=surface)
    print(vol_src)

    src_fn = subjects_dir +'%s/MEG/%s-vol-src.fif' %(subject, subject)
    mne.write_source_spaces(src_fn, vol_src, overwrite = True)

    # visualize it.
    #mne.viz.plot_bem(subject=subject, subjects_dir=subjects_dir, brain_surfaces='white', src=vol_src, orientation='coronal')

    ###################################################################################################
    # Step 3. foward model
    ###################################################################################################

    conductivity = (0.3,)  # for single layer
    # conductivity = (0.3, 0.006, 0.3)  # for three layers
    model = mne.make_bem_model(str(subject), ico=4, conductivity=conductivity, subjects_dir=subjects_dir)
    bem = mne.make_bem_solution(model)

    # to calculate foward solution, need trans done!
    raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(subject, subject)
    raw = mne.io.read_raw(raw_fname)

    # need to copy trans file to meg folder: cp /mnt/nfs/lss/lss_kahwang_hpc/Clock/11328/MEG/11328-trans.fif .
    trans_file = subjects_dir + 'trans/%s-trans.fif' %subject
    src_fn = subjects_dir +'%s/MEG/%s-vol-src.fif' %(subject, subject)
    vol_src = mne.read_source_spaces(src_fn)

    vol_fwd = mne.make_forward_solution(raw_fname, trans=trans_file, src=vol_src, bem=bem, meg=True, eeg=False, n_jobs=16, verbose=True)
    vol_fwd_fn = subjects_dir + '%s/MEG/%s-vol-fwd.fif' %(subject, subject)
    mne.write_forward_solution(vol_fwd_fn, vol_fwd, overwrite=True, verbose=None)

    ###################################################################################################
    # Step 4. spatial filter for lcmv
    ###################################################################################################
    vol_fwd_fn = subjects_dir + '%s/MEG/%s-vol-fwd.fif' %(subject, subject)
    vol_fwd = mne.read_forward_solution(vol_fwd_fn)
    raw_fname = subjects_dir + '%s/MEG/%s_clock_run1_dn_ds_sss_raw.fif' %(subject, subject)
    raw = mne.io.read_raw(raw_fname)

    try:
        filters = mne.beamformer.make_lcmv(raw.info, vol_fwd, data_cov, reg=0.05, noise_cov=noise_cov, pick_ori='max-power', weight_norm='unit-noise-gain', reduce_rank = True, rank=None)
        #need to investigate the effect of reduced rank

        filter_fn = subjects_dir + '%s/MEG/%s-vol-beamformer-lcmv.h5' %(subject, subject)
        filters.save(filter_fn, overwrite=True)

    except:
        msg = "beamformer for subject %s failed" %subject
        print(msg)
        continue


    ###################################################################################################
    # apply filter
    ###################################################################################################
    # filters = mne.beamformer.read_beamformer(filter_fn)
    # apply filter, convert to nibabel img
    # stc = mne.beamformer.apply_lcmv(fb_ep['feedback'].average(), filters, max_ori_out='signed')
    # img = stc.as_volume(vol_src)
    #
    # stcs = mne.beamformer.apply_lcmv_epochs(fb_ep['feedback'], filters, max_ori_out='signed')
    # # each trial can be converted to nibabel img
    # img = stcs[0].as_volume(vol_src)


# end of script
