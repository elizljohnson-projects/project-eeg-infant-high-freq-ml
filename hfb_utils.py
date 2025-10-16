import glob
import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
import mne

def load_eeg_files(data_dir = '.', pattern = '**/*.mat'):
    """
    Load all MATLAB EEG files from directory and convert to MNE Epochs objects.
    
    Parameters:
    -----------
    data_dir : str
        Directory containing .mat files (default: current directory)
    pattern : str
        Glob pattern for finding files (default: '**/*.mat')
    
    Returns:
    --------
    dict : Dictionary where keys are subject IDs with condition (e.g., 'Subj1_awake', 'Subj1_sleep')
           and values are mne.Epochs objects (epoch x channel x time).
    """
    # find all .mat files
    mat_files = glob.glob(os.path.join(data_dir, pattern), recursive = True)
    print(f'Found {len(mat_files)} files\n')
    
    all_data = {}
    
    # load all files
    for filepath in mat_files:
        fid = os.path.basename(filepath)
        
        # extract subject ID from fid: SubjX_eeg.mat -> SubjX
        sid = fid.replace('_eeg.mat', '')
        
        print(f'Loading {sid}...')
        
        # load file
        mat_contents = loadmat(filepath, squeeze_me = True, struct_as_record = False)
        
        # load awake_eeg and sleep_eeg structures
        for key in mat_contents.keys():
            if not key.startswith('__'):
                struct = mat_contents[key]
                
                # determine condition from structure name
                condition = 'awake' if 'awake' in key.lower() else 'sleep'
                
                # create unique key: SubjX_awake or SubjX_sleep
                data_key = f'{sid}_{condition}'
                
                # convert to mne.Epochs (n_epochs × n_channels × n_timepoints)
                epochs_data = np.array([t for t in struct.trial])
                
                # get time vector from 1st trial
                tmin = float(struct.time[0][0])
                
                # get channel list
                ch_names = [str(ch) for ch in struct.label]
                
                # create mne.Info object
                info = mne.create_info(
                    ch_names = ch_names,
                    sfreq = float(struct.fsample),
                    ch_types = 'eeg'
                )
                
                # create mne.EpochsArray
                all_data[data_key] = mne.EpochsArray(
                    data = epochs_data,
                    info = info,
                    tmin = tmin,
                    verbose = False
                )
    
    return all_data

def compute_hfb(all_data):
    """
    Compute high-frequency band (HFB: 70-150 Hz) power.
    
    Parameters:
    -----------
    all_data : dict
        Dictionary of MNE Epochs objects with keys like 'Subj1_awake', 'Subj1_sleep'
    
    Returns:
    --------
    results : dict
        Dictionary with keys like 'Subj1' containing wake and sleep HFB power,
        channel labels, channel indices, and frequency information
    """
    # define all 19 channels
    all_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 
                    'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    # set frequencies (70-80 Hz, 80-90 Hz, ... 140-150 Hz)
    fmin, fmax = 75, 145
    target_freqs = np.arange(fmin, fmax + 1, 10)
    bandwidth = 10
    
    # get subject IDs
    sids = list(set([key.split('_')[0] for key in all_data.keys()]))
    
    results = {}
    
    # loop through subjects
    for sid in sids:
        print(f'Processing {sid}...')
        
        # get wake and sleep data
        awake_epochs = all_data[f'{sid}_awake']
        sleep_epochs = all_data[f'{sid}_sleep']
        
        # get channel labels and create index mapping
        ch_awake = awake_epochs.ch_names
        ch_sleep = sleep_epochs.ch_names
        
        # create channel indices
        ch_indices_awake = [all_channels.index(ch) for ch in ch_awake if ch in all_channels]
        ch_indices_sleep = [all_channels.index(ch) for ch in ch_sleep if ch in all_channels]
        
        # compute PSD
        awake_psd, awake_freqs = mne.time_frequency.psd_array_multitaper(
            awake_epochs.get_data(),
            sfreq = awake_epochs.info['sfreq'],
            fmin = fmin,
            fmax = fmax,
            bandwidth = bandwidth,
            adaptive = False,  # equal taper weights
            low_bias = True,  # min spectral leakage
            normalization = 'full',  # PSD
            verbose = False
        )
        
        sleep_psd, sleep_freqs = mne.time_frequency.psd_array_multitaper(
            sleep_epochs.get_data(),
            sfreq = sleep_epochs.info['sfreq'],
            fmin = fmin,
            fmax = fmax,
            bandwidth = bandwidth,
            adaptive = False,  # equal taper weights
            low_bias = True,  # min spectral leakage
            normalization = 'full',  # PSD
            verbose = False
        )
        
        # select target frequencies
        freq_indices = [np.argmin(np.abs(sleep_freqs - f)) for f in target_freqs]
        selected_freqs = sleep_freqs[freq_indices]
        
        awake_psd_selected = awake_psd[:, :, freq_indices]
        sleep_psd_selected = sleep_psd[:, :, freq_indices]
        
        # normalize each frequency band by multiplying by frequency
        awake_norm2 = np.zeros_like(awake_psd_selected)
        sleep_norm2 = np.zeros_like(sleep_psd_selected)
        
        for f_idx, freq in enumerate(selected_freqs):
            awake_norm2[:, :, f_idx] = awake_psd_selected[:, :, f_idx] * freq
            sleep_norm2[:, :, f_idx] = sleep_psd_selected[:, :, f_idx] * freq
        
        # average across frequency bands
        awake_power = np.mean(awake_norm2, axis = 2)
        sleep_power = np.mean(sleep_norm2, axis = 2)
        
        # average frequencies
        mean_freq = np.mean(selected_freqs)
        
        # store results
        results[sid] = {
            'awake_power': awake_power,
            'sleep_power': sleep_power,
            'ch_awake': ch_awake,
            'ch_sleep': ch_sleep,
            'ch_indices_awake': ch_indices_awake,
            'ch_indices_sleep': ch_indices_sleep,
            'all_channels': all_channels,
            'mean_freq': mean_freq,
            'freqs': selected_freqs
        }
    
    return results

def create_hfb_df(hfb_results):
    """
    Create a dataframe with HFB power data for all epochs.
    
    Parameters:
    -----------
    hfb_results : dict
        Dictionary from compute_hfb() containing HFB results for all subjects
    
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with columns: sid, condition, fz_hfb, p3_hfb
        Each row represents one epoch
    """
    data_list = []
    
    # loop through subjects
    for sid, result in hfb_results.items():
        
        # process awake epochs
        n_awake_epochs = result['awake_power'].shape[0]
        
        for epoch_idx in range(n_awake_epochs):
            row = {'sid': sid, 'condition': 'awake'}
            
            # get Fz HFB if available
            if 'Fz' in result['ch_awake']:
                fz_idx = result['ch_awake'].index('Fz')
                row['fz_hfb'] = result['awake_power'][epoch_idx, fz_idx]
            else:
                row['fz_hfb'] = np.nan
            
            # get P3 HFB if available
            if 'P3' in result['ch_awake']:
                p3_idx = result['ch_awake'].index('P3')
                row['p3_hfb'] = result['awake_power'][epoch_idx, p3_idx]
            else:
                row['p3_hfb'] = np.nan
            
            data_list.append(row)
            
        # process sleep epochs
        n_sleep_epochs = result['sleep_power'].shape[0]
        
        for epoch_idx in range(n_sleep_epochs):
            row = {'sid': sid, 'condition': 'sleep'}
            
            # get Fz HFB if available
            if 'Fz' in result['ch_sleep']:
                fz_idx = result['ch_sleep'].index('Fz')
                row['fz_hfb'] = result['sleep_power'][epoch_idx, fz_idx]
            else:
                row['fz_hfb'] = np.nan
            
            # get P3 HFB if available
            if 'P3' in result['ch_sleep']:
                p3_idx = result['ch_sleep'].index('P3')
                row['p3_hfb'] = result['sleep_power'][epoch_idx, p3_idx]
            else:
                row['p3_hfb'] = np.nan
            
            data_list.append(row)
    
    # create df
    df = pd.DataFrame(data_list)
    
    return df
