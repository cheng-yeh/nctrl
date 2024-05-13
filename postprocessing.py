import numpy as np
import matplotlib.pyplot as plt
import mne
import os
from scipy.stats import spearmanr
import networkx as nx
import argparse
import sys

def find_max_weight_matching(weight_matrix):
    num_nodes = len(weight_matrix)
    
    # Create a directed graph
    G = nx.Graph()
    
    # Add nodes to both sides of the bipartite graph
    G.add_nodes_from(range(num_nodes), bipartite=0)
    G.add_nodes_from(range(num_nodes, 2 * num_nodes), bipartite=1)
    
    # Add edges with weights from the weight matrix
    for i in range(num_nodes):
        for j in range(num_nodes):
            G.add_edge(i, j + num_nodes, weight=weight_matrix[i][j])
    
    # Find maximum weight matching using NetworkX's implementation
    max_weight_matching = nx.max_weight_matching(G, maxcardinality=True)
    
    return max_weight_matching

# sfreq for broderick2019
sfreq = 128.0

parser = argparse.ArgumentParser(description='Process EEG data for specified subjects and runs.')
parser.add_argument('--num_subject', type=int, help='Number of subjects')
parser.add_argument('--num_run', type=int, help='Number of runs')
args = parser.parse_args()

subjects = range(1,1+args.num_subject)
runs = range(1,1+args.num_run)

pairs = []
loaded_data = {'low': {}, 'mid': {}, 'high': {}}

for sub in subjects:
    loaded_data['low'][sub] = {}
    loaded_data['mid'][sub] = {}
    loaded_data['high'][sub] = {}
    for run in runs:
        # Load data from the .npz file
        path_low = f'data/broderick2019_eeg/Natural Speech/test_data/lfreq10.0_hfreqNone_sfreq128.0_ch42_len10/Subject{sub}/Run{run}/x_z.npz'
        path_mid = f'data/broderick2019_eeg/Natural Speech/test_data/lfreq1.0_hfreq10.0_sfreq16.0_ch42_len10/Subject{sub}/Run{run}/x_z.npz'
        path_high = f'data/broderick2019_eeg/Natural Speech/test_data/lfreq0.1_hfreq1.0_sfreq2.0_ch42_len10/Subject{sub}/Run{run}/x_z.npz'

        if not (os.path.exists(path_low) and os.path.exists(path_mid) and os.path.exists(path_high)):
            continue
        pairs.append([sub, run])
        loaded_data['low'][sub][run] = np.load(path_low, allow_pickle=True)
        loaded_data['mid'][sub][run] = np.load(path_mid, allow_pickle=True)
        loaded_data['high'][sub][run] = np.load(path_high, allow_pickle=True)

        # Access the data by the keys specified during saving
        #for key in loaded_data:
        #    print(key, loaded_data[key])

print("Pairs of subject and run: ", pairs)
print("Keys of loaded data", list(loaded_data['low'][subjects[0]][runs[0]].keys()))
sys.stdout.flush()

num_channels = loaded_data['low'][subjects[0]][runs[0]]['X'].shape[-1]

postprocessed_data = {}

# Arveraging sliding window and upsampling
for mode in ['low', 'mid', 'high']:
    for sub in subjects:
        if sub not in postprocessed_data:
            postprocessed_data[sub] = {}
        for run in loaded_data[mode][sub]:
            if run not in postprocessed_data[sub]:
                postprocessed_data[sub][run] = {}
            # Averaging all the overlapping sliding windows
            # Create an array to hold the combined segments
            target_x_array = loaded_data[mode][sub][run]['X']
            target_x_recon_array = loaded_data[mode][sub][run]['X_recon']
            target_z_array = loaded_data[mode][sub][run]['Z_est']
            combined_x_array = np.zeros((target_x_array.shape[0] + 9, target_x_array.shape[2]))
            combined_x_recon_array = np.zeros((target_x_recon_array.shape[0] + 9, target_x_recon_array.shape[2]))
            combined_z_array = np.zeros((target_z_array.shape[0] + 9, target_z_array.shape[2]))

            overlap_counts = np.zeros(combined_x_array.shape)
            # Combine the segments by averaging the overlapping parts
            for i in range(target_x_array.shape[0]):
                combined_x_array[i:i+10] += target_x_array[i]
                combined_x_recon_array[i:i+10] += target_x_recon_array[i]
                combined_z_array[i:i+10] += target_z_array[i]
                overlap_counts[i:i+10] += 1

            print("Shape of count: ", overlap_counts.shape)
            print("Count: ", overlap_counts)
            combined_x_array /= overlap_counts
            combined_x_recon_array /= overlap_counts
            combined_z_array /= overlap_counts
            
            # Print the shape of the combined array
            print("Shape of combined x array: ", combined_x_array.shape)
            print("Shape of combined x_recon array: ", combined_x_recon_array.shape)
            print("Shape of combined z array: ", combined_z_array.shape)

            # Upsampling
            if mode == 'low':
                postprocessed_data[sub][run]['X_low'] = combined_x_array
                postprocessed_data[sub][run]['X_recon_low'] = combined_x_recon_array
                postprocessed_data[sub][run]['Z_low'] = combined_z_array
                continue

            if mode == 'mid':
                up_scale = sfreq / 16.0
            elif mode == 'high':
                up_scale = sfreq / 2.0

            resampled_x_array = mne.filter.resample(combined_x_array, up=up_scale, axis=0)
            resampled_x_recon_array = mne.filter.resample(combined_x_recon_array, up=up_scale, axis=0)
            resampled_z_array = mne.filter.resample(combined_z_array, up=up_scale, axis=0)
            print("Shape of resampled x array: ", resampled_x_array.shape)
            print("Shape of resampled x_recon array: ", resampled_x_recon_array.shape)
            print("Shape of resampled z array: ", resampled_z_array.shape)

            postprocessed_data[sub][run][f'X_{mode}'] = resampled_x_array
            postprocessed_data[sub][run][f'X_recon_{mode}'] = resampled_x_recon_array
            postprocessed_data[sub][run][f'Z_{mode}'] = resampled_z_array

# Reference subject for alignment
ref_sub = 1

all_sub_corr = {'low': {}, 'mid': {}, 'high': {}}

for mode in ['low', 'mid', 'high']:
    for sub in subjects:
        print(f"Computing subject-wise correlation for subject {sub}...")
        sys.stdout.flush()
        tmp_x = []
        tmp_z = []
        for run in loaded_data[mode][sub]:
            # X and Z time series data with shapes (samples, seg_length, channels)
            x_time_series = postprocessed_data[sub][run][f'X_{mode}']
            z_time_series = postprocessed_data[sub][run][f'Z_{mode}']
        
            # Reshape the data to (samples, channels) for easier iteration
            x_reshaped = np.reshape(x_time_series, (-1, x_time_series.shape[-1]))
            z_reshaped = np.reshape(z_time_series, (-1, z_time_series.shape[-1]))
     
            tmp_x.append(x_reshaped)
            tmp_z.append(z_reshaped)
    
        tmp_x = np.concatenate(tmp_x)
        tmp_z = np.concatenate(tmp_z)
        print("Shape of all x: ", tmp_x.shape)
        print("Shape of all z: ", tmp_z.shape)
    
        # Compute correlation coefficients
        correlation_coefficients = np.zeros((num_channels, num_channels))
        for i in range(num_channels):
            print(i, end=' ', flush=True)
            for j in range(num_channels):
                correlation_coefficients[i, j] = np.corrcoef(tmp_x[:, i], tmp_z[:, j])[0, 1]
        all_sub_corr[mode][sub] = correlation_coefficients

for mode in ['low', 'mid', 'high']:
    for sub in subjects:
        if sub == ref_sub:
            #aligned_sub_data[mode][ref_sub] = {}
            for run in loaded_data[mode][ref_sub]:
                postprocessed_data[sub][run][f'Z_{mode}_align'] = postprocessed_data[sub][run][f'Z_{mode}']
            #    aligned_sub_data[mode][ref_sub][run] = {f'Z_{mode}': postprocessed_data[sub][run][f'Z_{mode}']}
            continue
        
        print(f"Aligning subject {sub} with respect to reference subject {ref_sub}")
        sys.stdout.flush()
    
        sub_corr = np.matmul(all_sub_corr[mode][ref_sub].T, all_sub_corr[mode][sub])
        
        matching = find_max_weight_matching(sub_corr)
    
        matching = list(matching)
    
        match_count = 0.
        for i in range(num_channels):
            if matching[i][0] > matching[i][1]:
                matching[i] = (matching[i][1], matching[i][0] - num_channels)
            else:
                matching[i] = (matching[i][0], matching[i][1] - num_channels)
    
            if matching[i][0] == matching[i][1]:
                match_count += 1
        print("Ratio of unchanged matching: ", match_count / num_channels)
    
        matching = sorted(matching, key=lambda x: x[0])
        
        order_indices = np.array([m[1] for m in matching])
        
        print("Order of new matching: ", order_indices)
    
        for run in loaded_data[mode][sub]:
            postprocessed_data[sub][run][f'Z_{mode}_align'] = postprocessed_data[sub][run][f'Z_{mode}'][:, order_indices]
            all_sub_corr[mode][sub] = all_sub_corr[mode][sub][:, order_indices]
            print(postprocessed_data[sub][run][f'Z_{mode}_align'].shape)

# Combine Z_low, Z_mid, and Z_high together
for sub in subjects:
    for run in postprocessed_data[sub]:
        min_length = postprocessed_data[sub][run][f'Z_low'].shape[0]
        for mode in ['mid', 'high']:
            if min_length > postprocessed_data[sub][run][f'Z_{mode}'].shape[0]:
                min_length = postprocessed_data[sub][run][f'Z_{mode}'].shape[0]

        concatenated_array = np.concatenate((postprocessed_data[sub][run][f'X_low'][:min_length], 
                                             postprocessed_data[sub][run][f'X_mid'][:min_length], 
                                             postprocessed_data[sub][run][f'X_high'][:min_length]), axis=-1)
        print("shape of raw eeg (lmh): ", concatenated_array.shape)
        postprocessed_data[sub][run][f'X_lmh'] = concatenated_array

        concatenated_array = np.concatenate((postprocessed_data[sub][run][f'X_low'][:min_length], 
                                             postprocessed_data[sub][run][f'X_mid'][:min_length]), axis=-1)
        print("shape of raw eeg (lm): ", concatenated_array.shape)
        postprocessed_data[sub][run][f'X_lm'] = concatenated_array

        concatenated_array = np.concatenate((postprocessed_data[sub][run][f'X_mid'][:min_length], 
                                             postprocessed_data[sub][run][f'X_high'][:min_length]), axis=-1)
        print("shape of raw eeg (mh): ", concatenated_array.shape)
        postprocessed_data[sub][run][f'X_mh'] = concatenated_array

        concatenated_array = np.concatenate((postprocessed_data[sub][run][f'X_low'][:min_length], 
                                             postprocessed_data[sub][run][f'X_high'][:min_length]), axis=-1)
        print("shape of raw eeg (lh): ", concatenated_array.shape)
        postprocessed_data[sub][run][f'X_lh'] = concatenated_array

        for ver in ['', '_align']:
            concatenated_array = np.concatenate((postprocessed_data[sub][run][f'Z_low{ver}'][:min_length], 
                                                 postprocessed_data[sub][run][f'Z_mid{ver}'][:min_length], 
                                                 postprocessed_data[sub][run][f'Z_high{ver}'][:min_length]), axis=-1)
            print("shape of the-eeg (lmh): ", concatenated_array.shape)
            postprocessed_data[sub][run][f'Z_lmh{ver}'] = concatenated_array
    
            concatenated_array = np.concatenate((postprocessed_data[sub][run][f'Z_low{ver}'][:min_length], 
                                                 postprocessed_data[sub][run][f'Z_mid{ver}'][:min_length]), axis=-1)
            print("shape of the-eeg (lm): ", concatenated_array.shape)
            postprocessed_data[sub][run][f'Z_lm{ver}'] = concatenated_array
    
            concatenated_array = np.concatenate((postprocessed_data[sub][run][f'Z_mid{ver}'][:min_length], 
                                                 postprocessed_data[sub][run][f'Z_high{ver}'][:min_length]), axis=-1)
            print("shape of the-eeg (mh): ", concatenated_array.shape)
            postprocessed_data[sub][run][f'Z_mh{ver}'] = concatenated_array
    
            concatenated_array = np.concatenate((postprocessed_data[sub][run][f'Z_low{ver}'][:min_length], 
                                                 postprocessed_data[sub][run][f'Z_high{ver}'][:min_length]), axis=-1)
            print("shape of the-eeg (lh): ", concatenated_array.shape)
            postprocessed_data[sub][run][f'Z_lh{ver}'] = concatenated_array

# Load the standard BioSemi montage
montage = mne.channels.make_standard_montage("biosemi128")

# Get the positions of the electrodes from the montage
electrode_positions = montage.get_positions()

# Load 3D position
electrode_positions_3d = np.array(list(electrode_positions['ch_pos'].values()))
# Convert 3D position into 2D layout
electrode_positions_2d = np.array(list(electrode_positions['ch_pos'].values()))[:, :2]

# Drop channel arrocding to num_selected_channel
name_list = list(montage.ch_names)
step = len(name_list) // num_channels
selected_idx = [idx * step for idx in range(num_channels)]

pos_data = {}

pos_by_hierarchy = {"low": [], "mid": [], "high": []}

for mode in ["low", "mid", "high"]:
    for j in range(num_channels):
        #weight = np.power(np.abs(all_corr[sub][run][:, j]), 6)
        #weight = np.clip(all_corr[sub][run][:, j], a_min=0, a_max=None)
        weight = np.power(np.clip(all_sub_corr[mode][ref_sub][:, j], a_min=0, a_max=None), 2)
        total_weight = np.sum(weight)
        
        pos_by_hierarchy[mode].append([np.sum(weight * electrode_positions_3d[selected_idx, 0]) / total_weight,
                                       np.sum(weight * electrode_positions_3d[selected_idx, 1]) / total_weight,
                                       np.sum(weight * electrode_positions_3d[selected_idx, 2]) / total_weight])
    pos_by_hierarchy[mode] = np.array(pos_by_hierarchy[mode])
    pos_data[f"pos_Z_{mode[0]}"] = pos_by_hierarchy[mode]
    pos_data[f"pos_X_{mode[0]}"] = electrode_positions_3d[selected_idx]
    
pos_data[f"pos_Z_lm"] = np.concatenate([pos_by_hierarchy['low'], pos_by_hierarchy['mid']])
print("shape of the-eeg (lm)'s pos: ", pos_data[f"pos_Z_lm"].shape)
pos_data[f"pos_Z_lh"] = np.concatenate([pos_by_hierarchy['low'], pos_by_hierarchy['high']])
print("shape of the-eeg (lh)'s pos: ", pos_data[f"pos_Z_lh"].shape)
pos_data[f"pos_Z_mh"] = np.concatenate([pos_by_hierarchy['mid'], pos_by_hierarchy['high']])
print("shape of the-eeg (mh)'s pos: ", pos_data[f"pos_Z_mh"].shape)
pos_data[f"pos_Z_lmh"] = np.concatenate([pos_by_hierarchy['low'], pos_by_hierarchy['mid'], pos_by_hierarchy['high']])
print("shape of the-eeg (lmh)'s pos: ", pos_data[f"pos_Z_lmh"].shape)

pos_data[f"pos_X_lm"] = np.concatenate([electrode_positions_3d[selected_idx], electrode_positions_3d[selected_idx]])
pos_data[f"pos_X_lh"] = np.concatenate([electrode_positions_3d[selected_idx], electrode_positions_3d[selected_idx]])
pos_data[f"pos_X_mh"] = np.concatenate([electrode_positions_3d[selected_idx], electrode_positions_3d[selected_idx]])
pos_data[f"pos_X_lmh"] = np.concatenate([electrode_positions_3d[selected_idx], electrode_positions_3d[selected_idx], electrode_positions_3d[selected_idx]])

np.savez(f'../../eeg2text/brainmagick/data/broderick2019/download/Natural Speech/EEG/all_sub_pos.npz', **pos_data)

# Save the dictionary to a .npz file
for sub, run in pairs:
    print(f"Saving {sub}, {run}...")
    np.savez(f'../../eeg2text/brainmagick/data/broderick2019/download/Natural Speech/EEG/Subject{sub}/Subject{sub}_Run{run}.npz', **postprocessed_data[sub][run])
    print(postprocessed_data[sub][run].keys())
    print([postprocessed_data[sub][run][k].shape for k in postprocessed_data[sub][run].keys()])
