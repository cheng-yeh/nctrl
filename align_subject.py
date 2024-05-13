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

parser = argparse.ArgumentParser(description='Process EEG data for specified subjects and runs.')
parser.add_argument('--num_subject', type=int, help='Number of subjects')
parser.add_argument('--num_run', type=int, help='Number of runs')
args = parser.parse_args()

subjects = range(1,1+args.num_subject)
runs = range(1,1+args.num_run)

pairs = []
loaded_data = {}

for sub in subjects:
    loaded_data[sub] = {}
    for run in runs:
        # Load data from the .npz file
        path = f'data/broderick2019_eeg/Natural Speech/test_data/subjectall_runall/lfreq10_hfreqNone_len10/Subject{sub}/Run{run}/subject{sub}_bestrun5_stride2_lfreq10_hfreqNone_len10/x_z.npz'
        if not os.path.exists(path):
            continue
        pairs.append([sub, run])
        loaded_data[sub][run] = np.load(path, allow_pickle=True)

        # Access the data by the keys specified during saving
        #for key in loaded_data:
        #    print(key, loaded_data[key])

print("Pairs of subject and run: ", pairs)
print("Keys of loaded data", list(loaded_data[subjects[0]][runs[0]].keys()))
sys.stdout.flush()

num_channels = loaded_data[subjects[0]][runs[0]]['X'].shape[-1]

# Reference subject for alignment
ref_sub = 1

all_sub_corr = {}

for sub in subjects:
    print(f"Computing subject-wise correlation for subject {sub}...")
    sys.stdout.flush()
    tmp_x = []
    tmp_z = []
    for run in loaded_data[sub]:
        # X and Z time series data with shapes (samples, seg_length, channels)
        x_time_series = loaded_data[sub][run]['X']
        z_time_series = loaded_data[sub][run]['Z_est']
    
        # Reshape the data to (samples, channels) for easier iteration
        x_reshaped = np.reshape(x_time_series, (-1, x_time_series.shape[-1]))
        z_reshaped = np.reshape(z_time_series, (-1, z_time_series.shape[-1]))
 
        tmp_x.append(x_reshaped)
        tmp_z.append(z_reshaped)

    tmp_x = np.concatenate(tmp_x)
    tmp_z = np.concatenate(tmp_z)
    print("Shape of all z: ", tmp_z.shape)

    # Compute correlation coefficients
    correlation_coefficients = np.zeros((num_channels, num_channels))
    for i in range(num_channels):
        print(i, end=' ')
        for j in range(num_channels):
            correlation_coefficients[i, j] = np.corrcoef(tmp_x[:, i], tmp_z[:, j])[0, 1]
    all_sub_corr[sub] = correlation_coefficients

aligned_sub_data = {}
for sub in subjects:
    if sub == ref_sub:
        aligned_sub_data[ref_sub] = {}
        for run in loaded_data[ref_sub]:
            x_time_series = loaded_data[sub][run]['X']
            z_time_series = loaded_data[sub][run]['Z_est']
            x_time_series = np.reshape(x_time_series, (-1, x_time_series.shape[-1]))
            z_time_series = np.reshape(z_time_series, (-1, z_time_series.shape[-1]))
            aligned_sub_data[ref_sub][run] = {'X': x_time_series, 'aligned_Z': z_time_series}
        continue
 
    print(f"Aligning subject {sub} with respect to reference subject {ref_sub}")
    sys.stdout.flush()

    sub_corr = np.matmul(all_sub_corr[ref_sub].T, all_sub_corr[sub])
    
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

    aligned_sub_data[sub] = {}
    for run in loaded_data[sub]:
        x_time_series = loaded_data[sub][run]['X']
        z_time_series = loaded_data[sub][run]['Z_est']
        x_time_series = np.reshape(x_time_series, (-1, x_time_series.shape[-1]))
        z_time_series = np.reshape(z_time_series, (-1, z_time_series.shape[-1]))
        aligned_sub_data[sub][run] = {'X': x_time_series, 'aligned_Z': z_time_series[:, order_indices]}

# Load the standard BioSemi montage
montage = mne.channels.make_standard_montage("biosemi128")

# Get the positions of the electrodes from the montage
electrode_positions = montage.get_positions()

# Load 3D position
electrode_positions_3d = np.array(list(electrode_positions['ch_pos'].values()))
# Convert 3D position into 2D layout
electrode_positions_2d = np.array(list(electrode_positions['ch_pos'].values()))[:, :2]

pos = []

for j in range(num_channels):
    #weight = np.power(np.abs(all_corr[sub][run][:, j]), 6)
    #weight = np.clip(all_corr[sub][run][:, j], a_min=0, a_max=None)
    weight = np.power(np.clip(all_sub_corr[ref_sub][:, j], a_min=0, a_max=None), 2)
    total_weight = np.sum(weight)
    pos.append([np.sum(weight * electrode_positions_3d[:, 0]) / total_weight,
                np.sum(weight * electrode_positions_3d[:, 1]) / total_weight,
                np.sum(weight * electrode_positions_3d[:, 2]) / total_weight])
pos = np.array(pos)
pos_data = {"position": pos}
np.savez(f'../../eeg2text/brainmagick/data/broderick2019/download/Natural Speech/EEG/all_sub_pos.npz', **pos_data)

# Save the dictionary to a .npz file
for sub, run in pairs:
    print(f"Saving {sub}, {run}...")
    np.savez(f'../../eeg2text/brainmagick/data/broderick2019/download/Natural Speech/EEG/Subject{sub}/Subject{sub}_Run{run}.npz', **aligned_sub_data[sub][run])
    print("Shape of X", aligned_sub_data[sub][run]['X'].shape)
    print("Shape of aligned_Z", aligned_sub_data[sub][run]['aligned_Z'].shape)

