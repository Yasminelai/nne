import numpy as np
import pandas as pd
import os

point_cloud = np.loadtxt('Project1.xyz')

N = point_cloud.shape[0]
random_indices = np.random.choice(N, 1000, replace=False)
sampled_points = point_cloud[random_indices]

point_labels = np.zeros(1000)

metadata = pd.read_csv('Project1.csv')
object_ids = metadata['ID'].values
labels = metadata[' Label'].values

label_mapping = {
    'Pipe': 1,
    'HVAC_Duct': 2,
    'Structural_IBeam': 3,
    'Structural_ColumnBeam': 4
}

mask_folder = 'Project1.masks'
mask_files = os.listdir(mask_folder)

for mask_file in mask_files:
    mask_path = os.path.join(mask_folder, mask_file)
    object_id = os.path.splitext(mask_file)[0].replace('mask', '')

    label_idx = np.where(object_ids == object_id)[0]
    if len(label_idx) > 0:
        label = labels[label_idx[0]]
        numeric_label = label_mapping[label]
        
        mask_points = np.loadtxt(mask_path, dtype=int)
        mask_points_set = set(mask_points)
        for idx_pos, idx in enumerate(random_indices):
            if idx in mask_points_set:
                point_labels[idx_pos] = numeric_label

sampled_data_with_labels = np.concatenate([sampled_points, point_labels[:, np.newaxis]], axis=1)

np.save('sampled_pointnet_input_with_labels.npy', sampled_data_with_labels)