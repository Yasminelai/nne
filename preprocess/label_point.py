import numpy as np
import pandas as pd

def load_point_cloud(file_path):
    return np.loadtxt(file_path)

point_cloud = load_point_cloud('/Users/shiroixy/copenhagen/NNE/InnoTech3D/data/TrainingSet/Project1/Project1.xyz')
point_cloud = np.random.permutation(point_cloud)[:1000]
data = pd.read_csv('/Users/shiroixy/copenhagen/NNE/InnoTech3D/data/TrainingSet/Project1/Project1.csv')


def label_points(point_cloud, df):
    labels = []
    for point in point_cloud:
        x, y, z = point[:3]
        point_label = 'Unlabeled'  
        
        for _, row in df.iterrows():
            if (row[' BB.Min.X '] <= x <= row[' BB.Max.X '] and
                row[' BB.Min.Y '] <= y <= row[' BB.Max.Y '] and
                row[' BB.Min.Z '] <= z <= row[' BB.Max.Z']):
                point_label = row[' Label']
                break
        
        labels.append(point_label)
    
    return labels

point_labels = label_points(point_cloud, data)


labeled_points_df = pd.DataFrame(point_cloud, columns=['X', 'Y', 'Z', 'I', 'R', 'G', 'B'])
labeled_points_df['Label'] = point_labels

labeled_points_df.head(100)


output_file = 'labeled_point_cloud.csv'
labeled_points_df.to_csv(output_file, index=False)