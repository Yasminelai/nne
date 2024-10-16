import numpy as np
import pandas as pd

data = pd.read_csv('/Users/shiroixy/copenhagen/NNE/InnoTech3D/data/TrainingSet/Project1/labeled_point_cloud.csv')
data['Label'].value_counts()