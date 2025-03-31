import numpy as np
import pandas as pd
from tqdm import tqdm

def augment_points(df, points, aug_q=10):
    """
    Augment points with the given dataframe
    Args:
        df (pd.DataFrame): Dataframe with the augmentation data
        points (np.ndarray): 3d keypoints array of shape (frames, points, xyz)
    Returns:
        pd.DataFrame: Augmented annotations
        np.ndarray: Augmented points
    """

    aug_df = {
        'id': [],
        'elbow_error': [],
        'knee_error': [],
        'split': [],
    }

    #aug_points = {}
    aug_points = []

    #print("antes",len(df['id']))

    for i in tqdm(range(len(df))):
        row = df.iloc[i]

        for j in range(aug_q):
            aug_df['id'].append(row['id'])
            aug_df['elbow_error'].append(row['elbow_error'])
            aug_df['knee_error'].append(row['knee_error'])
            aug_df['split'].append(row['split'])
            #aug_points[row['id']] = points[row['id']][j::aug_q]
            aug_points.append(points[row['id']][j::aug_q])

    #print("despues",len(aug_df['id']))
    aug_df = pd.DataFrame(aug_df)
    aug_points = np.array(aug_points)
    return aug_df, aug_points