import numpy as np
import pandas as pd


def hyper_plane_vec(point_list):
    ones = np.ones((point_list.shape[0], 1))
    point_list = np.concatenate((point_list, ones), axis=1)
    zeros = np.zeros((point_list.shape[0]))
    zeros[0] = 1
    solution = np.linalg.inv(point_list) @ zeros
    return solution


xmin = 0
xmax = 1

X, Y = np.mgrid[xmin:xmax:32j, xmin:xmax:32j]
positions = np.vstack([X.ravel(), Y.ravel()])

positions = positions.T
dim = positions.shape[0]
ones = np.linspace(0, dim - 1, dim).astype(int)

res = pd.DataFrame(positions)


def patches(positions):
    result = []
    for ind in range(0, dim - 1):
        if not (1 in positions.iloc[ind:ind + 1, :].values):
            upper = [ind, ind + 1, ind + 33]
            lower = [ind, ind + 32, ind + 33]
            result.append(upper)
            result.append(lower)
    return pd.DataFrame(result)


def neighbour_ind(patches, points):
    result_list = []
    for ind in points.index:
        obj = []
        for pat in patches.index:
            if (ind in patches.iloc[pat].values):
                obj.append(pat)
        result_list.append(obj)
    return pd.DataFrame(result_list)


def neighbour_p(patches, points):
    list = []
    for ind in points.index:
        obj = []
        for pat in patches.index:
            if ind in patches.iloc[pat].values:
                loc_obj = [a for a in patches.iloc[pat].values
                           if a != ind
                           ]
                obj.append(loc_obj)
        list.append(obj)
    return pd.DataFrame(list)


def neighbour_point(patches, points):
    list = []
    for ind in points.index:
        obj = []
        for pat in patches.index:
            if ind in patches.iloc[pat].values:
                loc_obj = [res.iloc[a].values.tolist() for a in patches.iloc[pat].values]
                inverted = hyper_plane_vec(np.array(loc_obj))
                obj.append(inverted)
        list.append(obj)
    return pd.DataFrame(list)


def planes_to_values(point_df):
    point_neighbour = point_df.fillna(value=np.nan)
    alt_point = point_neighbour.T.fillna(method='ffill').T
    alt_point.columns = ['1', '2', '3', '4', '5', '6']
    alt_point['7'] = alt_point['1']
    alt_point['8'] = alt_point['2']
    stacked = alt_point.stack().values
    vec = np.array([0.5, 0.5, 1])
    result = [a @ vec for a in stacked]
    return result


