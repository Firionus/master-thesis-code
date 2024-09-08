import numpy as np


def view_up_to_matrix(view, up):
    """converts a view and up vector in a  right-handed coordinate system to a
    transform matrix where the columns are the normalized view, left and up
    vectors"""
    view /= np.linalg.norm(view)
    up /= np.linalg.norm(up)
    assert np.dot(view, up) == 0.0
    left = -np.cross(view, up)
    return np.array([view, left, up]).transpose()
