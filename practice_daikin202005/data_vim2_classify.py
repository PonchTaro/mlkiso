
import numpy as np
import tables

import data
import model_perceptron
import helpers as hlp

# Parameters for experimental setup.
_part = 0 # specify label for one-versus-all classification.

def gen():
    '''
    Data generation function for our
    customized feature data for the
    vim-2 participants.
    '''
    
    datafile = "vim-2_classify_mjh.h5"
    with tables.open_file(datafile, mode="r") as f:
        X_tr = f.get_node(where=f.root.tr, name="X").read()
        y_tr = f.get_node(where=f.root.tr, name="y").read()
        X_te = f.get_node(where=f.root.te, name="X").read()
        y_te = f.get_node(where=f.root.te, name="y").read()

    # Binarize the labels.
    idx_eq = y_tr == _part
    idx_neq = y_tr != _part
    y_tr[idx_eq] = 1
    y_tr[idx_neq] = 0

    idx_eq = y_te == _part
    idx_neq = y_te != _part
    y_te[idx_eq] = 1
    y_te[idx_neq] = 0

    return data.DataSet(X_tr=X_tr, X_te=X_te,
                        y_tr=y_tr, y_te=y_te,
                        name="vim-2_classify")


