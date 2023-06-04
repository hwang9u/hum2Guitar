def denormalize(x, minmax_values):
    min_, max_ = minmax_values
    return ((x + 1) / 2) * (max_ - min_) + min_

def normalize(x, return_minmax_values=True):
    x_norm = minmax(x) # [0, 1]
    x_norm = 2*x_norm -1 # [-1, 1]
    
    if return_minmax_values:
        minmax_values = x.min(), x.max()
        return x_norm, minmax_values
    else:
        return x_norm


def minmax(x):
    return (x-x.min()) / (x.max() - x.min())

