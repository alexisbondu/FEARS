def ts(ts_matrix):
    """
    Identity function for convenience
    """
    return ts_matrix

def zero_padding(ts_matrix, padding_size=1):
    """
    Append a column of zeros at the right-end side
    of input matrix
    """
    import numpy as np
    ts_matrix_padded = np.append(ts_matrix, np.zeros([len(ts_matrix), padding_size]), 1)
    return ts_matrix_padded

def n_derive(ts_matrix, n=1):
    """
    Derive matrix n times
    """
    assert n>=1, "Invalid n (n should be >=1)"
    import numpy as np
    if n==1:
        return np.diff(ts_matrix)
    else:
        return np.diff(n_derive(ts_matrix, n=n-1))

def n_derive_pad(ts_matrix, n=1):
    derive_matrix = n_derive(ts_matrix, n=n)
    derive_padded_matrix = zero_padding(derive_matrix, padding_size=n)
    return derive_padded_matrix

def derive(ts_matrix):
    """
    Derive matrix as per np.diff()
    and add a column of zeros on right-end side to maintain dimension
    """
    return n_derive_pad(ts_matrix, n=1)

def double_derive(ts_matrix):
    """
    Double derive matrix as per np.diff(np.diff())
    and add two columns of zeros on right-end side to maintain dimension
    """
    return n_derive_pad(ts_matrix, n=2)

def cumsum(ts_matrix):
    """
    Cumulative sum matrix as per np.cumsum()
    """
    import numpy as np
    return np.cumsum(ts_matrix, axis=1)

def double_cumsum(ts_matrix):
    """
    Double cumulative sum matrix as per np.cumsum(np.cumsum())
    """
    return cumsum(cumsum(ts_matrix))


def acf(ts_matrix):
    """
    Auto-correlate function applied from one time position relative to the next
    function is applied row by row (TS by TS) on entire array of TS
    """
    import numpy as np
    def acf_1d(ts_seq):
        return np.correlate(ts_seq, ts_seq, 'full')[len(ts_seq)-1:]
    return np.apply_along_axis(func1d=acf_1d, axis=1, arr=ts_matrix)


def ps(ts_matrix):
    """
    Power spectrum transformation function (time domain to freq domain)
    applied row by row on entire array of TS
    """
    import numpy as np
    def ps_1d(ts_seq):
        from scipy import signal
        freqs, output_ps = signal.periodogram(ts_seq, scaling='spectrum')
        return output_ps[1:]
    return np.apply_along_axis(func1d=ps_1d, axis=1, arr=ts_matrix)

####### NEW REP #######

def acf_d(ts_matrix):
    return acf(derive(ts_matrix))

def acf_dd(ts_matrix):
    return acf(double_derive(ts_matrix))

def acf_cumsum(ts_matrix):
    return acf(cumsum(ts_matrix))

def acf_dcumsum(ts_matrix):
    return acf(double_cumsum(ts_matrix))

def ps_d(ts_matrix):
    return ps(derive(ts_matrix))

def ps_dd(ts_matrix):
    return ps(double_derive(ts_matrix))

def ps_cumsum(ts_matrix):
    return ps(cumsum(ts_matrix))

def ps_dcumsum(ts_matrix):
    return ps(double_cumsum(ts_matrix))


# ----

##### NEW REP #####

transform_dict = {
    'TS' : ts,                  # Identity
    'D' : derive,               # Derivative
    'DD' : double_derive,       # Double-derivative
    'CUMSUM' : cumsum,          # Cumulative sum
    'DCUMSUM' : double_cumsum,  # Double-cumulative sum
    'ACF' : acf,                # Auto-correlation function
    'PS' : ps,                   # Power spectrum
    'ACF_D' : acf_d,
    'ACF_DD' : acf_dd,
    'ACF_CUMSUM' : acf_cumsum,
    'ACF_DCUMSUM' : acf_dcumsum,
    'PS_D' : ps_d,
    'PS_DD' : ps_dd,
    'PS_CUMSUM' : ps_cumsum,
    'PS_DCUMSUM' : ps_dcumsum,
}
