def expected_periodogram(omega, params, n1, n2, delta1, delta2):
    """
    Computes the expected periodogram for a single frequency using direct summation.
    This method is very slow for many frequencies.
    
    Args:
        omega (tuple): The frequency vector (omega1, omega2).
        params (list): Model parameters.
        n1 (int): The number of samples in the first dimension.
        n2 (int): The number of samples in the second dimension.
        delta1 (float): The sampling interval in the first dimension.
        delta2 (float): The sampling interval in the second dimension.
        
    Returns:
        complex: The expected periodogram value at the given frequency.
    """
    d = 2
    sum_val = 0j
    
    for u1 in range(n1):
        for u2 in range(n2):
            tmp = cn_bar(u1,u2, params) + \
                  cn_bar(u1-n1,u2-n2,params) + \
                  cn_bar(u1,u2-n2, params) + \
                  cn_bar(u1-n1,u2, params)
            
            tmp2 = -(omega[0] * u1 * delta1 + omega[1] * u2 * delta2) * 1j
            sum_val += tmp * cmath.exp(tmp2)

    return sum_val * (delta1 * delta2) / (2 * cmath.pi)**d

def expected_periodogram_summation_all(params, n1, n2, delta1, delta2):
    """
    Computes the expected periodogram for ALL frequencies using the slow, direct summation.
    This is for a fair comparison with the FFT method.
    """
    all_periodograms = np.zeros((n1, n2), dtype=complex)
    for u1 in range(n1):
        for u2 in range(n2):
            omega = (2 * cmath.pi * u1 / n1 / delta1, 2 * cmath.pi * u2 / n2 / delta2)
            all_periodograms[u1, u2] = expected_periodogram(omega, params, n1, n2, delta1, delta2)
    return all_periodograms



import numpy as np
import cmath
import pandas as pd
import time

def cgn(u):
    """
    Computes a 2D Bartlett window function (triangular window).
    
    Args:
        u (tuple): A tuple of lag indices (u1, u2).
        
    Returns:
        float: The window value.
    """
    u1, u2 = u
    return (1 - np.abs(u1) / 64) * (1 - np.abs(u2) / 128) 

def cov_x(u1, u2, params):
    """
    Computes the autocovariance of the original process.
    
    Args:
        u1 (int): The first lag index.
        u2 (int): The second lag index.
        params (list): A list of parameters for the covariance function.
                       Example: [sigma2, alpha1, alpha2].
        
    Returns:
        float: The autocovariance value.
    """
    sigma2, alpha1, alpha2 = params
    return sigma2 * np.exp(-np.sqrt((u1 / alpha1)**2 + (u2 / alpha2)**2))

def cov_laplacian(u1, u2, params):
    """
    Computes the autocovariance of the Laplacian-filtered process.
    
    Args:
        u1 (int): The first lag index.
        u2 (int): The second lag index.
        params (list): A list of parameters for the covariance function.
        
    Returns:
        float: The autocovariance value of the filtered process.
    """
    delta1, delta2 = 0.044, 0.063
    
    # Define the 5-point stencil of the discrete Laplacian
    stencil_weights = {(0, 0): -4, (0, 1): 1, (0, -1): 1, (1, 0): 1, (-1, 0): 1}
    
    cov = 0
    # Iterate through all pairs of points in the stencil
    for (a, b), w_ab in stencil_weights.items():
        for (c, d), w_cd in stencil_weights.items():
            # Calculate the effective lag vector
            lag_x = (u1 + a - c) * delta1
            lag_y = (u2 + b - d) * delta2
            
            # Add the weighted covariance term
            cov += w_ab * w_cd * cov_x(lag_x, lag_y, params)
            
    return cov

def cn_bar(u1, u2, params):
    """
    Computes the periodicized autocovariance by multiplying the
    Laplacian covariance with a 2D Bartlett window.
    
    Args:
        u1 (int): The first lag index.
        u2 (int): The second lag index.
        params (list): Model parameters.
        
    Returns:
        float: The periodicized and windowed autocovariance value.
    """
    u = (u1, u2)
    return cov_laplacian(u1, u2, params) * cgn(u)


def expected_periodogram_fft(params, n1, n2, delta1, delta2):
    """
    Computes the expected periodogram for ALL frequencies using a 2D FFT.
    This method is much faster for a full grid of frequencies.
    
    Args:
        params (list): Model parameters.
        n1 (int): The number of samples in the first dimension.
        n2 (int): The number of samples in the second dimension.
        delta1 (float): The sampling interval in the first dimension.
        delta2 (float): The sampling interval in the second dimension.
        
    Returns:
        np.ndarray: A 2D array of expected periodogram values for all frequencies.
    """
    cn_tilde_matrix = np.zeros((n1, n2), dtype=complex)
    
    for u1 in range(n1):
        for u2 in range(n2):
            cn_tilde_matrix[u1, u2] = cn_bar(u1, u2, params) + \
                                      cn_bar(u1 - n1, u2 - n2, params) + \
                                      cn_bar(u1, u2 - n2, params) + \
                                      cn_bar(u1 - n1, u2, params)
    
    fft_result = np.fft.fft2(cn_tilde_matrix)
    
    normalization_factor = (delta1 * delta2) / (2 * cmath.pi)**2
    expected_periodogram = fft_result * normalization_factor
    
    return expected_periodogram

def compute_2d_periodogram_from_df(df, value_column='laplacian', lat_column='Latitude', lon_column='Longitude'):
    """
    Computes the 2D periodogram from a pandas DataFrame containing spatial data.

    Args:
        df (pd.DataFrame): The input DataFrame.
        value_column (str): The name of the column containing the data values (e.g., 'laplacian').
        lat_column (str): The name of the column for the row index (e.g., 'Latitude').
        lon_column (str): The name of the column for the column headers (e.g., 'Longitude').

    Returns:
        np.ndarray: A 2D NumPy array of the periodogram values.
    """
    # 1. Pivot the DataFrame to reshape the 1D series into a 2D grid.
    # The `lat_column` is used as the index and `lon_column` as the columns
    # because Latitude typically represents the y-axis (rows) and Longitude the x-axis (columns).
    # Since Latitude changes more slowly, it makes sense to use it as the index.
    data_grid = df.pivot_table(index=lat_column, columns=lon_column, values=value_column)
    
    # 2. Convert the 2D pandas DataFrame to a 2D NumPy array.
    data_array = data_grid.values
    
    # 3. Compute the 2D FFT.
    # The number of rows and columns in the array.
    n1, n2 = data_array.shape
    
    # The `np.fft.fft2` function is used for a 2D FFT.
    fft_result = np.fft.fft2(data_array)
    
    # 4. Calculate the periodogram.
    # The periodogram is the squared magnitude of the FFT result, normalized by the number of samples.
    periodogram = (np.abs(fft_result)**2) / (n1 * n2)
    
    return periodogram

# Example of how to use the function with a sample DataFrame
# Assuming 'df' is your DataFrame from the image.
periodogram_values = compute_2d_periodogram_from_df(df1)


def likelihood(params, df):
    periodogram_values = compute_2d_periodogram_from_df(df)
    n1, n2 = periodogram_values.shape
    delta1, delta2 = 0.044, 0.063
    n = n1*n2
    # Ensure the expected periodogram's frequency order matches the data's periodogram
    expected_periodogram_values = expected_periodogram_fft(params, n1, n2, delta1, delta2)
    
    # Flatten both periodograms for easier computation
    periodogram_flat = periodogram_values.flatten()
    expected_flat = expected_periodogram_values.flatten()
    
    # Use the real part and ensure it's non-negative for the log-likelihood
    expected_flat_real = np.maximum(expected_flat.real, 1e-10)
    
    # Compute the negative log-likelihood using the real-valued expected periodogram
    nll = np.sum(np.log(expected_flat_real) + periodogram_flat / expected_flat_real)
    return nll/n

params = [20, 0.5, 0.5]  # Example parameters: [sigma2, alpha1, alpha2]
a = likelihood(params, df1)
print(a)


# laplacian positions
lats = cur['Latitude'].unique()
lons = cur['Longitude'].unique()


positions = [ (lats[0], lons[1]), (lats[1],lons[0]), (lats[1], lons[2]), (lats[2],lons[1])  ]
print(positions)
tmp = 0 
for x,y in positions:
    tmp+= cur.loc[(cur['Latitude'] == x) & (cur['Longitude'] == y), 'ColumnAmountO3'].values[0]

tmp -= cur.loc[(cur['Latitude'] == lats[1]) & (cur['Longitude'] == lons[1]), 'ColumnAmountO3'].values[0]*4
tmp

print(lats[1], lons[1])
