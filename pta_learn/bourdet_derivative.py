from .superposition_calculation import *


# function to calculate bourdet derivative
def der(df_super, L):

    """
    Computes Bourdet Derivatives

    This function calculates the derivative of pressure changes (`Delta_Pressure`) with respect to time (`Delta_Time`) 
    using Bourdet method. The derivative is computed for each point in the input DataFrame 
    based on neighboring points. Neighbor indices for all points are found at once with a vectorized
    binary search (`np.searchsorted`), so no Python-level loop over the data is needed.

    Parameters:
    - df_super (pd.DataFrame): Input DataFrame containing the following columns:
        - `Delta_Time`: Time differences between consecutive measurements.
        - `Superposition`: Superposition time values from superposition_calculation.py.
        - `Delta_Pressure`: Pressure differences corresponding to each time step.
    - L (float): A scaling factor used to compute exponential time thresholds for selecting neighboring points.

    Returns:
    - df_derivative (pd.DataFrame): A DataFrame containing the computed derivatives with the following columns:
        - `Superposition`: Superposition time values.
        - `Delta_Pressure`: Pressure differences.
        - `Delta_Time`: Time differences.
        - `Derivative`: Computed derivative values (logarithmically scaled).

    Mathematical Formulation:
    The derivative is calculated using a weighted average of pressure changes over time:
        derivative = ( (|p1 - p0| / (t1 - t0) * (t2 - t1) + (|p2 - p1| / (t2 - t1) * (t1 - t0) ) / ( (t1 - t0) + (t2 - t1) )
    where:
        - t0, t1, t2: Superposition times for the current, next, and subsequent points.
        - p0, p1, p2: Pressure differences at times t0, t1, and t2.

    The result is then scaled by `1 / log(10)` to convert it to a logarithmic derivative.
    """

    # set the smoothing factor, this is important when the data goes to the end, to ensure the real response, the L should be smaller.
    exp_L = np.exp(L)
    exp_L_half = np.exp(L / 2)

    # Delta_Time is sorted ascending, so neighbor lookups can use a vectorized binary search
    delta_times = df_super['Delta_Time'].to_numpy(dtype=float)
    superposition = df_super['Superposition'].to_numpy(dtype=float)
    delta_pressure = df_super['Delta_Pressure'].to_numpy(dtype=float)
    n = len(delta_times)

    # current point index (t0, p0) for every candidate point
    idx0 = np.arange(max(n - 2, 0))

    # next index: first point after idx0 with Delta_Time > dt0 * exp(L)
    idx1 = np.maximum(idx0 + 1, np.searchsorted(delta_times, delta_times[idx0] * exp_L, side='right'))
    valid = idx1 < n
    idx0, idx1 = idx0[valid], idx1[valid]

    # subsequent index: first point after idx1 with Delta_Time > dt1 * exp(L)
    idx2 = np.maximum(idx1 + 1, np.searchsorted(delta_times, delta_times[idx1] * exp_L, side='right'))

    # near the end of the data, fall back to the half smoothing window exp(L / 2)
    fallback = idx2 >= n
    idx2[fallback] = np.maximum(idx1[fallback] + 1,
                                np.searchsorted(delta_times, delta_times[idx1[fallback]] * exp_L_half, side='right'))
    valid = idx2 < n
    idx0, idx1, idx2 = idx0[valid], idx1[valid], idx2[valid]

    # Calculate derivatives
    x1 = superposition[idx1] - superposition[idx0]
    x2 = superposition[idx2] - superposition[idx1]
    if np.any(x1 == 0) or np.any(x2 == 0) or np.any(x1 + x2 == 0):
        raise ZeroDivisionError("float division by zero: repeated superposition time values")
    pp1 = np.abs(delta_pressure[idx1] - delta_pressure[idx0])
    pp2 = np.abs(delta_pressure[idx2] - delta_pressure[idx1])
    derivative = (pp1 / x1 * x2 + pp2 / x2 * x1) / (x1 + x2) / np.log(10)

    df_derivative = pd.DataFrame({'Superposition': superposition[idx1],
                                  'Delta_Pressure': delta_pressure[idx1],
                                  'Delta_Time': delta_times[idx1],
                                  'Derivative': derivative})

    return df_derivative





def select_period(df_bhp, df_rate, start_t, end_t):
    """
    Selects a period from the BHP (Bottom Hole Pressure) and rate DataFrames based on the specified start and end times.

    Parameters:
    - df_bhp (DataFrame): The input BHP DataFrame.
    - df_rate (DataFrame): The input rate DataFrame.
    - start_t (float): The start time of the period in hours.
    - end_t (float): The end time of the period in hours.

    Returns:
    - bhp_shut (DataFrame): The selected BHP DataFrame for the specified period.
    - rate_shut (DataFrame): The selected rate DataFrame for the specified period.
    """
    bhp_shut = df_bhp[(df_bhp['Time'] >= start_t) & (df_bhp['Time'] <= end_t)]
    rate_shut = df_rate[(df_rate['Time'] >= start_t) & (df_rate['Time'] <= end_t)]
    return bhp_shut, rate_shut


# new version of calculate loglog, this will be updated when the aw_viewer is ready

def cal_loglog_shut(df_bhp, df_rate, selected_list,all_breakpoints,rebuilt_rate,idx, L=0.1):

    # Make sure the provided index is within the range of selected_list
    if idx >= len(selected_list):
        raise ValueError(
            f"Provided index {idx} is out of range for selected_list of length {len(selected_list)}")

    df_bhp_1 = df_bhp.copy()
    df_rate_1 = df_rate.copy()

    # select the value in start/hr column in the row of idx
    ss = selected_list['start/hr'].iloc[idx]
    # select the value in end/hr column
    se = selected_list['end/hr'].iloc[idx]
    # select the value in start/timestamp column in the row of idx
    ss_time = selected_list['start/timestamp'].iloc[idx]
    # select the value in end/timestamp column
    se_time = selected_list['end/timestamp'].iloc[idx]
    # duration of the transient
    duration = selected_list['duration/hr'].iloc[idx]
    # select the data during shut-in
    df_target_s1 = df_bhp[(df_bhp_1['Time'] >= ss) & (df_bhp_1['Time'] <= se)].reset_index(drop=True)

    # the pressure for superposition plot
    super_p = df_target_s1['Pressure'].values[1:]

    # select the data before shut-in, this is necessary for loglog plot
    df_shut_bhp_1_before, df_shut_rate_1_before = select_period(df_bhp_1, df_rate_1, 0, se)

    # get all the breakpoints before the start of the shutin
    a1_breakpoints = all_breakpoints[all_breakpoints['Time'] <= ss].reset_index(drop=True)

    # get the rows from 0 to len(ia1_breakpoints) from rebuilt_rate
    rate_rebuilt_1 = rebuilt_rate.iloc[:len(a1_breakpoints)].reset_index(drop=True)

    # Extract the 'weighted average rate' column as a NumPy array
    weighted_rate_1 = rate_rebuilt_1['Weighted Averaged Rate'].to_numpy()

    # # superposition time in the shutin period
    df_super_detect_s1 = super_time_shutin(a1_breakpoints, weighted_rate_1, df_target_s1)

    # # derivative in the shutin period
    df_derivative_detect_s1 = der(df_super_detect_s1,L)

    return df_derivative_detect_s1, weighted_rate_1, a1_breakpoints

def cal_loglog_inj(df_bhp, df_rate, selected_list,all_breakpoints,rebuilt_rate,idx, L=0.2):

    # Make sure the provided index is within the range of selected_list
    if idx >= len(selected_list):
        raise ValueError(
            f"Provided index {idx} is out of range for selected_list of length {len(selected_list)}")

    df_bhp_1 = df_bhp.copy()
    df_rate_1 = df_rate.copy()

    # select the value in start/hr column in the row of idx
    ins = selected_list['start/hr'].iloc[idx]
    # select the value in end/hr column
    ine = selected_list['end/hr'].iloc[idx]
    # duration of the transient
    duration = selected_list['duration/hr'].iloc[idx]
    # select the value in start/timestamp column in the row of idx
    ins_time = selected_list['start/timestamp'].iloc[idx]
    # select the value in end/timestamp column
    ine_time = selected_list['end/timestamp'].iloc[idx]

    # select the data during shut-in
    df_target_i1 = df_bhp_1[(df_bhp_1['Time'] >= ins) & (df_bhp_1['Time'] <= ine)].reset_index(drop=True)
    df_target_rate_i1 = df_rate_1[(df_rate_1['Time'] >= ins) & (df_rate_1['Time'] <= ine)].reset_index(drop=True)
    
    # the pressure for superposition plot
    super_p = df_target_i1['Pressure'].values[1:]

    # select the data before shut-in, this is necessary for loglog plot
    df_inj_bhp_1_before, df_inj_rate_1_before = select_period(df_bhp_1, df_rate_1, 0, ine)

    # get all the breakpoints before the start of the shutin
    ia1_breakpoints = all_breakpoints[all_breakpoints['Time'] <= ins].reset_index(drop=True)

    # get the rows from 0 to len(ia1_breakpoints) from rebuilt_rate
    rate_rebuilt_1 = rebuilt_rate.iloc[:len(ia1_breakpoints)].reset_index(drop=True)

    # Extract the 'weighted average rate' column as a NumPy array
    weighted_rate_i1 = rate_rebuilt_1['Weighted Averaged Rate'].to_numpy()

    # superposition time in the inj period
    df_super_detect_i1 = super_time_inj(ia1_breakpoints,weighted_rate_i1,df_target_i1)

    # calculate the derivative in the inj period
    df_derivative_detect_i1 = der(df_super_detect_i1,L)

    return df_derivative_detect_i1, weighted_rate_i1,ia1_breakpoints
