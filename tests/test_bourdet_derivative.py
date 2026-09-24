from bisect import bisect_right

import numpy as np
import pandas as pd
import pytest

from pta_learn.bourdet_derivative import der
from pta_learn.superposition_calculation import super_time_inj, super_time_shutin

DERIVATIVE_COLUMNS = ["Superposition", "Delta_Pressure", "Delta_Time", "Derivative"]


def der_loop_reference(df_super, L):
    """Original loop-based Bourdet derivative, kept as the reference for the vectorized `der`."""
    exp_L = np.exp(L)
    exp_L_half = np.exp(L / 2)
    derivatives = []

    delta_times = df_super['Delta_Time'].tolist()
    superposition = df_super['Superposition'].tolist()
    delta_pressure = df_super['Delta_Pressure'].tolist()

    for i in range(len(delta_times) - 2):
        t0, p0, dt0 = superposition[i], delta_pressure[i], delta_times[i]
        idx1 = bisect_right(delta_times, dt0 * exp_L, i + 1)
        if idx1 < len(delta_times):
            t1, p1, dt1 = superposition[idx1], delta_pressure[idx1], delta_times[idx1]
            idx2 = bisect_right(delta_times, dt1 * exp_L, idx1 + 1)
            if idx2 >= len(delta_times):
                idx2 = bisect_right(delta_times, dt1 * exp_L_half, idx1 + 1)
            if idx2 < len(delta_times):
                t2, p2 = superposition[idx2], delta_pressure[idx2]
                x1, x2 = t1 - t0, t2 - t1
                pp1, pp2 = abs(p1 - p0), abs(p2 - p1)
                d = (pp1 / x1 * x2 + pp2 / x2 * x1) / (x1 + x2)
                derivatives.append([t1, p1, dt1, d / np.log(10)])

    return pd.DataFrame(derivatives, columns=DERIVATIVE_COLUMNS)


def make_transient(n, seed=0, shutin=False):
    """Synthetic transient with irregular sampling that starts after a multi-rate history."""
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(5000, 6000, n))
    time[0] = 5000
    pressure = 3000 + 200 * np.log1p(time - 5000) + rng.normal(0, 0.5, n)
    if shutin:
        pressure = 2 * 3000 - pressure
    return pd.DataFrame({"Time": time, "Pressure": pressure})


BREAKPOINTS = pd.DataFrame({"Time": [0, 2000, 2200, 2600, 2800, 4999.0]})
RATES_INJ = np.array([5000, 0, 3000, 0, 9000, 7000.0])
RATES_SHUTIN = np.array([5000, 0, 3000, 0, 9000, 0.0])


@pytest.mark.parametrize("n", [3, 10, 500, 5000])
@pytest.mark.parametrize("L", [0.05, 0.1, 0.2, 0.5, 1.0])
@pytest.mark.parametrize("shutin", [False, True])
def test_der_matches_loop_reference(n, L, shutin):
    if shutin:
        df_super = super_time_shutin(BREAKPOINTS, RATES_SHUTIN, make_transient(n + 1, seed=n, shutin=True))
    else:
        df_super = super_time_inj(BREAKPOINTS, RATES_INJ, make_transient(n + 1, seed=n))

    expected = der_loop_reference(df_super, L)
    result = der(df_super, L)

    assert list(result.columns) == DERIVATIVE_COLUMNS
    assert len(result) == len(expected)
    if len(expected):
        pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=1e-12)


def test_der_uses_half_window_fallback_near_end():
    # uniform sampling with a wide window forces the exp(L / 2) fallback at the tail
    df_super = super_time_inj(BREAKPOINTS, RATES_INJ, make_transient(200, seed=1))
    L = 1.5
    expected = der_loop_reference(df_super, L)
    result = der(df_super, L)
    assert len(result) > 0
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=1e-12)


@pytest.mark.parametrize("n", [0, 1, 2])
def test_der_too_few_points_returns_empty_frame(n):
    df_super = pd.DataFrame({
        "Superposition": np.arange(n, dtype=float),
        "Delta_Pressure": np.arange(n, dtype=float),
        "Delta_Time": np.arange(1, n + 1, dtype=float),
    })
    result = der(df_super, 0.1)
    assert result.empty
    assert list(result.columns) == DERIVATIVE_COLUMNS


def test_der_repeated_superposition_raises_like_loop():
    df_super = pd.DataFrame({
        "Superposition": [1.0, 1.0, 2.0, 3.0],
        "Delta_Pressure": [1.0, 2.0, 3.0, 4.0],
        "Delta_Time": [1.0, 2.0, 3.0, 4.0],
    })
    with pytest.raises(ZeroDivisionError):
        der_loop_reference(df_super, 0.1)
    with pytest.raises(ZeroDivisionError):
        der(df_super, 0.1)
