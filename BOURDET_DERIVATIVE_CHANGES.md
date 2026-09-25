# Vectorized Bourdet derivative (`der`)

Branch `claude/youthful-johnson-v3j184`, commit `b2545f9` on top of `main` (`34b87e1`).

## What changed

- **`pta_learn/bourdet_derivative.py`**: in `der(df_super, L)`, the per-point Python loop, which used `bisect_right`, is replaced with NumPy array operations.
  - For every point `i`, the next neighbour is `idx1 = max(i+1, searchsorted(Delta_Time, Delta_Time[i]·e^L, 'right'))`.
  - The one after is `idx2 = max(idx1+1, searchsorted(Delta_Time, Delta_Time[idx1]·e^L, 'right'))`.
  - Where `idx2` runs past the end of the data, it falls back to `e^(L/2)`, as the old `else` branch did.
  - Rows where no valid `idx1` or `idx2` exists are dropped, as before.
  - The Bourdet formula is unchanged, `(|Δp1|/x1·x2 + |Δp2|/x2·x1)/(x1+x2)/ln(10)`, and now runs on whole arrays.
  - The `bisect` import is removed.
- The signature and the output columns (`Superposition, Delta_Pressure, Delta_Time, Derivative`) are unchanged. Each output row is still taken at `idx1`.
- If two neighbouring points have the same superposition time (a zero `x1`, `x2` or `x1+x2`), it raises `ZeroDivisionError`, as the old loop did.
- Nothing else changed: `superposition_calculation.py`, `cal_loglog_shut`, `cal_loglog_inj` and all other callers are untouched.

## New tests: `tests/test_bourdet_derivative.py` (45 tests)

- The original loop is kept in the test file as `der_loop_reference`. The new `der` is checked against it (identical rows, values to rtol 1e-12) on synthetic injection transients (`super_time_inj`) and shut-in transients (`super_time_shutin`), for n = 3, 10, 500 and 5000, and L = 0.05, 0.1, 0.2, 0.5 and 1.0.
- Edge cases: the `e^(L/2)` fallback (checked to actually run), inputs of 0–2 points (empty table with the right columns), and repeated superposition times (`ZeroDivisionError`).
- The full suite passes, 104 tests. It was run on Python 3.11; `pyproject.toml` requires ≥3.12.

## Speed

About 5× faster at 1,000 points and 20–30× faster at 100,000.

| Points | Old loop | Vectorized |
|---|---|---|
| 1,000 | ~2.5 ms | ~0.5 ms |
| 10,000 | ~20 ms | ~1.2 ms |
| 100,000 | ~180–275 ms | ~8–10 ms |

## Data flow to check

`ti_workflow` transients → `cal_loglog_shut` / `cal_loglog_inj` (`bourdet_derivative.py`) → `super_time_shutin` / `super_time_inj` (`superposition_calculation.py`) → **`der`** → `normal_calc` (`normalization.py:32`, reads the `Delta_Pressure` and `Derivative` columns) → `plot_TI_family` (`ti_misc.py:85`).

`feature_extraction.PTAClassifier` and `pattern_recognition` take derivative data as input but don't call `der`.

## Points worth checking

1. **Sorted input:** the new code assumes `Delta_Time` is in ascending order, as the old `bisect` code did. This holds when `Time` is sorted. Check that every upstream path sorts it and that no transient has duplicate timestamps.
2. **Output dtypes and index:** columns are now always float64, including when the result is empty. Previously an empty result had `object` columns. The index is a fresh RangeIndex, same as before. Check that nothing downstream relied on the old dtypes.
3. **Input conversion:** input columns are converted with `.to_numpy(dtype=float)`. Non-numeric or nullable (`pd.NA`) values would now fail at that conversion, which is earlier than before.
4. **NaN or inf:** NaN or inf in `Superposition` or `Delta_Pressure` now produces NaN or inf in the result without an error, as the old loop did. NaN in `Delta_Time` breaks `searchsorted` ordering in both versions.
5. **Error timing:** the zero-division check now runs over all rows before anything is computed. The old loop failed partway through. The result is the same, since both raise and return nothing.
6. **The `L` values:** confirm the defaults, `L=0.1` for shut-in and `0.2` for injection, and any values passed through the README workflow, behave as expected on real field data, not just the synthetic test data.
