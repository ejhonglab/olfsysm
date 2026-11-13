"""Tests `get_time_index` against olfsysm's own time steps, with and without the pretime,
its `legacy_linspace` times, and `slice_to_odor_pulse` on both.
"""
import numpy as np
import pytest
import xarray as xr

import olfsysm as osm

from pyolfsysm.mb_model import get_time_index, slice_to_odor_pulse


def _orn_run(time_stim_end=None):
    """Returns (mp, rv) of an olfsysm run of the ORN / LN layers only, for one odor and two
    glomeruli (every layer's dynamics have the same time steps)."""
    mp = osm.ModelParams()
    if time_stim_end is not None:
        mp.time_stim_end = time_stim_end
    mp.orn.data.delta = np.array([[50.0], [20.0]])
    mp.orn.data.spont = np.array([[5.0], [3.0]])
    rv = osm.RunVars(mp)
    osm.run_ORN_LN_sims(mp, rv)
    return mp, rv


def _steps(mp):
    """olfsysm's step counts (ModelParams::Time::steps_all() and start_step(),
    unsigned((t - pre_start) / dt)): (# of steps, step of `mp.time_start`)."""
    n_all = int((mp.time_end - mp.time_pre_start) / mp.time_dt)
    start_step = int((mp.time_start - mp.time_pre_start) / mp.time_dt)
    return n_all, start_step


def _step_of(mp, t):
    return int((t - mp.time_pre_start) / mp.time_dt)


@pytest.mark.parametrize('remove_pretime', [False, True])
def test_time_index_is_olfsysm_step_grid(remove_pretime):
    """Step k is at time_pre_start + k dt (time_start + k dt once the pretime is removed),
    with olfsysm's number of steps, and a window closed at the pulse's end holds the step
    at which olfsysm turns the odor off."""
    mp, rv = _orn_run()
    n_all, start_step = _steps(mp)
    # (olfsysm's defaults: -2 to 0.75 s at 0.5 ms; pretime -2 to -0.5 s; odor 0 to 0.5 s)
    assert (n_all, start_step) == (5500, 3000)
    first = 0
    if remove_pretime:
        osm.remove_all_pretime(mp, rv)
        first = start_step
    n = rv.orn.sims[0].shape[-1]
    assert n == n_all - first

    ts = get_time_index(mp, rv)
    assert ts.name == 'seconds'
    assert len(ts) == n
    t = ts.to_numpy()
    t0 = mp.time_start if remove_pretime else mp.time_pre_start
    np.testing.assert_allclose(t, t0 + np.arange(n) * mp.time_dt, rtol=0, atol=1e-12)
    # step times are only float noise away from these, and steps on round times are exact
    np.testing.assert_allclose(t, mp.time_pre_start + np.arange(first, n_all) * mp.time_dt,
        rtol=0, atol=1e-12
    )
    assert t[0] == t0
    assert np.isclose(t[-1], mp.time_end - mp.time_dt, rtol=0, atol=1e-12)
    on = _step_of(mp, mp.time_stim_start) - first
    off = _step_of(mp, mp.time_stim_end) - first
    assert t[on] == mp.time_stim_start == 0.0
    assert t[off] == mp.time_stim_end == 0.5

    closed = (t >= mp.time_stim_start) & (t <= mp.time_stim_end)
    assert np.array_equal(np.flatnonzero(closed), np.arange(on, off + 1))
    assert closed.sum() == 1001
    # the odor's steps (olfsysm: on for steps [stim start step, stim end step))
    half_open = (t >= mp.time_stim_start) & (t < mp.time_stim_end)
    assert np.array_equal(np.flatnonzero(half_open), np.arange(on, off))


@pytest.mark.parametrize('remove_pretime', [False, True])
def test_time_index_matches_olfsysm_odor_steps(remove_pretime):
    """The step at which the ORN rate first leaves its spontaneous level is labelled
    time_stim_start, and the step at which the (smoothed) odor input first falls, the
    step olfsysm turns the odor off at, is labelled time_stim_end."""
    # (a short pulse, so the smoothed odor input is still rising when it ends)
    mp, rv = _orn_run(time_stim_end=0.05)
    if remove_pretime:
        osm.remove_all_pretime(mp, rv)
    t = get_time_index(mp, rv).to_numpy()
    x = np.asarray(rv.orn.sims[0], dtype=float)[0]
    assert len(x) == len(t)

    onset = int(np.flatnonzero(x != x[0])[0])
    assert t[onset] == mp.time_stim_start

    # olfsysm: x(t) = (1 - mul) x(t - 1) + mul o(t), with o the smoothed odor input, so
    # d(t) = x(t) - x(t - 1) - (1 - mul) (x(t - 1) - x(t - 2)) = mul (o(t) - o(t - 1))
    mul = mp.time_dt / mp.orn.taum
    d = (x[2:] - x[1:-1]) - (1 - mul) * (x[1:-1] - x[:-2])
    d_step = np.arange(2, len(x))
    after = d_step > onset
    assert (d[after & (d_step < onset + 100)] > 0).all()
    falls = int(d_step[after & (d < 0)][0])
    assert t[falls] == mp.time_stim_end == 0.05
    assert falls - onset == 100

    closed = (t >= mp.time_stim_start) & (t <= mp.time_stim_end)
    assert closed[falls] and closed.sum() == 101


@pytest.mark.parametrize('remove_pretime', [False, True])
def test_time_index_legacy_linspace(remove_pretime):
    """`legacy_linspace=True` gives the pre-2026-11-13 times (n steps spread from the
    first time to time_end inclusive), which drop the step at the pulse's end from a
    closed window."""
    mp, rv = _orn_run()
    if remove_pretime:
        osm.remove_all_pretime(mp, rv)
    n = rv.orn.sims[0].shape[-1]
    t0 = mp.time_start if remove_pretime else mp.time_pre_start
    old = get_time_index(mp, rv, legacy_linspace=True)
    assert old.name == 'seconds'
    np.testing.assert_array_equal(old.to_numpy(), np.linspace(t0, mp.time_end, num=n))

    t = get_time_index(mp, rv).to_numpy()
    lag = (old.to_numpy() - t) / mp.time_dt
    np.testing.assert_allclose(lag, np.arange(n) / (n - 1), rtol=0, atol=1e-9)

    o = old.to_numpy()
    assert ((o >= 0) & (o <= mp.time_stim_end)).sum() == 1000
    # half-open windows with edges on the step grid select the same steps either way
    assert np.array_equal((o >= 0) & (o < mp.time_stim_end), (t >= 0) & (t < mp.time_stim_end))


def test_slice_to_odor_pulse_same_steps_with_either_times():
    """`slice_to_odor_pulse` gives the steps the odor is on, with the exact times and with
    the legacy ones (e.g. of saved dynamics)."""
    mp, rv = _orn_run()
    osm.remove_all_pretime(mp, rv)
    x = np.asarray(rv.orn.sims[0], dtype=float)
    n_all, start_step = _steps(mp)
    on = _step_of(mp, mp.time_stim_start) - start_step
    off = _step_of(mp, mp.time_stim_end) - start_step
    for legacy in (False, True):
        ts = get_time_index(mp, rv, legacy_linspace=legacy)
        da = xr.DataArray(x, dims=['glomerulus', 'time_s'], coords={'time_s': ts.to_numpy()})
        s = slice_to_odor_pulse(da, mp)
        assert s.sizes['time_s'] == off - on == 1000
        np.testing.assert_array_equal(s.values, x[:, on:off])

        s = slice_to_odor_pulse(da, mp, include_before=True)
        np.testing.assert_array_equal(s.values, x[:, :off])
