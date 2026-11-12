"""The model's helpers that were in al_analysis's `al_util` (moved with the model, 2026-11).

Paths of the model's data (`data_root`, this package's own data directory), loaders of
Remy's KC data and of the committed ORN dF/F (`load_*_dff`), the megamat odor set, and the
keyword arguments that reproduce the paper's models (`paper_repro_kws`). al_analysis's
`al_util` still provides these names, forwarded here when pyolfsysm is installed (the
`al_analysis[model]` extra).
"""
import os
from pathlib import Path
from typing import List, Optional, Union
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# TODO replace w/ conditional import of seaborn / scipy as needed?
# if trying to run tests w/ valgrind, this import currently seems to raise:
# SystemError: initialization of beta_ufunc raised unreported exception
# from a from scipy.stats._boost.beta_ufunc import
if os.getenv('PYTHONMALLOC') != 'malloc':
    import seaborn as sns
else:
    warnings.warn('could not import scipy (and dependent packages), because '
        'PYTHONMALLOC=malloc'
    )

from hong2p import olf
from hong2p import io as hong2p_io
from hong2p.io import read_csv
from hong2p.olf import diag_panel_str, panel2name_order
from hong2p.pandas import addlevel
from hong2p.util import fly_cols
from hong2p.viz import rotate_xticklabels

try:
    from importlib.resources import files
# importlib.resources.files was added in Python 3.9 (backport for Python 3.8)
except ImportError:
    from importlib_resources import files


# TODO can i also set weight_divisor=20, _drop_glom_with_plus=False here, or need
# to not do either of those for uniform case? check latter in uniform case?
# (probably can't. uniform test at least loops over the two values of _drop_glom...
# currently, though maybe doesn't need to)
paper_repro_kws = dict(
    target_sparsity=0.0915, drop_kcs_with_no_input=False, hardcode_initial_sp=True,
    sp_lr_coeff=10.0, max_iters=10,
    # TODO rename hardcode_initial_sp as repro...<something> and then have it imply both
    # of these?
    binary_search_on_overshoot=False, linear_lr_falloff=False
)
hemibrain_paper_repro_kws = dict(
    # I would not normally recommend you hardcode any of these except perhaps
    # weight_divisor=20. The defaults target_sparsity=0.1 and
    # _drop_glom_with_plus=True should be fine.
    weight_divisor=20, _drop_glom_with_plus=False, **paper_repro_kws
)


def _have_fly_cols(data: Union[pd.DataFrame, pd.Series]) -> bool:
    """Returns whether `data` has all `fly_cols` as index level names.

    For DataFrame input, also checks column level names and column labels.
    """
    if isinstance(data, pd.DataFrame):
        names_lists = [data.columns.names, data.columns, data.index.names]
    else:
        assert isinstance(data, pd.Series)
        names_lists = [data.index.names]

    return any(all(x in names for x in fly_cols) for names in names_lists)


megamat_odor_names = set(panel2name_order['megamat'])

def odor_is_megamat(odor: str) -> bool:
    """Takes odor str (with conc) like 'va @ -3' to whether it's among 17 megamat odors.
    """
    return olf.parse_odor_name(odor, require_conc=False) in megamat_odor_names


# the model's data, shipped in this package (pyolfsysm/data; al_analysis/data until 2026-11)
# TODO move to top (+ caps)
data_root: Path = files('pyolfsysm') / 'data'

sent_to_remy: Path = data_root / 'sent_to_remy'

# TODO TODO regenerate these (and recommit in new dir, leaving old data), so we also
# have newer parquet / response_calc_json / etc outputs?
# TODO TODO add tests we can re-run al_analysis (w/ appropriate response calcs) and
# regenerate this as well as similar paper outputs (w/ older response calc)
#
# Contains subdirs [kiwi_control|megamat|validation2]_signed-max, each with dF/F
# CSV/pickle outputs and plots committed.
signedmax_orn_dff_dir: Path = sent_to_remy / '2025-09-30_tom_orn_data_signed-max'

# TODO TODO add tests that we can successfully load data with all these fns (both dff
# and est spike deltas). will be mainly important when trying to move data handling from
# stuff in an editable repo on disk to files provided by importlib_resources.
#
# TODO add similar fns for megamat/control (and both for paper response calc and newer
# signedmax one)
# TODO use in tests to check we can repro mean_est_spike_deltas for all, and everything
# downstream
def load_natmix_dff(**kwargs) -> pd.DataFrame:
    """Loads committed kiwi/control (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'kiwi_control_signed-max'

    # TODO also load parquet (and prefer that, if we have it). need to regen and commit
    # new outputs first
    # TODO factor out this name to share w/ mb_model/al_analysis
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)

    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'kiwi', 'control'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
# TODO TODO add flag to load old stat computation version of this (for recreating
# outputs), or just leave that to the spike delta version of this fn? (can't pass into
# model_mb_responses tho..., at least without changing that)
# (that fn `mb_model.megamat_orn_deltas` also does not have a flag to load old data.
# see hemibrain repro test, which i assume references some committed version, presumably
# of the ORN deltas. oh nvm i have the paper_megamat_orn_deltas fn for that. still want
# a corresponding fn / flag here, which should be able to reproduce those)
# TODO TODO TODO presumably i have some output committed under data/ which is based to
# point to as "the paper version" of the dF/F. what is that version? or should i just
# recompute w/ mean stat? (again see scripts/repro_remy_paper_modeling.py, which was
# never completely successful. i think
# dasta/sent_to_remy/2023-10-29/pebbled_ij_certain-roi_stats.csv should do)
def load_megamat_dff(**kwargs) -> pd.DataFrame:
    """Loads committed megamat (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'megamat_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'megamat'
    }
    return df


# TODO assert matches subset of
# data/internal/for_dff_to_spiking_fn/ij_certain-roi_stats.parquet?
# TODO use in tests
def load_validation2_dff(**kwargs) -> pd.DataFrame:
    """Loads committed validation2 (+ diagnostic panel) ORN dF/F (signed max response).

    Args:
        **kwargs: passed to `read_csv`
    """
    data_dir = signedmax_orn_dff_dir / 'validation2_signed-max'
    csv_path = data_dir / 'ij_certain-roi_stats.csv'
    df = read_csv(csv_path, **kwargs)
    assert set(df.index.get_level_values('panel').unique()) == {
        diag_panel_str, 'validation2'
    }
    return df


# TODO implement this one too (may need to recalc and commit? or instead have flags for
# validation / megamat load fns?
# TODO rename load_paper_megamat_dff?
#def load_remypaper_dff(**kwargs) -> pd.DataFrame:


# TODO also anchor path to script dir? would only be to support running from elsewhere,
# which i prob don't care about
remy_data_dir: Path = data_root / 'from_remy'

# NOTE: contains sparsities in top level CSVs, as well as individual fly binarized
# responses in subdirectories
#
# downloaded from Dropbox folder:
# Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/\
#     response_breadth/by_trialavg_ref_stim/megamat
# (Remy sent me a link on Slack 2024-04-04)
remy_sparsity_dir: Path = (remy_data_dir /
    'response_rates/refstim__ep_at_-3.0__median__0.120-0.130'
)

# TODO refactor to share?
remy_date_col: str = 'date_imaged'
# she uses 'fly_num' same as I do, to number flies within each day (i.e. the numbers not
# unique across days). these two can generally be used to compute/lookup values for
# remy_fly_id.
remy_fly_cols: List[str] = [remy_date_col, 'fly_num']

# 0, 1, ..., 3 (or all 0 in one of the CSVs, by accident, but that CSV redundant
# anyway)
remy_fly_id: str = 'acq'

# e.g. '1-5ol @ -3.0'
remy_odor_col = 'stim'

remy_conc_str = ' @ -3.0'
# TODO refactor to share w/ some similar fns?
def _strip_remy_concs(x: Union[pd.Series, pd.Index]) -> Union[pd.Series, pd.Index]:
    assert x.str.endswith(remy_conc_str).all()
    return x.str.replace(remy_conc_str, '', regex=False)


# contains CSVs remy made from the pickles she sent earlier under 'by_acq'
# (which i couldn't load)
remy_binary_response_dir: Path = remy_sparsity_dir / 'by_acq_csvs'

# looks like the top-level sparsity CSVs are computed from peak amplitude alone
# (rather than options involving "std"), so only going to load these pickles
# (out of the 3 options in each fly_dir)
remy_fly_binary_response_fname: str = 'df_stim_responders_from_peak.csv'


# TODO add verbose kwarg and set False when calling for debug purposes from kc response
# loading code
def load_remy_fly_binary_responses(fly_sparsity_path: Path,
    acq_ledger: Optional[pd.DataFrame] = None, *, reset_index: bool = True,
    _seen_date_fly_combos=None) -> pd.DataFrame:
    """Loads single fly (tab separated) CSV to boolean (cell X odor) DataFrame.

    Args:
        fly_sparsity_path: CSV in a directory named like '<date>__fly<NN>__...'

        acq_ledger: if passed in, has `index.names == remy_fly_cols`, and has
            `remy_fly_id in df.columns`

            maps ['date', 'fly_num'] to int 'acq' (`remy_fly_id`) (0-indexed), which is
            added as a row index level.

        reset_index: if True, only return int 'cell' index. otherwise, retains all
            metadata columns in source CSV (i.e. row index names in the DataFrame)

        _seen_date_fly_combos: internal use only (set of (date, fly_num) loaded so far,
            to check no fly is loaded twice. updated in place.)
    """
    fly_dirname = fly_sparsity_path.parent.name

    if hong2p_io.verbose:
        print(fly_sparsity_path)

    # sep/index_col params from snippet Remy sent me
    #
    # there do only seem to be 5 index columns in these CSVs. all columns after are
    # odor names (e.g. '1-5ol @ -3.0')
    fly_df = pd.read_csv(fly_sparsity_path, sep='\t', index_col=list(range(5)))

    # 17 columns, 1 for each megamat odor.
    # index levels: cells_level_0 iscell iscell_xid0 xid0 embedding0
    #
    # iscells seems to all be 1, so no need to subset based on this/keep
    assert fly_df.index.get_level_values('iscell').astype(bool).all()

    # TODO delete (probably unimportant anyway, as no flies have only some elements
    # True, so this never has any info...)
    # TODO why is this true for some but not all?
    '''
    try:
        # not sure how this is different (from above), but at least it's also all True
        assert fly_df.index.get_level_values('iscell_xid0').all()
        print('iscell_xid0 assertion pass')
    except AssertionError:
        # NOTE: all False for this one fly (2022-11-10/1 [/megamat0_dsub3])
        # (all True for the other 3 final megamat flies)
        print('iscell_xid0 assertion fail!')
    '''

    # so i guess if cells were subset, they were renumbered (or more likely cells
    # were not subset at this point)
    cells_level_0 = fly_df.index.get_level_values('cells_level_0')
    assert pd.RangeIndex(cells_level_0.max() + 1).equals(cells_level_0)

    # don't think there's anything worth checking about 'embedding0'

    # TODO TODO do something w/ 'xid0' index level? as part of comparison wrt
    # response data (the stuff loaded in KC corr loading fn)?
    # TODO see if which xid0 cells have are consistent w/ those in responses i'm
    # loading elsewhere?
    #import ipdb; ipdb.set_trace()

    assert set(np.unique(fly_df.values)) == {0, 1}
    fly_df = fly_df.astype(bool)

    # TODO can i recreate these from the data Remy sent to anoop (what should
    # still be the final 4 megamat KC flies), which is in our Dropbox at:
    # Remy/odor_space_collab/for_mendy/data/megamat17
    # (using thresholds from CSV above)?

    # NOTE: there must be double counted cells. don't think Remy makes any attempt
    # to de-duplicate them.
    n_cells = len(fly_df)

    silent_cells = ~ (fly_df.any(axis='columns'))
    assert len(silent_cells) == n_cells
    n_silent = silent_cells.sum()

    parts = fly_dirname.split('__')
    assert len(parts) >= 3
    date_str, fly_str = parts[:2]

    date = pd.Timestamp(date_str)

    # e.g. 'fly01' -> 1
    fly_num = int(fly_str.replace('fly', ''))

    if _seen_date_fly_combos is not None:
        # we already know the ledger doesn't have duplicate flies (or 'acq' values),
        # from checks in main caller of this code.
        #
        # now we are just checking the individual files we are loading also don't ever
        # refer to the same fly in >1 of the files (could accidentally be 2 recordings
        # of same fly).
        assert (date, fly_num) not in _seen_date_fly_combos
        _seen_date_fly_combos.add( (date, fly_num) )

    if reset_index:
        fly_df = fly_df.reset_index(drop=True)
        # TODO what was it before? (maybe still 'cell', the reset_index() call seems to
        # clear this)
        # TODO still do this in else case?
        fly_df.index.name = 'cell'

    if acq_ledger is not None:
        curr_acq = acq_ledger.loc[(date, fly_num), remy_fly_id]
        fly_df = addlevel(fly_df, remy_fly_id, curr_acq)

        msg = f'{fly_dirname} ({remy_fly_id}={curr_acq})'
    else:
        msg = f'{fly_dirname}'

    if hong2p_io.verbose:
        print(msg)

        # silent = responds to no odors
        #
        # NOTE: not sure if silent/not cells are equally represented (i.e. equally
        # double counted), so maybe fraction of silent cells is off?
        print(f'{n_silent} silent / {n_cells} cells ({(n_silent / n_cells):.3f})')
        print()

    return fly_df


_remy_megamat_kc_binary_responses = None
def load_remy_megamat_kc_binary_responses() -> pd.DataFrame:
    """Loads Remy's binarized megamat KC responses, for each fly.

    Result is cached, so later calls return the same object (which should not be
    modified).

    Returns boolean DataFrame of cells X odors, with `remy_fly_id` ('acq') and 'cell'
    row index levels, and megamat odor names (without concentrations) as columns.
    """
    global _remy_megamat_kc_binary_responses
    if _remy_megamat_kc_binary_responses is not None:
        return _remy_megamat_kc_binary_responses

    # should be able to map acq to fly CSV dir using first 4 cols:
    # ['acq' (=remy_fly_id), 'date_imaged' (=remy_date_col), 'fly_num',
    # 'thorimage_name'] (last 2 cols not important)
    acq_ledger = pd.read_csv(remy_sparsity_dir / 'df_acqs.csv')
    acq_ledger[remy_date_col] = pd.to_datetime(acq_ledger[remy_date_col])

    acq_ledger = acq_ledger.set_index(remy_fly_cols, verify_integrity=True)
    assert not acq_ledger[remy_fly_id].duplicated().any()

    if hong2p_io.verbose:
        print()
        print('loading Remy megamat KC binarized responses (for sparsity + S1C):')

    _seen_date_fly_combos = set()
    fly_binary_responses_list = []

    for fly_sparsity_path in sorted(
            remy_binary_response_dir.glob(f'*/{remy_fly_binary_response_fname}')
        ):

        fly_df = load_remy_fly_binary_responses(fly_sparsity_path, acq_ledger,
            _seen_date_fly_combos=_seen_date_fly_combos
        )
        fly_binary_responses_list.append(fly_df)

    binary_responses = pd.concat(fly_binary_responses_list, verify_integrity=True)
    binary_responses.columns.name = remy_odor_col

    binary_responses.columns = _strip_remy_concs(binary_responses.columns)
    assert binary_responses.shape[1] == len(megamat_odor_names)

    binary_responses = binary_responses.sort_index()

    assert _remy_megamat_kc_binary_responses is None
    _remy_megamat_kc_binary_responses = binary_responses

    return binary_responses


def remy_megamat_sparsity() -> float:
    """Returns mean response rate in Remy's final megamat KC data.

    Weights each fly (of 4) equally, regardless of number of cells per fly.
    """
    # md5 of this and
    # response_rates/old_megamat_sparsities/refstim__ep_at_-3.0__median__0.120-0.130
    # match, so they are the same.
    megamat_csv = remy_sparsity_dir / 'tidy_sparsities_ascending.csv'

    # columns: ['acq', 'stim', 'peak_amp_thresh', 'max_sparsity', 'min_sparsity',
    # 'kc_soma']
    #
    # 'kc_soma' should be between or equal to '[max|min]_sparsity' limits (which should
    # correspond to range of thresholds referenced in path name). 'kc_soma' is what Remy
    # said I should use.
    df = pd.read_csv(megamat_csv)
    sparsity_col = 'kc_soma'

    # TODO what are units/meaning of "0.120-0.130" in path? how does it relate to
    # 'peak_amp_thresh' col (min=0.95, max=1.75)?

    # TODO delete? only important column in this should match the corresponding (diff
    # named) column in other csv
    megamat_csv2 = remy_sparsity_dir / 'df_sparsity_recomputed.csv'
    df2 = pd.read_csv(megamat_csv2)
    sparsity_col2 = 'sparsity_from_peak_thr'

    assert df2[remy_odor_col].equals(df[remy_odor_col])
    assert df2[sparsity_col2].equals(df[sparsity_col])
    # df2[remy_fly_id] won't match. df2[remy_fly_id].unique() == [0] (through a
    # mistake on her end, Remy created this CSV w/ ID 0 for all the data in this CSV,
    # even though it actually has data from 4 flies. other CSV includes the correct IDs,
    # and rest of data matches)
    del df2

    assert not df.isna().any().any()
    # TODO delete? ideally, i'd use more flies, but i'm not sure if she calculated the
    # sparsity (esp not pegged to one odor, if this was) for the other partial-panel
    # flies she's now using for all megamat corr calculations in paper
    n_final_megamat_kc_flies = 4
    assert df[remy_fly_id].nunique() == n_final_megamat_kc_flies
    #
    assert set(x.stim.nunique() for _, x in df.groupby(remy_fly_id)) == {
        len(megamat_odor_names)
    }

    df[remy_odor_col] = _strip_remy_concs(df[remy_odor_col])

    # low->high response rate
    odors_in_sparsity_order = df.groupby(remy_odor_col, sort=False)[sparsity_col].mean(
        ).sort_values().index

    df = df.sort_values(remy_odor_col, kind='stable',
        key=lambda x: x.map(odors_in_sparsity_order.get_loc)
    )
    # TODO just force ep to be last? sort order actually diff (-> imply data is diff?)?
    # maybe just slightly though... (?)

    # TODO TODO TODO save this again + compare to model response rates
    # (or did i have another plot directly making that comparison?)
    plot = False
    if plot:
        fig, ax = plt.subplots()
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, hue=remy_fly_id)
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)
        # https://stackoverflow.com/questions/44620013
        ax.get_legend().set_title('fly')

        # should match preprint S1B
        fig, ax = plt.subplots()
        # should have the same 95% CI as reported in paper
        sns.pointplot(ax=ax, data=df, x=remy_odor_col, y=sparsity_col, color='black')
        ax.set_xlabel('odor')
        ax.set_ylabel('response rate')
        ax.set_ylim([0, 0.2])
        rotate_xticklabels(ax, 90)

        # TODO put mean_sparsity in xlabel, like in sparsity_per_odor plots
        # (silent cells too?) (delete comment? still care?)

        # TODO save plots? return figs?

    # TODO TODO recreate model KC response rates plot, on model tuned on hallem, but
    # subset to 17 megamat odors (using this code too, in case there was an issue w/
    # other code?) (done? still want?)

    mean_of_odor_means = df.set_index([remy_odor_col, remy_fly_id]).groupby(
        level=remy_odor_col, sort=False).mean()[sparsity_col].mean()

    mean_sparsity = df[sparsity_col].mean()
    # as expected from the math. numerically slightly (but not consequentially) diff
    assert np.isclose(mean_sparsity, mean_of_odor_means)

    binary_responses = load_remy_megamat_kc_binary_responses()
    assert set(binary_responses.columns) == set(df[remy_odor_col])

    recomputed_df = binary_responses.groupby(remy_fly_id).mean().melt(
        value_name=sparsity_col, ignore_index=False).reset_index()

    recomputed_df = recomputed_df.set_index([remy_fly_id, remy_odor_col],
        verify_integrity=True).sort_index()

    df = df.set_index([remy_fly_id, remy_odor_col], verify_integrity=True).sort_index()
    # changing index affects outcome, but only in irrelevant numerical way
    assert np.isclose(df[sparsity_col].mean(), mean_sparsity)

    for acq, recomputed_fly_df in recomputed_df.groupby(remy_fly_id):
        recomputed_ser = recomputed_fly_df.droplevel(remy_fly_id).iloc[:, 0]
        remy_fly_ser = df.loc[acq, sparsity_col]
        assert np.allclose(recomputed_ser, remy_fly_ser)

    # NOTE: can't just do binarized_responses.mean().mean(), as there are different
    # numbers of cells for different flies, so we need to average within each fly first
    # (to not weight some flies more than others)
    assert np.isclose(recomputed_df[sparsity_col].mean(), mean_sparsity)

    return mean_sparsity
