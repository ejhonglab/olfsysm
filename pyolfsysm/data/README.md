The data the model reads, shipped with pyolfsysm (`pyolfsysm.util.data_root`). Moved from
al_analysis's `al_analysis/data` (2026-11, with its history); the AL data and the other
records of data sent to collaborators stay there (see its `data/README.md`).

- `from_prat`
  - `2024-09-13`
    - `weight` # of T-bar-to-PSD connections between two neurons (same as for hemibrain
       input, though maybe called `x.weight` there). generally threshold at >= 5 (not
       sure if this is what I did previously)

  - `PNtoKC_connections_raw.xlsx` (currently under root `data/`. need to move to subdir
     here) was received 2023-08-08 on Slack. contains hemibrain data. his neuprint query
     already filtered stuff where `c.weight > 5` (well, according to the code snippet
     Pratyush gave me, but `df['c.weight'].min() == 4`, so maybe it was `>= 4`?)

  - `2025-09-17`
    - contains hemibrain connectome PN/KC/APL data, with synapses grouped into claws,
      but not yet grouped into PN boutons

    - PN->KC data should be final, but there may have been a bug (see Slack messages).
      might still contain enough information to merge with last version
      (`2025-09-11/PN-KC_Connectivity_Synapses.csv`), in order to get the x/y/z
      coordinates that one (and all prior from Prat) was missing? This data
      (`PN-KC_withPosns_v2.parquet`) was the only file actually sent on the 18th, not
      the 17th.

    - NOTE: can not currently commit `APL_Synapses_Pairwise_Distance.parquet`, as it is
      3.7GB in current form. Try to reduce it down and commit that? Should only need
      this data if we want to use his tree-distances for a version of the model with
      coupling between APL "compartments", which Tianpei will probably not finish
      implementing by end of his summer work. This file will currently be available on
      our NAS `matrix`, where Pratyush originally put it, and I also copied it under my
      `Tom` subdirectory on `tensor-archive`. (deleted that file and replaced w/ new
      version of same: `2025-09-24/APL_Pairwise_Node_Distances.parquet`)


- `sent_to_remy` (the model inputs / outputs among the data sent to Remy; the rest
  of `sent_to_remy` is in al_analysis)
  - `2025-03-18`
    - contains final megamat hemibrain model responses
      - wd20, but no connectome APL weights


- `sent_to_anoop` (the rest of `sent_to_anoop` is in al_analysis)
  - `2024-05-16`
    - final megamat uniform model responses (binarized) (+ parameters & wPNKC)
      (these responses also sent to Remy on Slack 2025-02-05).

      `test_uniform_paper_repro` checks we can reproduce.

    - also contains megamat hemibrain responses, but those are probably not quite the
      final versions, as I believe the wd20 code came later (that included some
      influence of synapse counts, rather than just counting each PN-KC pair as a claw).

- `from_remy` (`2024-06-05`, the odor metadata, is in al_analysis)
  - `megamat17`
    - downloaded from Dropbox folder `Remy/odor_space_collab/for_mendy` on 2024-07-05
      (though shouldn't have changed in a while, and should be same data Anoop is
      loading). only copied the KC soma subset of the data from that (large) Dropbox
      folder.

    - `per_fly`
      - copied + renamed from `for_mendy/data/megamat17`
      - used by:
        - `al_analysis.py`:
          - to make figs (#?) comparing model vs real KC correlations
          - any other figs? check.

        - this is also the subset of the Dropbox data that Anoop is loading

    - `respvec_concat`
      - copied from `for_mendy/concatenated_respvecs_by_imaging_type/kc_soma/megamat17/respvec_concat`

    - `README.html`
      - some of the folder / file structure descriptions may be wrong now, given my
        re-organization of the above


  - `response_rates`
    - was used (by `al_analysis.py`) to calculate new 0.0915 response rate target now
      used for all modelling (was previously using 0.1 target)

    - `refstim__ep_at_-3.0__median__0.120-0.130`
      - downloaded from Dropbox folder:
        `Remy/odor_space_collab/analysis_outputs/multistage/multiregion_data/response_breadth/by_trialavg_ref_stim/megamat`

      - received: 2024-04-04, via a Dropbox link on Slack (files may have been on
        Dropbox for longer)

    - `old_megamat_sparsities`
      - received: 2024-04-03, via Slack

      - only used to verify one of the files match the corresponding file from folder
        above (they did).


  - `2024-11-12`
    - old megamat data (from prior to final 4 flies I had been using until now). these
      experiments each only have a subset of the 17 megamat odors. Betty asked us to
      include all the megamat data for all paper KC correlations now, so I'll use these
      in combination with the 4 flies I had already been using

      - from `matrix/Remy-Data/projects/odor_space_collab/outputs/megamat`

    - `stim_rdms__iscell_good_xid0__correlation.xlsx` and `xrda_stim_rdm_concat__iscell_good_xid0__correlation.nc`
      - these are the two files Remy asked Anoop to use in an email the week before she
        oriented me to the old megamata data. I copied these in to check against
        correlations I compute from the old data.

      - from Remy's private repo `ejhonglab/OdorSpaceShare`, under `manuscript/data/by_imaging_panel/megamat_new_and_old_by_fly_17/kc_soma_nls`


- `from_matt`:
  - copied from my local copy of his code (`~/src/matt/matt-modeling/data/hemibrain`),
    which I had previously copied from his code on hal.

  - removed some larger files and some I don't use.


- `ann_model_outputs`
  - contents:
    - CSVs with ORN (Hallem) / KC responses and correlations.

  - used by:
    - `analyze_ann_outputs.py` (at the root of al_analysis)

  - generated by my script `save_ann_outputs.m`, which is currently under
    `~/Documents/MATLAB`. may commit here or (more likely) under my copy of her
    `mushroomBody` repo.
