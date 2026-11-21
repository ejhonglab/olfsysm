# Male CNS model tables (left and right calyx)

The model's wiring of the male CNS for `connectome_wPNKC(connectome='malecns-left' |
'malecns-right', prat_claws=True)` and its APL weights for `connectome_APL_weights` (and so
`fit_mb_model(pn2kc_connections='malecns-left' | 'malecns-right', one_row_per_claw=True,
prat_claws=True, ...)`).

Source: hong-data commit 8dc9cea (github.com/ejhonglab/hong-data),
reference_data/connectome/malecns-<side>_<connection>_model.parquet, copied unchanged. Made there
by reference_data/make_connectome.py from Pratyush's male CNS tables, run 2026-10-25-17-31-06 (the
KC-remapping queue's shared directory, data/Male-CNS_Tom/), with the steps of olfsysm's hemibrain
loader (`connectome_wPNKC` with `prat_claws=True, prat_boutons=True,
multibouton_claws='argmax'`; `connectome_APL_weights`), which make_connectome.py checks give
the hemibrain's wiring and APL weights exactly, applied to the male CNS with the choices of the
queue's E97: no ROI rule, and the PN types' glomeruli renamed as the hemibrain's (VC3 and VC5 ->
VC3, VM6 -> VC5). Body IDs as int64. Pratyush's raw male CNS tables are not copied here (about
1 GB); the raw tables of hong-data (`*_raw.parquet`) are not needed by the model.

Per side, five tables, one line per claw (or bouton) of the model:

- `pn_kc`: each claw (`kc_id`, `claw_id`, `kc_type`, `kc_subtype`, `claw_x/_y/_z` in µm
  centered on the middle of each range, `claw_compartment`), its PN (`pn_id`, connectome
  `glomerulus`, the model's `model_glomerulus`, `pn_type`, `pn_bundle`) and its one bouton
  (`bouton_id`: of its PN's boutons it has synapses from, the one with the most, ties to the
  lowest numbered; `bouton_x/_y/_z`, `bouton_compartment`), `weight` 1 and `synapses` (the
  claw's synapses from its PN).
- `kc_apl`, `apl_kc`: the synapses between each claw and APL (`weight`; 0 where none).
- `pn_apl`, `apl_pn`: the synapses between each of the model's boutons and APL.

Claws: 5 or more synapses from one PN of the model's glomeruli (claws with input from more than
one such PN dropped), of KCs of the three types (g, ab, a'b'). Left: 10,077 claws of 1,864
KCs on 471 boutons, 53 glomeruli; right: 10,008 claws of 1,884 KCs on 434 boutons, 53
glomeruli (the hemibrain: 9,867 claws of 1,732 KCs on 387 boutons with 'argmax').
Compartments (`compartment` index level) by the hemibrain's rule (0 within half the largest
distance of any claw from the claws' centroid, else 1), over each side's own claws; the
loader recomputes them and checks they equal the tables'. Whether the male CNS should have a
rule of its own is for Yang to decide.

Checked (olfsysm, 2026-11-20): an independent recount from Pratyush's tables gives the same
claws, boutons and APL weights on both sides, and the claws per glomerulus (claws, PNs,
boutons, KCs) and per KC type are E97's counted claws (results/E97_glomerulus_claw_shares,
a1_counts.csv `main` at threshold 5, a1_kcs.csv).

sha256:
    2e62b78bdf2d672c0b92bf1a9cbc2fc8c72d63ff993c6705a520965abb874cd9  malecns-left_apl_kc_model.parquet
    93b920c77e0bcbe48b64c23fb4047037cf1484462da2c553906075dccc614f44  malecns-left_apl_pn_model.parquet
    93cdcf7d5952bc0f35fe577108cda87eb02cd3de9e98a10eba36c93311610bfd  malecns-left_kc_apl_model.parquet
    6009245561d4dfd1ddb2dec39beeaf53f87e7712b0ddd7ff93edcada29c03af2  malecns-left_pn_apl_model.parquet
    cc1c6b1387df6c3206baa5231fb76a2ce74864d8db5687df7ce47f30531b32de  malecns-left_pn_kc_model.parquet
    1158863c26076ebf9337735d873c47296e0d83274a86a87939e7f414b5c61eb3  malecns-right_apl_kc_model.parquet
    23539698283d55fc199021676045001dbb2b3af0f47b9ebbf6434d38dea720e4  malecns-right_apl_pn_model.parquet
    533d46c8f101e3219ba8101595e7b04e4d96c971747f0ace551b602eab314de4  malecns-right_kc_apl_model.parquet
    1551ff775c13a2b78e158a761c8b5b6a7bcd48bc16a8e7710dd24ebf81ad28de  malecns-right_pn_apl_model.parquet
    f46ca877a81ad3d761616fd1b88c9503016ec7bb2d2bc60f2c221559aa49d344  malecns-right_pn_kc_model.parquet
