This repository has the model's C++ code (libolfsysm, bindings/: the compiled `olfsysm`
module) and, since 2026-11, its Python side, the package pyolfsysm (pyolfsysm/: formerly
al_analysis's mb_model.py, wpnkc_shuffles.py and the model-only part of al_util.py, with
the data the model reads in pyolfsysm/data), scripts that run the model (scripts/) and the
tests of both (test/). al_analysis installs it with its optional `model` extra, and its
al_analysis.mb_model / wpnkc_shuffles only forward to pyolfsysm. pyolfsysm must not import
al_analysis. pyolfsysm.readout, kc_stats, effects and tuning came from the KC-remapping
experiment queue (model_experiment_queue, 2026-11-12), which keeps wrappers with its own
defaults; their outputs are frozen in test/test_frozen_from_queue.py (inputs and expected
outputs in test/test_data/frozen_from_queue/), so a change to them that changes those
numbers needs Tom's attention, as for the model.

Tests for the C++ code and pyolfsysm are in test/ (test/test_mb_model.py and the other
model tests). Any changes to the model should preserve existing behavior, and write
additional tests to verify this unless there seems to be some bug or misunderstanding in
the current implementation. Bring my attention to any such cases when evaluating current
tests or implementing new ones. Run pytest from this repository's root with the
environment variable QUICK=1 set, unless extensively testing changes to the model, to skip
most of the different model cases. There is no venv here: use the one that has this
package installed editable (e.g. ../al_analysis/venv; check with `pip show olfsysm`).

The extension is a C++ build: re-run `pip install -e .` (with each venv that has it
installed editable) after any C++ change, and commit C++ changes before producing outputs
(pyolfsysm refuses to run the model with a build whose C++ sources are dirty or differ
from the checkout's; Python-only commits need no rebuild). See the README.

Regarding the python bindings, to the extent it's relevant here:
For now, try to preserve support for Python 3.8 (with last pandas 1.5), as well as the
current ./venv/ python (should be 3.14 with pandas 3). Do this by either
asking-for-forgiveness-not-permission (e.g. try/except handling), or explicit checking
of versions if necessary. If code only exists to support old pandas 1.5 (or other old py
thon 3.8 stuff), add a comment saying which old thing that part of the code is for.
