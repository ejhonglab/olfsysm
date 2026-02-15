#include "olfsysm.hpp"

#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <iostream>
#include <functional>
#include <sstream>

#include <unordered_set>
#include <iomanip>
#include <set>
#include <map>
#include <cmath>
#include <stdexcept>

/* So code can be compiled single threaded, to support debugging.
 * Only other OMP references should be in the preprocessor directives, which I think can
 * just be ignored (though that will generate compilation warning, which is good).
 * https://stackoverflow.com/questions/7847900 */
#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
#endif

/* Include cnpy.h only if we we have that .h file available (and referenced properly in
 * compilation). Optional library added as a git submodule, for writing dynamics
 * directly to .npy files for reading in Python. See README. */
#if defined __has_include
# if __has_include ("cnpy.h")
#  include "cnpy.h"
# endif
#endif

Logger::Logger() {}
Logger::Logger(Logger const&) {
    throw std::runtime_error("Can't copy Logger instances.");
}
void Logger::operator()(std::string const& msg) const {
    std::lock_guard<std::mutex> lock(mtx);
    if (!fout) return;
    fout << msg << std::endl;
}
void Logger::operator()() const {
    this->operator()("");
}
void Logger::redirect(std::string const& path) {
    std::lock_guard<std::mutex> lock(mtx);
    fout.close();
    fout.open(path, std::ofstream::out | std::ofstream::app);
}
void Logger::disable() {
    std::lock_guard<std::mutex> lock(mtx);
    fout.close();
}

/* Concatenate all the given arguments, which can be of any type, into one
 * string. No separator is placed between the arguments! */
template<class... Args>
std::string cat(Args&&... args) {
    std::stringstream ss;
    (ss << ... << std::forward<Args>(args));
    return ss.str();
}

/* For random number generation. */
// NOTE: g_randdev only used in definition of g_randgen
// TODO TODO would this not need a mutex across threads? is one in use somewhere?
// diff seed for each thread too?
// https://stackoverflow.com/questions/21237905
thread_local std::random_device g_randdev;
thread_local std::mt19937 g_randgen{g_randdev()};

// TODO move this assertion-replacement stuff into it's own header file -> include in
// all build system stuff
/* To replace usage of builtin C++ assert, which has two problems:
 * 1. it triggers an abort(), which does not seem like it can be easily handled by
 *    pytest (for tests in `al_analysis/test/test_mb_model.py`), potentially screwing up
 *    pytest state and subsequent test executation with abrupt exit.
 *
 *    https://p403n1x87.github.io/running-c-unit-tests-with-pytest.html#in-the-wild
 *    may have some more complicated alternative approaches to continuing testing after
 *    an abort, by using subprocesses. couldn't find a pre-existing extension with this
 *    purpose, but pytest-xdist may be helpful for running tests in subprocesses?
 *
 * 2. it requires special undefing NDEBUG defined by current build system, which is
 *    necessary to suppress some spurious assert failures that would often otherwise get
 *    tripped within the Eigen code
 */
// TODO log as custom exception type (in pybind11 bindings file) to indicate it's an
// error from within c++ code (or maybe convert to AssertionError?)?
// see: https://pybind11.readthedocs.io/en/stable/advanced/exceptions.html
// TODO also include the print_stack_trace(2) call source does after part i copied?
// (would need to pull in more code, and may want to do it a diff way...)
// TODO add (sep version?) w/ optional message
//
// NOTE: will still just abort if any of these fails within multithreaded sections!!!
//
// NOTE: could potentially use __FUNCTION__ instead of __func__, if later is ever
// unavailable (may be the case on some compilers?). see:
// https://stackoverflow.com/questions/4384765
//
// Adapted from indiv's answer in https://stackoverflow.com/questions/37473
#define check(expr) do {                                                               \
    if (!(expr)) {                                                                     \
        throw std::invalid_argument(cat(                                               \
            __FILE__, ":", __LINE__, " in `", __func__, "` check `", #expr, "` failed" \
        ));                                                                            \
    }                                                                                  \
} while(0)

ModelParams const DEFAULT_PARAMS = []() {
    ModelParams p;

    p.time.pre_start  = -2.0;
    p.time.start      = -0.5;
    p.time.end        = 0.75;
    p.time.stim.start = 0.0;
    p.time.stim.end   = 0.5;
    p.time.dt         = 0.5e-3;

    p.orn.taum             = 0.01;
    p.orn.n_physical_gloms = 51;

    p.ln.taum   = 0.01;
    p.ln.tauGA  = 0.1;
    p.ln.tauGB  = 0.4;
    p.ln.thr    = 1.0;
    p.ln.inhsc  = 500.0;
    p.ln.inhadd = 200.0;

    p.pn.taum       = 0.01;
    p.pn.offset     = 2.9410;
    p.pn.tanhsc     = 5.3395;
    p.pn.inhsc      = 368.6631;
    p.pn.inhadd     = 31.4088;
    p.pn.noise.mean = 0.0;
    p.pn.noise.sd   = 0.0;
    p.pn.apl_taum   = 0.05;
    p.pn.tau_apl2pn = 0.01;

    // If this remains 0, will default to get_ngloms below.
    p.pn.n_total_boutons = 0;

    /* bouton related model param initialization*/
    // TODO delete this one? or repurpose in correct spot?
    p.pn.pn_apl_tune     = false;
    //
    p.pn.preset_Btn      = false;
    p.pn.preset_wAPLPN   = false;
    p.pn.preset_wPNAPL   = false;

    p.kc.N                     = 2000;
    p.kc.nclaws                = 6;
    p.kc.uniform_pns           = false;
    p.kc.pn_drop_prop          = 0.0;
    p.kc.preset_wPNKC          = false;
    p.kc.seed                  = 0;
    p.kc.tune_apl_weights      = true;
    p.kc.preset_wAPLKC         = false;
    p.kc.preset_wKCAPL         = false;

    // TODO just always unconditionally do this? (it's just temporary, for first phase
    // of fit_sparseness, where thresholds are picked with what should be silent APL [or
    // at least where APL not influencing KC activity]) (won't prevent needing to set
    // threshold high for threshold-picking phase of fit_sparseness though, since that
    // is to prevent spiking from resetting KCs, not just to prevent APL activity)
    p.kc.zero_wAPLKC           = false;

    // TODO rename to indicate this is controlling whether spiking is required to drive
    // APL (and where APL input comes from)
    p.kc.pn_claw_to_APL        = false;
    p.kc.ignore_ffapl          = false;
    p.kc.fixed_thr             = 0;
    p.kc.n_claws_active_to_spike = -1;
    // TODO TODO ever make sense to have this false? delete? set default true at least?
    // maybe when using threshold per-claw it might make more sense (idk...)?
    // (only seem to set it False in mb_model.py when using homeostatic thresholds, and
    // i could move that logic in here)
    p.kc.add_fixed_thr_to_spont= false;
    p.kc.use_fixed_thr         = false;
    p.kc.use_vector_thr        = false;
    p.kc.use_homeostatic_thrs  = true;
    p.kc.thr_type              = "";
    p.kc.sp_target             = 0.1;
    p.kc.sp_factor_pre_APL     = 2.0;
    p.kc.sp_acc                = 0.1;
    p.kc.sp_lr_coeff           = 10.0;

    // Will probably remain necessary to set this true, to exactly replicate Matt's
    // outputs, and paper outputs. Matt just did this to simplify and hasten convergence
    // in the particular case he was working with, but could also have been done by
    // hardecoding an initial weight scale stepsize, rather than pretending we start
    // with a particular response rate across KCs...
    //
    // For all other cases, we should try to keep this false.
    p.kc.hardcode_initial_sp   = false;

    p.kc.max_iters             = 10;
    p.kc.apltune_subsample     = 1;

    // TODO doc how each of these are diff (w/ units if i can). not currently mentioned
    // in .hpp file
    p.kc.taum                  = 0.01;
    p.kc.apl_Cm                = 0.1;
    p.kc.apl_taum              = 0.05;
    p.kc.tau_apl2kc            = 0.01;

    p.kc.apl_coup_const        = -1;
    p.kc.comp_num              = 0;

    p.kc.tau_r                 = 1.0;
    // olfsysm.hpp says that setting this to 0 should disable synaptic depression
    // (tau_r above is another parameter for synaptic depression)
    p.kc.ves_p                 = 0.0;

    p.kc.save_vm_sims          = false;
    p.kc.save_spike_recordings = false;
    p.kc.save_nves_sims        = false;
    p.kc.save_inh_sims         = false;
    p.kc.save_Is_sims          = false;
    p.kc.save_claw_sims        = false;

    p.ffapl.taum         = p.kc.apl_taum;
    p.ffapl.w            = 1.0;             // appropriate for LTS
    p.ffapl.coef         = "lts";
    p.ffapl.zero         = true;
    p.ffapl.nneg         = true;
    p.ffapl.gini.a       = 1.0;
    p.ffapl.gini.source  = "(-s)/s";
    p.ffapl.lts.m        = 1.5;

    p.kc.kc_ids.clear();
    p.kc.wPNKC_one_row_per_claw = false;
    p.kc.allow_net_inh_per_claw = false;

    return p;
}();

/* (utility) Split a string by commas, and fill vec with the segments.
 * vec must be sized correctly! */
void split_regular_csv(std::string const& str, std::vector<std::string>& vec);

/* The exponential ('e') part of the smoothts MATLAB function included in the
 * Kennedy source.
 * Instead of returning the smoothed matrix, it smooths it in-place. */
void smoothts_exp(Matrix& vin, double wsize);

/* Fill out with numbers generated by rng. */
void add_randomly(std::function<double()> rng, Matrix& out);

/* Calculate the Gini-type FFAPL coefficient. */
double ffapl_coef_gini(ModelParams const& p,
        Column const& pn, Column const& pn_spont);

/* Calculate the lifetime sparseness FFAPL coefficient. */
double ffapl_coef_lts(ModelParams const& p,
        Column const& pn, Column const& pn_spont);

/* Build PNKC connectivity matrix w in place, with glom choice weighted by cxnd
 * and drop_prop (see ModelParams). */
void build_wPNKC_from_cxnd(
        Matrix& w, unsigned nc, Row const& cxnd, double drop_prop);

/* Build wPNKC as specified by the ModelParams. */
void build_wPNKC(ModelParams const& p, RunVars& rv);

/* Sample spontaneous PN output from odor 0 (but sampling before odor onset, so doesn't
 * matter which). */
Column sample_PN_spont(ModelParams const& p, RunVars const& rv);

/* Decide a KC threshold column from KC membrane voltage data. */
Column choose_KC_thresh(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in);

/* Remove all columns <step in timecourse.*/
void remove_before(unsigned step, Matrix& timecourse);
/* Remove all pretime columns in all timecourses in r. */
void remove_all_pretime(ModelParams const& p, RunVars& r);

/* Get the list of odors that should be simulated (non-tuning). */
std::vector<unsigned> get_simlist(ModelParams const& p);

// TODO TODO try to adapt to something that takes a std::vector<Matrix> and writes as
// one .npy file (or at least a fn to handle those, even if across multiple files. we
// typically have a list of Matrices, one per odor) (or takes an Eigen tensor and does
// the same, but we don't actually use those here)
// TODO test this first?
//
// TODO use Matrix instead of Eigen::MatrixXd?
//
// adapting some solutions in https://github.com/rogersce/cnpy/issues/56
namespace np {
    using namespace Eigen;

    // NOTE: using template like this seems to only work within a namespace like this
    //template <typename Derived>
    //void to_npy(const MatrixBase<Derived> &mat, std::string fname) {
    // this didn't work. led to many errors about failed template argument
    //void to_npy(const Matrix<Derived> &mat, std::string fname) {
    // also didn't work.
    //void to_npy(const MatrixXd<Derived> &mat, std::string fname) {
    // also didn't work.
    //void to_npy(const Eigen::Matrix &mat, std::string fname) {
    void to_npy(const Eigen::MatrixXd &mat, std::string fname) {

        // TODO need this code to handle non-Matrix stuff (everything besides `else`
        // contents)? (well, currently works as commented)
        /*
        int rows = mat.rows();
        int cols = mat.cols();
        // Vector case
        if (cols == 1) {
            // copy to std::vector
            std::vector<typename Derived::Scalar> data(rows);
            for (int i = 0; i < rows; i++) {
                data[i] = mat(i);
            }
            // save
            cnpy::npy_save(fname, &data[0], {(size_t)rows}, "w");
            return;
        } else {

            // TODO can i delete? work even w/ this?
            Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic> tmat = mat.transpose();
            // TODO TODO need to swap rows/cols for one or both of calls below?
            Map<const Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic>> out(
                tmat.data(), tmat.rows(), tmat.cols()
                //mat.data(), mat.rows(), mat.cols()
            );
            // save to file
            // TODO restore?
            //cnpy::npy_save(fname, out.data(), {(size_t)mat.cols(), (size_t)mat.rows()},
            // TODO need?
            cnpy::npy_save(fname, out.data(), {(size_t)tmat.cols(), (size_t)tmat.rows()},
                "w"
            );

        }
        */

        // NOTE: putting below inside the else statement (w/ same `if` code as from orig
        // answer) did not fix below. did not try w/ transposing first yet...

        // TODO can i delete? work even w/ this? (well, does compile. and does without)
        //Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic> tmat = mat.transpose();

        // TODO TODO need to swap rows/cols for one (or both?) of calls below?
        // (assuming no transpose above. does that even help? one GH comment was saying
        // it alone wouldn't deal w/ row vs col -major order)
        // Need to reference Matrix as Eigen::Matrix, b/c of Matrix def in olfsysm.hpp.
        // Assuming input is ColMajor order (which I believe is default for Eigen).
        //Map<const Eigen::Matrix<typename Derived::Scalar, Dynamic, Dynamic, RowMajor>> out(
        // TODO is there something generic i can do other than Derived::Scalar? double
        // always ok? some way i can leave that part blank?
        Map<const Eigen::Matrix<double, Dynamic, Dynamic, RowMajor>> out(
            //tmat.data(), tmat.rows(), tmat.cols()
            // TODO TODO how to fix:
            //   libolfsysm/src/olfsysm.cpp: In instantiation of ‘void np::to_npy(const
            //   Eigen::MatrixBase<Derived>&, std::string) [with Derived =
            //   Eigen::Matrix<double, -1, -1>; std::string =
            //   std::__cxx11::basic_string<char>]’:
            //  libolfsysm/src/olfsysm.cpp:2618:49:   required from here
            //  libolfsysm/src/olfsysm.cpp:335:17: error: ‘const class
            //  Eigen::MatrixBase<Eigen::Matrix<double, -1, -1> >’ has no member named
            //  ‘data’ 335 |             mat.data(), mat.rows(), mat.cols()
            mat.data(), mat.rows(), mat.cols()
        );
        // TODO TODO TODO are there other cnpy calls i can use, to write the size
        // separately, then write the matrices one-by-one? i assume that's essentially
        // the storage order for 3-tensors?
        // TODO need?
        //cnpy::npy_save(fname, out.data(), {(size_t)tmat.cols(), (size_t)tmat.rows()},
        // TODO restore?
        cnpy::npy_save(fname, out.data(), {(size_t)mat.rows(), (size_t)mat.cols()},
            "w"
        );

    }
}

/*******************************************************************************
********************************************************************************
*********************                                      *********************
*********************            IMPLEMENTATIONS           *********************
*********************                                      *********************
********************************************************************************
*******************************************************************************/
inline unsigned get_ngloms(ModelParams const& mp) {
    return mp.orn.data.delta.rows();
}
inline unsigned get_nodors(ModelParams const& mp) {
    return mp.orn.data.delta.cols();
}

ModelParams::Time::Time() : stim(*this) {
}
ModelParams::Time::Time(Time const& o) : stim(*this) {
    pre_start  = o.pre_start;
    start      = o.start;
    end        = o.end;
    stim.start = o.stim.start;
    stim.end   = o.stim.end;
    dt         = o.dt;
}
ModelParams::Time::Stim::Stim(ModelParams::Time& o) : _owner(o) {
}
unsigned ModelParams::Time::Stim::start_step() const {
    return (start - _owner.pre_start)/_owner.dt;
}
unsigned ModelParams::Time::Stim::end_step() const {
    return (end - _owner.pre_start)/_owner.dt;
}
Row ModelParams::Time::Stim::row_all() const {
    Row ret(1, _owner.steps_all());
    ret.setZero();
    for (unsigned i = start_step(); i < end_step(); i++) {
        ret(i) = 1.0;
    }
    return ret;
}
unsigned ModelParams::Time::start_step() const {
    return (start-pre_start)/dt;
}
unsigned ModelParams::Time::steps_all() const {
    return (end-pre_start)/dt;
}
unsigned ModelParams::Time::steps() const {
    return (end-start)/dt;
}
Row ModelParams::Time::row_all() const {
    Row ret(1, steps_all());
    ret.setOnes();
    return ret;
}

RunVars::RunVars(ModelParams const& p) : orn(p), ln(p), pn(p), ffapl(p), kc(p) {
}
RunVars::ORN::ORN(ModelParams const& p) :
    sims(get_nodors(p), Matrix(get_ngloms(p), p.time.steps_all())) {
}
RunVars::LN::LN(ModelParams const& p) :
    inhA{std::vector<Vector>(get_nodors(p), Row(1, p.time.steps_all()))},
    inhB{std::vector<Vector>(get_nodors(p), Row(1, p.time.steps_all()))} {
}
RunVars::PN::PN(ModelParams const& p) :
    wAPLPN( p.pn.n_total_boutons ? p.pn.n_total_boutons : get_ngloms(p), 1 ),
    wPNAPL( 1, p.pn.n_total_boutons ? p.pn.n_total_boutons : get_ngloms(p) ),
    sims(get_nodors(p), Matrix(get_ngloms(p), p.time.steps_all())),
    bouton_sims(get_nodors(p), Matrix(p.pn.n_total_boutons ? p.pn.n_total_boutons : 0,
        p.time.steps_all())) {
}
RunVars::FFAPL::FFAPL(ModelParams const& p) :
    vm_sims(get_nodors(p), Row(1, p.time.steps_all())),
    coef_sims(get_nodors(p), Row(1, p.time.steps_all())) {
    for (auto& v : vm_sims) {
        v.setZero();
    }
}
RunVars::KC::KC(ModelParams const& p) :
    wPNKC(
        p.kc.wPNKC_one_row_per_claw
        // TODO is kc_ids.size() # claws? use a dedicated var to be more clear?
        // (already have nclaws_total? use that? or delete it?)
        ? int(p.kc.kc_ids.size())
        : int(p.kc.N),
        get_ngloms(p)
    ),

    // TODO try to use something other than p.kc.kc_ids.size()?
    // kc_ids.size() should be total # of claws
    wAPLKC(p.kc.wPNKC_one_row_per_claw ? int(p.kc.kc_ids.size()) : int(p.kc.N), 1),
    wKCAPL(1, p.kc.wPNKC_one_row_per_claw ? int(p.kc.kc_ids.size()) : int(p.kc.N)),

    wAPLKC_scale(1.0),
    wKCAPL_scale(1.0),

    // TODO want this to always be shape of thr (probably)? or always # KCs (not #
    // claws?) matter (outside of [pretty much unused] n_claws_active_to_spike > 0 code?
    // test that code with this too?)?
    pks(p.kc.n_claws_active_to_spike > 0 ? int(p.kc.kc_ids.size()) : p.kc.N,
        get_nodors(p)
    ),
    // NOTE: n_claws_active_to_spike > 0 should also imply wPNKC_one_row_per_claw=true
    // kc_ids.size() should be total # of claws
    thr(p.kc.n_claws_active_to_spike > 0 ? int(p.kc.kc_ids.size()) : p.kc.N, 1),
    responses(p.kc.N, get_nodors(p)),
    spike_counts(p.kc.N, get_nodors(p)),

    vm_sims(p.kc.save_vm_sims ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    spike_recordings(p.kc.save_spike_recordings ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    nves_sims(p.kc.save_nves_sims ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    inh_sims(p.kc.save_inh_sims ? get_nodors(p) : 0,
             Matrix(1, p.time.steps_all() )),
    // TODO TODO make sure Is_sims has each compartment separate (need commented code
    // below for that?)
    Is_sims(p.kc.save_Is_sims ? get_nodors(p) : 0,
            Matrix(1,p.time.steps_all() )),
    // TODO delete? or need this? (prob need unless reshaped later, for compartmented
    // APL case)
    // inh_sims(p.kc.save_inh_sims ? get_nodors(p) : 0,
    //          Matrix( (p.kc.apl_coup_const != -1) ? int(p.kc.comp_num) : 1,
    //                  p.time.steps_all() )),
    // Is_sims(p.kc.save_Is_sims ? get_nodors(p) : 0,
    //         Matrix( (p.kc.apl_coup_const != -1) ? int(p.kc.comp_num) : 1,
    //                 p.time.steps_all() )),

    // TODO TODO need this up here (in order for resize below to work? resize below even
    // working?) (maybe use 0 instead of p.kc.N, if resizing below does work w/ this [or
    // delete if it doesn't also require this line])
    // TODO TODO maybe temporarily just define new var for # claws, and use that here
    // (if i can't easily get some dynamic approach based on raw.size() below to work)
    // TODO default to some shape that effectively disables this if not in
    // one-row-per-claw case? or just assert that at runtime?
    // TODO rename to something w/ units? what are proper units (and do they make sense
    // as-is? does it really matter?)?
    claw_sims(p.kc.save_claw_sims ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),

    // TODO need? seems initialized in fit_sparsesness anyway.
    tuning_iters(0)

{
    if (p.kc.wPNKC_one_row_per_claw) {
        const auto& raw = p.kc.kc_ids;  // One body ID per claw
        claw_to_kc.resize(raw.size());

        // TODO just do at runtime? didn't seem to work up here (prob b/c need to resize
        // for each odor, not once on the outer vector of Matrix objects...?)?
        // TODO otherwise need to loop over claw_sims (std::vector of Matrix),
        // resizing reach Matrix to some 2d shape (currently doing inside sim_KC_layer)
        //claw_sims.resize(p.kc.save_claw_sims ? get_nodors(p) : 0, raw.size(),
        //    p.time.steps_all());

        std::unordered_map<unsigned, int> id2idx;
        // TODO better name (-> delete comment above check after loop)
        int nextIndex = 0;

        // TODO what is this doing?
        for (size_t i=0; i<raw.size(); ++i) {
            unsigned bid = raw[i];
            auto it = id2idx.find(bid);
            if (it == id2idx.end()) {
                id2idx[bid] = nextIndex;
                claw_to_kc(i) = nextIndex++;
            } else {
                claw_to_kc(i) = it->second;
            }
        }

        // Optional safety check: confirm number of unique KCs matches expected N
        check(nextIndex == int(p.kc.N));
    } else {
        claw_to_kc.resize(0);  // For clarity, make sure it's empty
    }
}

// TODO make matrix const? that limit anything outside of this fn?
bool all_nonneg(Matrix& mat) {
    // TODO this actually have any perfomance benefit over: (mat.array() >= 0).all()?
    // does Eigen evaluate >= 0 for all elements of matrix before all, or one-by-one
    // (terminating after first False)? benchmark, if not clear from docs?
    for (int i=0; i<mat.rows(); ++i) {
        for (int j=0; j<mat.cols(); ++j) {
            if (mat(i,j) < 0) {
                return false;
            }
        }
    }
    return true;
}

// TODO TODO do i need to support both std::vector<unsigned> and Eigen::VectorXi here,
// (or preferably just change claw_to_kc->vector (or Btn_to_pn to VectorXi))
//Eigen::VectorXd duplicate_vals_for_each_subunit_id(Eigen::VectorXd unit_vec, std::vector<unsigned>, )
//
// TODO vector and Vector have some consistent indexing syntax?
// TODO just want vector input/output always?
template<typename T>
Eigen::VectorXd duplicate_vals_for_each_subunit_id(Eigen::VectorXd unit_vec,
    T unit_id_for_each_subunit) {

    // TODO .size() available on both vector and Vector?
    const unsigned n_subunits = unit_id_for_each_subunit.size();
    check(n_subunits > unit_vec.size());

    // TODO assert anything on min/max of unit IDs? (start at 0, all < n_units?)
    // TODO assert none missing?

    check(!unit_vec.hasNaN());

    // TODO want to require N passed in, to initialize this? compute # of unique
    // unit_vec otherwise (optional N?)? assert N passed in (if available) matches #
    // unique values?
    Eigen::VectorXd subunit_vec = Eigen::VectorXd(n_subunits);
    subunit_vec.setConstant(NAN);
    for (unsigned i=0; i<n_subunits; i++) {
        // TODO try () or .at() instead?
        unsigned unit_idx = unit_id_for_each_subunit[i];

        subunit_vec[i] = unit_vec[unit_idx];
    }
    check(!subunit_vec.hasNaN());
    return subunit_vec;
}

// TODO maybe i want this to be refactored to share w/ a fn grouping over boutons
// too tho? even need a fn like that?
// TODO do we actually want to support input other than vectors? (could try
// template<typename T> if needed) don't think so tho
// TODO any reason i can't use const for rv here? (seems ok so far...)
Eigen::VectorXd sum_across_claws_within_each_kc(ModelParams const& p, RunVars const& rv,
    Eigen::VectorXd claw_vec) {
    // TODO check wPNKC_one_row_per_claw=true (should only be called then anyway...)?

    // TODO some way to avoid this needing to be computed each call (other than passing
    // in)? matter (prob not)?
    const unsigned n_claws = rv.kc.claw_to_kc.size();
    check(claw_vec.size() == n_claws);
    Eigen::VectorXd kc_vec = Eigen::VectorXd::Zero(p.kc.N);
    for (unsigned claw=0; claw<n_claws; ++claw) {
        unsigned kc = rv.kc.claw_to_kc[claw];
        // TODO what happens if claw_to_kc is set with signed values? should also be
        // checking it's >0 in python anyway... not sure claw_to_kc def in .hpp
        // currently enforces a certain type though (just Eigen::VectorXi currently)
        check(kc < p.kc.N);
        kc_vec[kc] += claw_vec(claw);
    }
    return kc_vec;
}

// TODO TODO add fn(s) for checking variables are either row/col vectors
// (use something in eigen for this already? or just use the appropriate eigen types to
// have compiler already recognize this [apparently there are options for row/col vec
// Matrices, so not just restricted to VectorXd]. would simplify things like computing
// dot products, or other fns expecting vector-like inputs)

// TODO also include checking of wPNKC (-> rename)? or another fn that does both?
void check_APL_weights(ModelParams const& p, RunVars& rv) {
    check(rv.kc.wAPLKC.cols() == 1);
    check(rv.kc.wKCAPL.rows() == 1);

    if (!p.kc.wPNKC_one_row_per_claw) {
        check(rv.kc.wAPLKC.rows() == p.kc.N);
        check(rv.kc.wKCAPL.cols() == p.kc.N);
    } else {
        // TODO use something else for num_claws? refactor to consistently use the same
        // thing everywhere?
        unsigned num_claws = rv.kc.claw_to_kc.size();
        check(rv.kc.wAPLKC.rows() == num_claws);
        check(rv.kc.wKCAPL.cols() == num_claws);
    }

    // TODO also check no +/- inf values? some builtin check for that, like np.isfinite?
    check(!rv.kc.wAPLKC.hasNaN());
    check(all_nonneg(rv.kc.wAPLKC));

    check(!rv.kc.wKCAPL.hasNaN());
    check(all_nonneg(rv.kc.wKCAPL));
}

void split_regular_csv(std::string const& str, std::vector<std::string>& vec) {
    int seg = 0;
    std::string growing;
    for (char ch : str) {
        if (ch == ',') {
            vec[seg++] = growing;
            growing = "";
        }
        else {
            growing += ch;
        }
    }
    vec[vec.size()-1] = growing;
}

/* Helper function for load_hc_data(). */
void load_hc_data_line(
        std::string const& line, std::vector<std::string>& segs,
        Matrix& out, unsigned col) {
    unsigned const N_HC_GLOMS  = 23;
    split_regular_csv(line, segs);
    unsigned g8fix = 0; // decrement the column ID of odors after glom 8
    for (unsigned glom = 0; glom < N_HC_GLOMS+1; glom++) {
        /* Ignore the 8th glom column (Kennedy does this). */
        if (glom == 7) {
            g8fix = 1;
            continue;
        }

        out(glom-g8fix, col) = std::stod(segs[glom+2]);
    }
}
void load_hc_data(ModelParams& p, std::string const& fpath) {
    unsigned const N_HC_ODORS  = 110; // all original HC odors
    unsigned const N_HC_GLOMS  = 23;  // all good HC gloms
    unsigned const N_ODORS_ALL = 186; // all odors in Kennedy's HC data file

    // TODO is it an issue if input data exceeds these sizes? or where else if resizing
    // happening? just setting directly via pybind11 work? add tests to check?
    p.orn.data.delta.resize(N_HC_GLOMS, N_HC_ODORS);
    p.orn.data.spont.resize(N_HC_GLOMS, 1);

    std::ifstream fin(fpath);
    std::string line;

    /* Discard the first two (header) lines. */
    std::getline(fin, line);
    std::getline(fin, line);

    /* Read the rest of the lines (except the last one). */
    std::vector<std::string> segs(N_HC_GLOMS+2+1); // 2 ID cols, 1 bad glom col
    for (unsigned odor = 0; odor < N_ODORS_ALL; odor++) {
        std::getline(fin, line);

        /* We need to read to the end of the file, but we aren't interested in
         * any of the the non-HC odors. */
        if (odor >= N_HC_ODORS) continue;

        /* Parse and store data. */
        load_hc_data_line(line, segs, p.orn.data.delta, odor);
    }

    /* Load the spontaneous rates line. */
    std::getline(fin, line);
    load_hc_data_line(line, segs, p.orn.data.spont, 0);

    /* Load connectivity distribution data. */
    p.kc.cxn_distrib.resize(1, N_HC_GLOMS);
    // TODO move cxn_distrib init to default params (out from load_hc_data at least)?
    // or move to separate fn to just init that (and only call that fn, not
    // load_hc_data, for cases in mb_model where we only need cxn_distrib)?
    // TODO check these values against values parsed from Caron 2013 supplementary table
    // 1? would need to OCR... (+ use that to define for more than the hallem glomeruli?
    // or at least save that as a CSV somewhere, for use in al_analysis.mb_model, if not
    // in here?)
    /* Data presumably taken from some real measurements.
     * Taken from Kennedy source. */
    p.kc.cxn_distrib <<
        2.0, 24.0, 4.0, 30.0, 33.0, 8.0, 0.0,
        // no #8!
        29.0, 6.0, 2.0, 4.0, 21.0, 18.0, 4.0,
        12.0, 21.0, 10.0, 27.0, 4.0, 26.0, 7.0,
        26.0, 24.0;
}

void smoothts_exp(Matrix& vin, double wsize) {
    double extarg = wsize;
    if (wsize > 1.0) {
        extarg = 2.0/(wsize+1.0);
    }
    for (int i = 1; i < vin.cols(); i++) {
        vin.col(i) = extarg*vin.col(i) + (1-extarg)*vin.col(i-1);
    }
}

void add_randomly(std::function<double()> rng, Matrix& out) {
    for (unsigned i = 0; i < out.rows(); i++) {
        for (unsigned j = 0; j < out.cols(); j++) {
            out(i, j) += rng();
        }
    }
}

double ffapl_coef_gini(ModelParams const& p,
        Column const& pn, Column const& spont) {
    Column src;
    if (p.ffapl.gini.source == "=")
        src = pn;
    else if (p.ffapl.gini.source == "-spont")
        src = pn-spont;
    else if (p.ffapl.gini.source == "/spont")
        src = pn.array()/spont.array();
    else if (p.ffapl.gini.source == "(-s)/s")
        src = (pn-spont).array()/spont.array();
    else
        return 1.0;

    double mu = src.mean();
    if (abs(mu) < 1e-5)
        return 1.0;

    int n = src.size();
    double g = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            g += abs(src(i)-src(j));
        }
    }

    double dn = double(n);
    g = 1.0 - p.ffapl.gini.a*g/(2.0*dn*dn*mu);
    if (g < 0.0)
        g = 0.0;

    return g;
}

double ffapl_coef_lts(ModelParams const& p,
        Column const& pn, Column const& spont) {
    Column delta = pn-spont;
    delta = (delta.array() < 0).select(0, delta);

    // TODO why need this special case? something blow up as difference gets small?
    if ((delta.array()/spont.array()).abs().maxCoeff() < 0.05) {
        // <5% change from spont in all channels
        return 1.0;
    }

    double L = pow(delta.sum(), 2.0)/(delta.array()*delta.array()).sum();
    L =  (1.0 - L/delta.size())/(1.0 - 1.0/delta.size());
    if (isnan(L)) {
        L = 1.0;
    }

    double m = p.ffapl.lts.m;
    return m + L*(1.0-m);
}

void build_wPNKC_from_cxnd(
        Matrix& w, unsigned nc, Row const& cxnd, double drop_prop) {
    w.setZero();
    std::vector<double> flat(cxnd.size());
    double sum = 0;
    for (unsigned i = 0; i < cxnd.size(); i++) {
        flat[i] = cxnd(0, i);
        sum += flat[i];
    }
    flat.push_back(drop_prop*sum/(1.0-drop_prop));
    std::discrete_distribution<int> dd(flat.begin(), flat.end());
    for (unsigned kc = 0; kc < w.rows(); kc++) {
        for (unsigned claw = 0; claw < nc; claw++) {
            int idx = dd(g_randgen);
            if (idx < cxnd.size()) {
                w(kc, idx) += 1.0;
            }
        }
    }
}

void build_wPNKC(ModelParams const& p, RunVars& rv) {
    if (p.kc.preset_wPNKC) {
        return;
    }
    if (p.kc.seed != 0) {
        g_randgen.seed(p.kc.seed);
        rv.log(cat("build_wPNKC: g_randgen seed=", p.kc.seed ));
    }
    unsigned nc = p.kc.nclaws;
    double pdp = p.kc.pn_drop_prop;
    if (p.kc.uniform_pns) {
        rv.log("building UNIFORM connectivity matrix");
        // TODO also log nc? other things? what's going on w/ seed?
        // TODO can i even repro old checks against matt's stuff from model_test.py?

        // TODO delete
        rv.log(cat("get_ngloms: ", get_ngloms(p) ));
        // should be N KCs
        rv.log(cat("wPNKC.rows(): ", rv.kc.wPNKC.rows() ));
        // should be same as get_ngloms(p)?
        rv.log(cat("wPNKC.cols(): ", rv.kc.wPNKC.cols() ));
        //
        Row cxnd(1, get_ngloms(p));
        cxnd.setOnes();
        // TODO modify to add logging / duplicate up here to inspect output of rng
        build_wPNKC_from_cxnd(rv.kc.wPNKC, nc, cxnd, pdp);
    }
    else {
        rv.log("building WEIGHTED connectivity matrix");
        build_wPNKC_from_cxnd(rv.kc.wPNKC, nc, p.kc.cxn_distrib, pdp);
    }
    // TODO what is this doing? currents size an issue?
    // .currents not referenced anywhere else anyway...
    if (p.kc.currents.size()) {
        // TODO delete (not actually running anyway)
        rv.log("multiplying wPNKC by kc.currents.asDiagonal()");
        //

        rv.kc.wPNKC *= p.kc.currents.asDiagonal();
        // TODO delete. did i add this line or was it from matt? i assume it's equiv to
        // above?
        //rv.kc.wPNKC = rv.kc.wPNKC.array().colwise() * p.kc.currents.array();
    }
}

// TODO assert actually >1 rows and 1 column for output?
Column sample_PN_spont(ModelParams const& p, RunVars const& rv) {
    /* Sample from halfway between time start and stim start to stim start. */
    unsigned sp_t1 =
        p.time.start_step()
        + unsigned((p.time.stim.start-p.time.start)/(2*p.time.dt));
    // TODO use p.time.stim.start_step() instead? should be same (was in first case i
    // checked, and don't think that would change)
    unsigned sp_t2 =
        p.time.start_step()
        + unsigned((p.time.stim.start-p.time.start)/(p.time.dt));
    int row_dim;
    // TODO TODO delete. it should always be # gloms, right? do i need separate code to
    // sample bouton spont now?
    /*
    if (p.pn.n_total_boutons) {
        // TODO TODO fix. need to use pn<>bouton ID maps?
        // (this used to be `get_ngloms(p) * p.pn.Btn_num_per_glom`, and should have
        // worked then (for test_btn_expansion)... (but that was just b/c pn_sims was a
        // diff shape then, i believe. # boutons, not # gloms)
        row_dim = p.pn.n_total_boutons;
    } else {
        row_dim = get_ngloms(p);
    }
    */
    row_dim = get_ngloms(p);

    // TODO delete eventually?
    //
    // TODO may have issues if not all odors are in sim_only? check! don't often test
    // that
    //
    // checking that it doesn't matter which odor index we use (loop over all odors
    // and check all against first?)
    for (unsigned i=0; i < rv.pn.sims.size(); i++) {
        if (i == 0) {
            continue;
        }
        // TODO delete
        /*
        rv.log(cat("p.pn.n_total_boutons:", p.pn.n_total_boutons));
        rv.log(cat("row_dim:", row_dim));
        rv.log(cat("rv.pn.sims[i].rows():", rv.pn.sims[i].rows()));
        rv.log(cat("rv.pn.sims[i].cols():", rv.pn.sims[i].cols()));
        */
        //
        // TODO why is this failing in test_btn_separate (first case) now?
        // do need the .array() to use the (a == b).all() equality testing
        // (b/c pn.sims still only had 54 rows, despite row_dim=162 (=# boutons))
        check((rv.pn.sims[i].block(0, sp_t1, row_dim, sp_t2-sp_t1).rowwise().mean().array() ==
               rv.pn.sims[0].block(0, sp_t1, row_dim, sp_t2-sp_t1).rowwise().mean().array()
              ).all() );
    }
    //

    Column pn_spont;
    // TODO will prob need to change implementation if we ever want to support variable
    // # of boutons per glomerulus?
    pn_spont = rv.pn.sims[0].block(0, sp_t1, row_dim, sp_t2-sp_t1).rowwise().mean();

    if (p.pn.n_total_boutons > 0) {
        // TODO also make this a Matrix of shape (n, 1), for consistency? (currently
        // will be a vector)
        pn_spont = duplicate_vals_for_each_subunit_id(pn_spont, rv.pn.Btn_to_pn);
        check(pn_spont.size() == p.pn.n_total_boutons);
    }
    return pn_spont;
}

// TODO what is appropriate output type actually? Row?
// TODO can i also specify shape of output type here? just assert within? or take ref to
// claw_drive, and set into that, rather than returning?
// TODO need to pass t? prob at least for cases w/ pn<>apl interactions? can i
// precompute across all time, and then apply apl after (prob not...)?
Eigen::VectorXd pn_to_kc_drive_at_t(ModelParams const& p, RunVars const& rv,
    Matrix const& pn_t, unsigned t, Matrix& bouton_sims) {

    // TODO TODO TODO do i actually need the baseline subtracting tianpei
    // was doing in order for the math to work out? do that then? if so,
    // refactor pn spont handling, so i can just copy that here?
    //
    // TODO TODO what happens if we initialize to one size, and then set with rvalue
    // that is another size? (or e.g. a matrix) err hopefully? or need to add
    // checks that size is as expected, even if predefining size?
    // NOTE: was previously only declared w/ this size explicitly in compartment code,
    // but i'm assuming it should work w/ other branches
    Eigen::VectorXd claw_drive(rv.kc.wPNKC.rows());
    if (p.pn.n_total_boutons) {
        // TODO is bouton_sims always defined here? what about if not preset_Btn (e.g.
        // for Btn_separate=True dict(use_connectome_APL=True) / dict() cases)? (should
        // be handled correctly already? don't think i ever test anything other than
        // n_total_boutons to init it...)
        //
        // TODO TODO still condense this down to one expression (across all time),
        // assuming no PN<>APL interactions?
        // TODO also support bouton_sims in wPNKC_one_row_per_claw=false
        // case?
        // TODO be clear (in comment) about what possible interpretation for
        // pn_t.rows() are (and pn_t.cols() are always # timesteps, right?)
        for (unsigned i=0; i<pn_t.rows(); i++) {
            std::vector<unsigned> btn_indices = rv.pn.pn_to_Btns[i];
            unsigned n_glom_boutons = btn_indices.size();
            // TODO duplicate in python check in advance?
            check(n_glom_boutons > 0);
            for (const unsigned& btn_idx : btn_indices) {
                // TODO check input is all 0 here? (should always be. maybe not worth
                // checking)
                //
                // TODO log shapes and indices here, at least on first
                // iteration, to check?
                //
                // TODO will i need to go odor by odor or no? (assuming not?
                // why does fit_sparseness end up needing that? just so it
                // can fit on only a subset of odors, if requested?)
                // TODO this assignment copies by default, right? need
                // .array() or anything?
                // TODO what is correct syntax for slicing each side here?
                // what are shapes of inputs?
                // ...am i going to (between this and `t`), slice RHS down
                // to a scalar?? want to do in some way that is not just
                // operating on scalars (prob don't care...)
                //
                // TODO TODO TODO need to divide by # boutons? only work (or
                // only equiv to non-separate-bouton code? care?) if doing
                // the baseline (pn spont) subtracting tianpei was doing?
                bouton_sims(btn_idx, t) = pn_t(i, t) / n_glom_boutons;
            }
        }
        // TODO what happens if RHS (of assignment) is ever not same shape as claw_drive
        // is defined w/ above? test?
        claw_drive = rv.kc.wPNKC * bouton_sims.col(t);

        // TODO TODO does `t` index actually start at 0 though (no)?
        // (start_step is 3000, at least for run_KC_sims?) matter (prob
        // not?)?
        //
        // TODO delete
        /*
        if (t == p.time.start_step() + 1) {
            // t: 3001
            // p.time.start_step(): 3000
            // wPNKC.rows(): 11043
            // wPNKC.cols(): 162
            // bouton_sims.rows(): 162
            // bouton_sims.cols(): 5500
            // bouton_sims.col(t).size(): 162
            // bouton_sims.col(t).rows(): 162
            // bouton_sims.col(t).cols(): 1
            // claw_drive.rows(): 11043
            // claw_drive.cols(): 1

            rv.log(cat("t: ", t ));
            rv.log(cat("p.time.start_step(): ", p.time.start_step() ));
            rv.log(cat("wPNKC.rows(): ", rv.kc.wPNKC.rows() ));
            rv.log(cat("wPNKC.cols(): ", rv.kc.wPNKC.cols() ));
            rv.log(cat("bouton_sims.rows(): ", bouton_sims.rows() ));
            rv.log(cat("bouton_sims.cols(): ", bouton_sims.cols() ));

            // TODO delete
            rv.log(cat("bouton_sims.col(t).size(): ",
                        bouton_sims.col(t).size()
            ));
            rv.log(cat("bouton_sims.col(t).rows(): ",
                        bouton_sims.col(t).rows()
            ));
            rv.log(cat("bouton_sims.col(t).cols(): ",
                        bouton_sims.col(t).cols()
            ));
            //

            // TODO compare to claw_drive shape in branch below? should be
            // same, right?
            rv.log(cat("claw_drive.rows(): ", claw_drive.rows() ));
            rv.log(cat("claw_drive.cols(): ", claw_drive.cols() ));
        }
        */
        //

    } else {
        // TODO what happens if RHS (of assignment) is ever not same shape as claw_drive
        // is defined w/ above? test?
        claw_drive = rv.kc.wPNKC * pn_t.col(t);
    }

    return claw_drive;
}

// TODO assert actually >1 rows and 1 column for output?
Column choose_KC_thresh_uniform(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in) {

    double thr_const;

    // TODO <= 0, for consistency?
    if (p.kc.n_claws_active_to_spike < 0) {
        unsigned tlist_sz = KCpks.cols();
        KCpks.resize(1, KCpks.size());                     // flatten
        std::sort(KCpks.data(), KCpks.data()+KCpks.size(),
                [](double a, double b){return a>b;});      // dec. order

        // TODO log what we would get if we used values +/- 1 from the index used for
        // KCpks? (to try to figure out limits of precision in sparsity achievable
        // through setting threshold alone)
        thr_const = KCpks(std::min(
            int(p.kc.sp_target * p.kc.sp_factor_pre_APL * double(p.kc.N*tlist_sz)),
            int(p.kc.N*tlist_sz)-1));

    // TODO delete. if i implement automatic threshold picking for this case, move into
    // fit_sparseness, or a separate fn called from there. if just going to keep
    // hardcoding, move that into python, and require fixed_thr == 'fixed' (rather than
    // allowing == 'uniform) in C++
    } else {
        //
        // final sparsity (w/ n_claws_active_to_spike=2): 0.217
        //thr_const = 20;
        //
        // final sparsity (w/ n_claws_active_to_spike=2): 0.1507
        //thr_const = 40;
        //
        // final sparsity (w/ n_claws_active_to_spike=2): 0.109 (within tolerance of
        // 0.1)
        // WORKS FOR =2
        //thr_const = 60;

        // WORKS FOR =3
        thr_const = 40;

        // TODO will i need (or will it be much easier) to implement something
        // iterative? want separate (more stringent) tolerance parameter for this step?
        // (can probably binary search between min and max claw spontaneous activities)
    }
    return thr_const + spont_in.array()*2.0;
}

// TODO assert actually >1 rows and 1 column for output?
Column choose_KC_thresh_homeostatic(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in) {
    /* Basically do the same procedure as the uniform algorithm, but do it for
     * each KC (row) separately instead of all together.
     * To sort each row in place, we first flatten the entire list, and then
     * sort portions of it in place. This is an unfortunate consequence of the
     * lack of stl iterators in Eigen <=3.4. */
    Column thr = 2.0*spont_in;
    unsigned cols = KCpks.cols();
    unsigned wanted = p.kc.sp_target * p.kc.sp_factor_pre_APL * double(cols);
    KCpks.transposeInPlace();
    KCpks.resize(1, KCpks.size());
    /* Choose a threshold for each KC by inspecting its sorted responses. */
    for (unsigned i = 0; i < p.kc.N; i++) {
        unsigned offset = i*cols;
        std::sort(KCpks.data()+offset, KCpks.data()+offset+cols,
                std::greater<double>());
        thr(i) += KCpks(offset+wanted);
    }
    return thr;
}

// TODO assert actually >1 rows and 1 column for output?
Column choose_KC_thresh_mixed(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in) {
    /* Just average uniform and homeostatic thresholding. */
    // choose_KC_thresh_X methods mess with KCpks, so we have to give them each
    // their own.
    Matrix& KCpks1 = KCpks;
    Matrix KCpks2 = KCpks;
    Column uniform = choose_KC_thresh_uniform(p, KCpks1, spont_in);
    Column hstatic = choose_KC_thresh_homeostatic(p, KCpks2, spont_in);
    return (uniform+hstatic)/2.0;
}

void sparsity_nonconvergence_failure(ModelParams const&p, RunVars const& rv) {
    rv.log(cat("did not get within tolerance of p.kc.sp_target within "
        "p.kc.max_iters=", p.kc.max_iters, " iterations! failure!"
    ));
    // will fail here if we did not converge
    check(rv.kc.tuning_iters <= p.kc.max_iters);
}

void scale_APLKC_weights(ModelParams const& p, RunVars& rv, double sp) {
    bool wAPLKC_scale_is_positive = false;
    double lr;
    double delta;
    double rel_sp_diff;

    // to check we don't add additional 0 entries later (which would not be able
    // to be updated on any future iterations).
    // code below will currently fail if count of either changes (don't want to
    // add 0s). no current implementation of backtracking or any other way to
    // rectify the situation without failing.
    int n_wAPLKC_eq0_initial = (rv.kc.wAPLKC.array() == 0.0).count();
    int n_wKCAPL_eq0_initial = (rv.kc.wKCAPL.array() == 0.0).count();

    // TODO will i still need this loop, assuming sp is not hardcoded above?
    // check!
    do {
        /* Modify the APL<->KC weights in order to move in the
         * direction of the target sparsity. */
        lr = p.kc.sp_lr_coeff / sqrt(double(rv.kc.tuning_iters));
        // do need to calculate this way, rather than using rel_sp_diff like:
        // delta = rel_sp_diff * lr;
        // ...just to preserve exact numerical behavior for previous outputs. probably
        // not otherwise important. could also compare against wAPLKC in old outputs w/
        // np.isclose/similar.
        delta = (sp - p.kc.sp_target) * lr / p.kc.sp_target;
        rel_sp_diff = (sp - p.kc.sp_target) / p.kc.sp_target;

        // TODO or should these be defined before the do-while loop? don't think so?
        // to rollback, if needed
        Column prev_wAPLKC = rv.kc.wAPLKC;
        Row prev_wKCAPL    = rv.kc.wKCAPL;
        double prev_wAPLKC_scale = 0;
        if (p.kc.preset_wAPLKC) {
            prev_wAPLKC_scale = rv.kc.wAPLKC_scale;
        } else {
            // this `else` def only used for logging. don't need for wKCAPL.
            prev_wAPLKC_scale = rv.kc.wAPLKC.mean();
        }
        double prev_wKCAPL_scale = 0;
        if (p.kc.preset_wKCAPL) {
            prev_wKCAPL_scale = rv.kc.wKCAPL_scale;
        }
        //

        if (!p.kc.preset_wAPLKC) {
            if (p.kc.wPNKC_one_row_per_claw) {
                // TODO refactor to share w/ elsewhere
                double change = delta;
                for (unsigned claw=0; claw<rv.kc.claw_to_kc.size(); ++claw) {
                    // TODO TODO is it something fucky in this loop that's causing
                    // convergence issues? (didn't change tho so idk...)
                    unsigned kc = rv.kc.claw_to_kc[claw];
                    const std::size_t cnt = rv.kc.kc_to_claws[kc].size();
                    const double val = change / static_cast<double>(cnt ? cnt : 1);
                    // TODO do i actually want the update to be different for
                    // different claws (in diff KCs)? necessary for some of the
                    // tests (and care enough to keep?)? change this path to
                    // somehow also use w[APLKC|KCAPL]_scale (maybe initializing
                    // the corresponding w[APLKC|KCAPL] vectors in here? maybe
                    // in python?)
                    rv.kc.wAPLKC(claw, 0) += val;
                }
            } else {
                // TODO why using .array() for +=, but not for direct assignment
                // operations? is .array() actually necessary in this case?
                // what does .array() do?
                rv.kc.wAPLKC.array() += delta;
            }
            //rv.log(cat("rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));

            // TODO need .array() here?
            if ((rv.kc.wAPLKC.array() > 0).all()) {
                wAPLKC_scale_is_positive = true;
            }
        } else {
            rv.kc.wAPLKC_scale += delta;

            // TODO delete?
            //rv.log(cat("rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

            if (rv.kc.wAPLKC_scale > 0) {
                wAPLKC_scale_is_positive = true;
            }
            rv.kc.wAPLKC = rv.kc.wAPLKC_scale * rv.kc.wAPLKC_unscaled;
        }

        // TODO TODO also log either mean or scalar wAPLKC (to get a sense of delta
        // relative to absolute there)
        // TODO TODO why do deltas not seem to be the differnce between wAPLKC_scale on
        // successive iterations? am i doing something stupid? or just misinterpreting
        // something?
        rv.log(cat("i=", rv.kc.tuning_iters, " sp=", sp, " target=", p.kc.sp_target,
            " rel_sp_diff=", rel_sp_diff, " acc=", p.kc.sp_acc,
            " wAPLKC_scale=", prev_wAPLKC_scale, " delta(wAPLKC)=", delta, " lr=", lr
        ));

        if (!wAPLKC_scale_is_positive) {
            rv.kc.tuning_iters++;
            // TODO TODO figure out why calculation above isn't/wasn't updating w/
            // increase in tuning_iters (should change lr, right?)
            if (rv.kc.tuning_iters > p.kc.max_iters) {
                // will exit with err
                sparsity_nonconvergence_failure(p, rv);
            }
            rv.log(cat("incrementing tuning_iters=", rv.kc.tuning_iters,
                " to get lower step size (to keep all scaled wAPLKC values "
                "positive)"
            ));
            // rollback to previous values before picking new (smaller) step size
            rv.kc.wAPLKC = prev_wAPLKC;
            if (p.kc.preset_wAPLKC) {
                rv.kc.wAPLKC_scale = prev_wAPLKC_scale;
            }
            rv.kc.wKCAPL = prev_wKCAPL;
            if (p.kc.preset_wKCAPL) {
                rv.kc.wKCAPL_scale = prev_wKCAPL_scale;
            }
        }
    } while (!wAPLKC_scale_is_positive);

    // NOTE: assuming that is wAPLKC scale is positive, we won't have
    // similar issues w/ wKCAPL scale below. may not be true?

    // TODO TODO TODO integrate something like below into loop above (not just checking
    // whether scale is positive, but also checking whether we introduced any 0 values
    // after subtracting delta?) (and also rolling back steps in these circumstances)
    if (!p.kc.preset_wKCAPL) {
        if (p.kc.wPNKC_one_row_per_claw) {
            // TODO refactor
            double change = delta / double(p.kc.N);
            for (unsigned claw=0; claw<rv.kc.claw_to_kc.size(); ++claw) {
                unsigned kc = rv.kc.claw_to_kc[claw];
                const std::size_t cnt = rv.kc.kc_to_claws[kc].size();
                const double val = change / static_cast<double>(cnt ? cnt : 1);
                rv.kc.wKCAPL(0, claw) += val;
            }
        } else {
            rv.kc.wKCAPL.array() += delta / double(p.kc.N);
        }
        rv.log(cat("rv.kc.wKCAPL mean: ", rv.kc.wKCAPL.mean()));
    } else {
        rv.kc.wKCAPL_scale += delta / double(p.kc.N);

        // TODO delete?
        rv.log(cat("rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));

        check(rv.kc.wKCAPL_scale > 0.0);
        rv.kc.wKCAPL = rv.kc.wKCAPL_scale * rv.kc.wKCAPL_unscaled;
    }

    // probably want to abort (so we can change tuning params and re-run)
    // rather than clip values (which would break overall shape of vector(s)
    // from connectome). or otherwise take steps to avoid this state (would
    // probably be better if we didn't have to abort).
    // TODO TODO come up with strategy to backoff delta if we add any 0
    // values in w[APLKC|KCAPL] or get w[APLKC|KCAPL]_scale <= 0
    // TODO in meantime, give people a message to choose different step size
    // params (that can be seen before assertion causes failure. [also?]
    // print directly to stderr? assert failure already do that with enough
    // info?)
    // TODO check None of Tianpei's code paths trip any of the assertion on
    // APL weight values <= 0 below
    //
    /* If we learn too fast in the negative direction we could end
     * up with negative weights. */
    if (delta < 0.0) {
        if (!p.kc.preset_wAPLKC) {
            int n_wAPLKC_lt0 = (rv.kc.wAPLKC.array() < 0.0).count();
            check(n_wAPLKC_lt0 == 0);

            int n_wAPLKC_eq0 = (rv.kc.wAPLKC.array() == 0.0).count();
            check(n_wAPLKC_eq0_initial == n_wAPLKC_eq0);

            // TODO delete? should currently always do nothing, given
            // assertions above.
            rv.kc.wAPLKC = (rv.kc.wAPLKC.array() < 0.0).select(
                    0.0, rv.kc.wAPLKC);
        }

        if (!p.kc.preset_wKCAPL) {
            int n_wKCAPL_lt0 = (rv.kc.wKCAPL.array() < 0.0).count();
            check(n_wKCAPL_lt0 == 0);

            int n_wKCAPL_eq0 = (rv.kc.wKCAPL.array() == 0.0).count();
            check(n_wKCAPL_eq0_initial == n_wKCAPL_eq0);

            // TODO delete? should currently always do nothing, given
            // assertions above.
            rv.kc.wKCAPL = (rv.kc.wKCAPL.array() < 0.0).select(
                    0.0, rv.kc.wKCAPL);
        }
    }

    // TODO delete (or do similar print for all cases, not just
    // preset_wAPLKC=true)
    //
    // for debugging + trying to support scaling of arbitrary positive
    // vector wAPLKC/wKCAPL inputs
    if (p.kc.preset_wAPLKC) {
        double wAPLKC_mean = rv.kc.wAPLKC.mean();
        // TODO if keeping, try to combine w/  previous .log call above?
        rv.log(cat("wAPLKC_mean: ", wAPLKC_mean));
    }
    if (p.kc.preset_wKCAPL) {
        double wKCAPL_mean = rv.kc.wKCAPL.mean();
        // TODO if keeping, try to combine w/  previous .log call above?
        rv.log(cat("wKCAPL_mean: ", wKCAPL_mean));
    }
    //
}

void fit_sparseness(ModelParams const& p, RunVars& rv) {
    rv.log("fitting sparseness");

    std::vector<unsigned> tlist = p.kc.tune_from;
    if (!tlist.size()) {
        for (unsigned i = 0; i < get_nodors(p); i++) tlist.push_back(i);
    }

    Column temp_wAPLKC;
    Row temp_wKCAPL;
    // TODO only define these if `zero_wAPLKC=true`? only used then, right?
    // TODO also get initial values for other cases, to also support zero_wAPLKC there?
    // (and then eventually should be able to default to it, with no change in behavior,
    // but not high priority)
    if (p.kc.preset_wAPLKC) {
        temp_wAPLKC = rv.kc.wAPLKC;
        temp_wKCAPL = rv.kc.wKCAPL;
    }

    /* Calculate spontaneous input to KCs. */
    Column pn_spont = sample_PN_spont(p, rv);
    // TODO TODO TODO start here and see where differences first emerge in
    // test_btn_expansion test (need to separately baseline PNs like tianpei had? does
    // that even make sense biologically?)

    check(pn_spont.cols() == 1);
    check(pn_spont.rows() > 1);
    check(!pn_spont.hasNaN());

    // TODO why was this multiplication not throwing some error? (when wPNKC had #
    // boutons for cols, and pn_spont was a vector (col/row?) of length # gloms
    // (gave output that was a vector of length # KCs still, i believe? just not all
    // initialized correctly?)
    Column spont_in_ini = rv.kc.wPNKC * pn_spont;
    // TODO delete
    rv.log("");
    rv.log("at spont_in_ini def:");
    rv.log(cat("rv.kc.wPNKC.rows(): ", rv.kc.wPNKC.rows()));
    rv.log(cat("rv.kc.wPNKC.cols(): ", rv.kc.wPNKC.cols()));
    rv.log(cat("rv.kc.wPNKC.hasNaN(): ", rv.kc.wPNKC.hasNaN()));
    rv.log(cat("rv.kc.wPNKC.array().isNaN().count(): ",
                rv.kc.wPNKC.array().isNaN().count()
    ));
    rv.log(cat("rv.kc.wPNKC.minCoeff(): ", rv.kc.wPNKC.minCoeff()));
    rv.log(cat("rv.kc.wPNKC.maxCoeff(): ", rv.kc.wPNKC.maxCoeff()));
    rv.log("");
    rv.log(cat("pn_spont.rows(): ", pn_spont.rows()));
    rv.log(cat("pn_spont.cols(): ", pn_spont.cols()));
    rv.log(cat("pn_spont.hasNaN(): ", pn_spont.hasNaN()));
    rv.log(cat("pn_spont.array().isNaN().count(): ",
                pn_spont.array().isNaN().count()
    ));
    rv.log(cat("pn_spont.minCoeff(): ", pn_spont.minCoeff()));
    rv.log(cat("pn_spont.maxCoeff(): ", pn_spont.maxCoeff()));
    rv.log("");
    rv.log(cat("spont_in_ini.rows(): ", spont_in_ini.rows()));
    rv.log(cat("spont_in_ini.cols(): ", spont_in_ini.cols()));
    rv.log(cat("spont_in_ini.hasNaN(): ", spont_in_ini.hasNaN()));
    rv.log(cat("spont_in_ini.array().isNaN().count(): ",
                spont_in_ini.array().isNaN().count()
    ));
    rv.log(cat("spont_in_ini.minCoeff(): ", spont_in_ini.minCoeff()));
    rv.log(cat("spont_in_ini.maxCoeff(): ", spont_in_ini.maxCoeff()));
    //
    Column spont_in;
    // if `n_claws_active_to_spike > 0`, spont_in should be of length equal to # claws
    // (as spont_in_ini should be here)
    if (p.kc.wPNKC_one_row_per_claw && p.kc.n_claws_active_to_spike <= 0) {
        // TODO delete
        rv.log("setting spont_in w/ sum_across...");
        //
        // TODO TODO or do i actually want/need separate spont input for each bouton
        // (for some ways of running that code? for test_btn_expansion to pass?)
        // TODO TODO TODO is this actually doing what i want?
        spont_in = sum_across_claws_within_each_kc(p, rv, spont_in_ini);
    } else {
        // TODO delete
        rv.log("setting spont_in to spont_in_ini...");
        //
        spont_in = spont_in_ini;
        if (p.kc.n_claws_active_to_spike > 0) {
            check(spont_in.size() == rv.kc.nclaws_total);
        }
    }
    // TODO TODO sanity check spont_in here? initialize w/ some sentinel (NAN?) and
    // check we have no more of that? seems set incorrectly now
    if (p.kc.n_claws_active_to_spike <= 0) {
        check(spont_in.size() == p.kc.N);
        // TODO factor into generic column checking (/ change of types)?
        check(spont_in.cols() == 1);
    }
    // TODO delete
    rv.log("");
    rv.log("at spont_in def:");
    rv.log(cat("spont_in.hasNaN(): ", spont_in.hasNaN()));
    rv.log(cat("spont_in.array().isNaN().count(): ",
                spont_in.array().isNaN().count()
    ));
    rv.log(cat("spont_in.minCoeff(): ", spont_in.minCoeff()));
    rv.log(cat("spont_in.maxCoeff(): ", spont_in.maxCoeff()));
    //
    rv.kc.spont_in = spont_in;

    if (p.kc.preset_wAPLKC) {
        rv.log(cat("INITIAL rv.kc.wAPLKC.mean(): ", rv.kc.wAPLKC.mean()));
        rv.log(cat("INITIAL rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));
        // should be a deep copy
        rv.kc.wAPLKC_unscaled = rv.kc.wAPLKC;
    }
    if (p.kc.preset_wKCAPL) {
        rv.log(cat("INITIAL rv.kc.wKCAPL.mean(): ", rv.kc.wKCAPL.mean()));
        rv.log(cat("INITIAL rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));
        rv.kc.wKCAPL_unscaled = rv.kc.wKCAPL;
    }

    // TODO are preset_wAPLKC and wPNKC_one_row_per_claw mutually exclusive?
    // adapt wPNKC_one_row_per_claw code to initialize w[APLKC|KC] similar
    // to how preset_w[APLKC|KCAPL]=true code does, and then use scale
    // *_scale params at output?

    /* Set starting values for the things we'll tune. */
    // TODO matter? seems to be overwritten below in this case anyway...
    // (and put inside this conditional to avoid overwriting values set in python, via
    // pybind11)
    if (p.kc.tune_apl_weights) {
        // We prob want to set zero for either cases so that the threshold can be set properly
        if (!p.kc.preset_wAPLKC) {
            rv.kc.wAPLKC.setZero();
        }
        if (!p.kc.preset_wKCAPL) {
            if (p.kc.wPNKC_one_row_per_claw) {
                // TODO TODO refactor to share w/ other code doing similar
                double preset_wKCAPL_base = 1.0/float(p.kc.N);
                for (unsigned i_c=0; i_c<rv.kc.claw_to_kc.size(); ++i_c) {
                    unsigned kc = rv.kc.claw_to_kc[i_c];
                    // # claws for this KC
                    const std::size_t cnt = rv.kc.kc_to_claws[kc].size();
                    // TODO what is right part of this line doing?
                    const double val = preset_wKCAPL_base / static_cast<double>(cnt ? cnt : 1);
                    // TODO change indexing to remove the 0?
                    rv.kc.wKCAPL(0, i_c) = val;
                }
            } else {
                rv.kc.wKCAPL.setConstant(1.0/float(p.kc.N));
            }
        }

        // TODO just always unconditionally do this?
        if (p.kc.zero_wAPLKC) {
            rv.kc.wAPLKC.setZero();
            rv.kc.wKCAPL.setZero();
        }
    }

    check_APL_weights(p, rv);

    if (!p.kc.use_vector_thr) {
        if (!p.kc.use_fixed_thr) {
            rv.log("setting threshold higher than should ever be "
                "reached, to disable KC spiking for threshold-setting phase"
            );
            // Higher than will ever be reached. I believe this is so the KCs don't have
            // their Vm reset at arbitrary points when we are trying to pick the
            // thresholds. Also has consequence of disabling APL, at least for main path
            // where it depends on KC spiking (will want to manually [temporarily] zero
            // APL weights to achieve same effect for code path where APL activity does
            // not depend on KC spiking)
            rv.kc.thr.setConstant(1e5);
        }
        else {
            rv.log(cat("using FIXED threshold: ", p.kc.fixed_thr));
            // TODO would it ever make sense to have add_fixed_thr_to_spont=False?
            // when? in any cases i use? doc (only seem to set it False in mb_model.py
            // when using homeostatic thresholds, and i could move that logic in here)
            if (p.kc.add_fixed_thr_to_spont) {
                // TODO TODO why again the *2? doc!
                // TODO delete + replace w/ similar commented line below
                // (after confirming the 2 things w/ factor 2 cancel out...)
                rv.log("adding fixed threshold to 2 * spontaneous PN input");
                // TODO what are units of spont_in? doc these as units of fixed_thr
                rv.kc.thr = p.kc.fixed_thr + spont_in.array()*2.0;
            } else {
                rv.kc.thr.setConstant(p.kc.fixed_thr);
            }
        }
    } else {
        rv.log("using prespecified vector thresholds");
        // TODO even want to allow `add_fixed_thr_to_spont = False`? don't think it's
        // useful now
        if (p.kc.add_fixed_thr_to_spont) {
            rv.log("adding threshold to 2 * spontaneous PN input");

            // TODO delete
            rv.log(cat("(before adding spont) rv.kc.thr.mean(): ", rv.kc.thr.mean()));

            // do i need .array() here? yes, need LHS .array() to avoid err, at least w/
            // RHS as it is here
            // TODO i assuming changing <x>.array() also changes values in <x> (assuming
            // it's a Matrix/similar)? confirm
            rv.kc.thr.array() += spont_in.array()*2.0;

            // TODO delete
            rv.log(cat("(after adding spont) rv.kc.thr.mean(): ", rv.kc.thr.mean()));
        }
    }

    // TODO or should i use std::numeric_limits<double>::quiet_NaN() from C++ <limits>?
    // this is C NaN from <math.h>
    // TODO why does this not seem to be doing anything? log right after?
    // (or at least, by the time we call `...pks = KCpks`, no NaN remain. not surprising
    // really...)
    rv.kc.pks.setConstant(NAN);

    /* Used for measuring KC voltage; defined here to make it shared across all
     * threads.*/
    // TODO TODO use tlist.size() instead of get_nodors for pks too? or get_nodors for
    // both? (somewhat recently added size def of pks in initializer up top, w/
    // get_nodors. could prob remove, and set size of that here)
    Matrix KCpks(p.kc.N, tlist.size());
    KCpks.setZero();

    /* Used to store odor response data during APL tuning. */
    Matrix KCmean_st(p.kc.N, 1+ ((tlist.size() - 1) / p.kc.apltune_subsample));

    // TODO delete if don't keep
    //Matrix over_thr(p.kc.N, tlist.size());
    //
    // TODO work (seems to)? need to specify size (don't think so)? maybe init w/ NaN or
    // 0 to check what values are actually written in tests below?
    // TODO go back to specifying size for these? any reason to (/ not to?)?
    Matrix thr_diff;
    Matrix over_thr2;
    //

    bool sparsity_not_converged = true;
    bool under_max_iters = true;

    /* Used to store the current sparsity.
     * Initially set to the below value because, given default model
     * parameters, it causes tuning to complete in just one iteration. */
    double sp;
    if (p.kc.hardcode_initial_sp) {
        // NOTE: currently need to keep this initial value as-is, in order to reproduce
        // (at least) the hemibrain paper responses exactly (tests to reproduce those,
        // and some other outputs, currently set hardcode_initial_sp=True).
        sp = 0.0789;
        rv.log(cat("pretending initial sparsity is ", sp, ", rather than calculating "
            "it, because p.kc.hardcode_initial_sp=true. may be necessary to reproduce"
            " certain outputs exactly."
        ));
    }

    /* Used to count number of times looped; the 'learning rate' is decreased
     * as 1/sqrt(count) with each iteration. */
    rv.kc.tuning_iters = 0;

    // TODO delete these and just always check against strs? why have these?
    unsigned const TTFIXED = 1;
    unsigned const TTHSTATIC = 2;
    unsigned const TTMIXED = 3;
    unsigned const TTUNIFORM = 4;
    unsigned const TTINVALID = 5;
    std::string tt = p.kc.thr_type;
    bool nott = (tt == "");
    unsigned thrtype =
        nott ?
            p.kc.use_fixed_thr ? TTFIXED :
            p.kc.use_homeostatic_thrs ? TTHSTATIC :
            TTUNIFORM
        :   tt == "uniform" ? TTUNIFORM :
            tt == "hstatic" ? TTHSTATIC :
            tt == "mixed" ? TTMIXED :
            tt == "fixed" ? TTFIXED :
        // TODO what is purpose of abort + nullptr here? explain?
        // raise an error (that can be converted by pybind11) instead?
        (abort(), TTINVALID);

    /* Break up into threads. */
#pragma omp parallel
    {
        // TODO is it necessary to define these matrices after the OMP parallel
        // statement? why? what would be different if they were defined above? is this
        // not potentially allocating memory per thread (just perhaps not b/c of eigen's
        // features)?
        /* Output matrices for the KC simulation. */
        Matrix Vm(p.kc.N, p.time.steps_all());
        Matrix spikes(p.kc.N, p.time.steps_all());
        Matrix nves(p.kc.N, p.time.steps_all());
        Matrix inh(1, p.time.steps_all());
        Matrix Is(1, p.time.steps_all());
        // NOTE: rv.kc.nclaws_total currently initialized at start of run_KC_sims, right
        // before fit_sparseness call
        // TODO TODO is this actually correct? do i not care to also index into
        // rv.kc.claw_sims vector<Matrix>, as code below does for rv.pn.sims/similar?
        // matter here, so long as we get it right for run_KC_sims?
        // TODO TODO TODO at least check it doesnt matter?
        Matrix claw_sims(rv.kc.nclaws_total, p.time.steps_all());

        Matrix bouton_sims(p.pn.n_total_boutons, p.time.steps_all());

        // TODO assuming i want this for use_vector_thr. why don't i want to do below
        // for TTFIXED?
        if (thrtype != TTFIXED && !p.kc.use_vector_thr) {
#pragma omp single
            {
                // TODO print str value for thrtype instead? (may need to add something
                // to invert mapping above. seems like some cases above currently don't
                // use the existing string p.kc.thr_type [but that could be changed?])
                rv.log(cat("choosing thresholds from spontaneous input (thrtype=",
                           thrtype, ")"));
            }
            if (p.kc.zero_wAPLKC) {
                check(rv.kc.wAPLKC.isZero());
                check(rv.kc.wKCAPL.isZero());
            }

            // TODO maybe i still want to sim_KC_layer in use_vector_thr case
            // (just not use it to pick a thr)?

            /* Measure voltages achieved by the KCs, and choose a threshold
             * based on that. */
#pragma omp for
            for (unsigned i = 0; i < tlist.size(); i++) {
                sim_KC_layer(p, rv, rv.pn.sims[tlist[i]], rv.ffapl.vm_sims[tlist[i]],
                    Vm, spikes, nves, inh, Is, claw_sims, bouton_sims
                );
                // TODO TODO (probably at end of sim_KC_layer), check all quantities are
                // fully initialized? maybe init w/ NaN or something (instead of 0,
                // where easily possible), to make that check easier?
                // TODO TODO need high threshold for all calls above to make sure
                // KCs don't get reset and screw up threshold calculation? should i do
                // one more set of sim_KC_layer calls w/ correct threshold value after,
                // or an easier way to calculate sparsity w/o re-running?
                check(inh.isZero());
                check(Is.isZero());
#pragma omp critical
                // TODO TODO why subtracting 2x spont_in first? just cancel out what
                // similar offset in threshold choosing fns? doc (+ make sure per-claw
                // threshold case handled correctly too)
                KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*2.0;
            }

#pragma omp single
            {
                rv.kc.pks = KCpks;

                /* Finish picking thresholds. */
                rv.kc.thr =
                    (thrtype == TTHSTATIC ? choose_KC_thresh_homeostatic :
                     thrtype == TTMIXED ? choose_KC_thresh_mixed :
                     choose_KC_thresh_uniform)
                    (p, KCpks, spont_in);

                // TODO compute + log sparsity here? (from KCpks)
                // TODO + save into new rv variable, for use in al_analysis?
                // (even worth? i assume that w/ reasonable pre-conditions, we can
                // always get pretty bang-on here?)
            }
        }

        // TODO if i move the stuff in this `#pragma omp single` block up enough, can i
        // avoid need to switch back to single threaded? (without it here,
        // `use_connectome_APL_weights=True` sensitivity analysis check repro-ing output
        // w/ fixed wAPLKC/wKCAPL is failing, b/c crazy high values on output
        // wAPLKC/etc)
#pragma omp single
        {
            if (p.kc.preset_wAPLKC) {
                if (p.kc.tune_apl_weights) {
                    // is it gauranteed that subsequent code will not be run by other
                    // threads, until after this single block completes (don't want
                    // other threads using uninitialized data)? or how to guarantee?
                    // yes, other threads wait unless a nowait clause is added:
                    // https://www.openmp.org/spec-html/5.0/openmpsu38.html
                    //
                    // TODO should i not also be zeroing and then resetting wKCAPL here
                    // too? why did matt not do that above (make sense that APL will
                    // have no influence w/ even this 0, but inh/Is still 0 above [when
                    // i'm not expecting, when wKCAPL is non-zero])
                    rv.kc.wAPLKC = rv.kc.wAPLKC_unscaled;
                } else {
                    // TODO delete
                    rv.log(cat("FIXED rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

                    rv.kc.wAPLKC = rv.kc.wAPLKC_scale * rv.kc.wAPLKC_unscaled;
                }
            }
            if (!p.kc.tune_apl_weights && p.kc.preset_wKCAPL) {
                // TODO delete
                rv.log(cat("FIXED rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));

                rv.kc.wKCAPL = rv.kc.wKCAPL_scale * rv.kc.wKCAPL_unscaled;
            }
        }

        // TODO if `!tune_apl_weights` just return here, so i can de-ident code below?
        // or does some or it need to run? (we are still inside a parallel block tho.
        // would have to at least synchronize or something, no? or could we break up
        // large current parallel block w/ two smaller ones, one above and one below?)
        /* Enter this region only if APL use is enabled; if disabled, just exit
         * (at this point APL->KC weights are set to 0). */
        if (p.kc.tune_apl_weights) {
#pragma omp single
        {
            // TODO delete
            rv.log("");
            //
            rv.log(cat("tuning APL<->KC weights; tuning begin (",
                        "target=", p.kc.sp_target,
                        " acc=", p.kc.sp_acc,
                        " thr=", rv.kc.thr.mean(),
                        ")"));

            rv.kc.tuning_iters = 1;
            // TODO maybe require/assume input preset vectors will be normalized or
            // scaled in a certain way? or compute appropriate w[APLKC|KCAPL]_scale
            // constants to have mean (after multiplying by preset vectors) equal to
            // what we would have been starting with before (maybe to average value of 1
            // [this is what al_analysis is currently doing], so we can set *_scale
            // factors to same as wAPLKC/wKCAPL being set below)?
            /* Starting values for to-be-tuned APL<->KC weights. */

            // TODO (after also defiing temp_* vars above in !preset_wAPLKC case) also
            // support resetting those cases here
            if (p.kc.preset_wAPLKC && p.kc.zero_wAPLKC) {
                rv.kc.wAPLKC = temp_wAPLKC;
                rv.kc.wKCAPL = temp_wKCAPL;
                rv.log(cat("after thresh initialization wAPLKC: ", rv.kc.wAPLKC.mean()));
            }

            if (!p.kc.preset_wAPLKC) {
                if (p.kc.wPNKC_one_row_per_claw) {
                    // TODO refactor to share w/ duplicated code?
                    const double base = 2.0 * ceil(-log(p.kc.sp_target));
                    for (unsigned claw=0; claw<rv.kc.claw_to_kc.size(); ++claw) {
                        unsigned kc = rv.kc.claw_to_kc[claw];
                        const std::size_t cnt = rv.kc.kc_to_claws[kc].size();
                        const double val = base / static_cast<double>(cnt ? cnt : 1);
                        rv.kc.wAPLKC(claw, 0) = val;
                    }
                } else {
                    rv.kc.wAPLKC.setConstant(2*ceil(-log(p.kc.sp_target)));
                }
                rv.log(cat("setConst initial rv.kc.wAPLKC sum: ", rv.kc.wAPLKC.sum()));
                rv.log(cat("setConst initial rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));
            } else {
                rv.kc.wAPLKC_scale = 2*ceil(-log(p.kc.sp_target));
                // TODO delete
                rv.log(cat("INITIAL rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

                rv.kc.wAPLKC = rv.kc.wAPLKC_scale * rv.kc.wAPLKC_unscaled;
            }
            if (!p.kc.preset_wKCAPL) {
                if (p.kc.wPNKC_one_row_per_claw) {
                    const double base = 2*ceil(-log(p.kc.sp_target)) / double(p.kc.N);
                    for (unsigned claw=0; claw<rv.kc.claw_to_kc.size(); ++claw) {
                        unsigned kc = rv.kc.claw_to_kc[claw];
                        // TODO refactor
                        const std::size_t cnt = rv.kc.kc_to_claws[kc].size(); // claws of this KC
                        const double val = base / static_cast<double>(cnt ? cnt : 1);
                        rv.kc.wKCAPL(0, claw) = val;  // row vector
                    }
                } else {
                    rv.kc.wKCAPL.setConstant(2*ceil(-log(p.kc.sp_target)) / double(p.kc.N));
                }
                rv.log(cat("setConst initial rv.kc.wKCAPL sum: ", rv.kc.wKCAPL.sum()));
                rv.log(cat("setConst initial rv.kc.wKCAPL mean: ", rv.kc.wKCAPL.mean()));
            } else {
                rv.kc.wKCAPL_scale = 2*ceil(-log(p.kc.sp_target)) / double(p.kc.N);
                rv.log(cat("INITIAL rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));
                rv.kc.wKCAPL = rv.kc.wKCAPL_scale * rv.kc.wKCAPL_unscaled;
            }

            // TODO keep this check? only do this one, and remove earlier check?
            // do again (/only?) at end of fit_sparseness (prob not needed...)?
            check_APL_weights(p, rv);
        }

        /* Continue tuning until we reach the desired sparsity. */
        do {
#pragma omp barrier

#pragma omp single
            {
                /*
                // TODO actually need this in loop below? shouldn't...
                if (! p.kc.hardcode_initial_sp) {
                    // TODO TODO TODO instead, define initial sp from rv.kc.pks
                    //
                    // rv.kc.pks.rows(): 1828
                    // rv.kc.pks.cols(): 17
                    // rv.kc.thr.rows(): 1828
                    // rv.kc.thr.cols(): 1
                    rv.log("");
                    rv.log("when trying to recalc sp:");
                    // TODO TODO TODO why do both of these have NaN? are they all NaN?
                    rv.log(cat("rv.kc.pks.rows(): ", rv.kc.pks.rows()));
                    rv.log(cat("rv.kc.pks.cols(): ", rv.kc.pks.cols()));
                    rv.log(cat("rv.kc.pks.hasNaN(): ", rv.kc.pks.hasNaN()));
                    rv.log(cat("rv.kc.pks.array().isNaN().count(): ",
                                rv.kc.pks.array().isNaN().count()
                    ));
                    rv.log(cat("rv.kc.pks.minCoeff(): ", rv.kc.pks.minCoeff()));
                    rv.log(cat("rv.kc.pks.maxCoeff(): ", rv.kc.pks.maxCoeff()));
                    rv.log("");
                    rv.log(cat("rv.kc.thr.rows(): ", rv.kc.thr.rows()));
                    rv.log(cat("rv.kc.thr.cols(): ", rv.kc.thr.cols()));
                    rv.log(cat("rv.kc.thr.hasNaN(): ", rv.kc.thr.hasNaN()));
                    rv.log(cat("rv.kc.thr.array().isNaN().count(): ",
                                rv.kc.thr.array().isNaN().count()
                    ));
                    rv.log(cat("rv.kc.thr.minCoeff(): ", rv.kc.thr.minCoeff()));
                    rv.log(cat("rv.kc.thr.maxCoeff(): ", rv.kc.thr.maxCoeff()));
                    rv.log("");
                    // TODO this work? omit .array() or only do in second step?
                    // didn't work
                    //auto const to_thr = rv.kc.pks.array() + 2*spont_in;
                    //
                    // compiles, but output (over_thr) is 1828x1 (not x17 i want)
                    //auto const to_thr = rv.kc.pks + 2*spont_in;
                    // TODO cast to something other than bool, to get sum/mean to work
                    // w/o compilation warnings? or can i compute sparsity in some way
                    // that won't trigger those warnings? any other way to count # of
                    // true values?
                    // TODO cast to unsigned instead? or that make .mean() harder?
                    // .sum() still work there?
                    // TODO get version using to_thr (w/ spont_in added) to
                    // work. currently getting 1828x1 output. want 1828x17 output
                    // like i had before.
                    //
                    // worked, but might need to offset one or the other by spont_in
                    // (one way or another, couldn't get sparsity calculation to seem to
                    // work. not sure if that's just b/c offset or something more
                    // fundamental)
                    //auto const over_thr = (rv.kc.pks.array() > rv.kc.thr.array()).cast<double>();
                    //auto const offset_thr = rv.kc.thr;
                    //auto const over_thr = (rv.kc.pks.array() > offset_thr.array()).cast<double>();
                    //over_thr = (rv.kc.pks.array() > rv.kc.thr.array()).cast<double>();
                    // TODO are these not both double? why getting mixed dtype
                    // compilation error now? (when assigning into prespecified Matrix,
                    // at least)
                    //over_thr = rv.kc.pks.array() > rv.kc.thr.array();
                    // auto const does actually remove a compilation error from above
                    // (but now back to needing to cast output of this comparison, in
                    // order to not have sum op warning)
                    auto const over_thr = (rv.kc.pks.array() > rv.kc.thr.array()).cast<double>();
                    //auto const over_thr = (rv.kc.pks.array() > rv.kc.thr.array()).cast<double>();
                    // TODO delete
                    // doesn't compile
                    //over_thr = rv.kc.pks.array() > rv.kc.thr.array();
                    //
                    // doesn't compile
                    //over_thr = (rv.kc.pks.rowwise() - rv.kc.thr);
                    // why does this also not compile? (b/c thr was also a matrix, even
                    // though it just has 1 column)
                    //over_thr = (rv.kc.pks.colwise() - rv.kc.thr);
                    //
                    // produces something of shape (1828, 1) instead of (1828, 17)
                    //over_thr = (rv.kc.pks - rv.kc.thr);
                    // can direct inequality work w/ columnwise too? no.
                    // > operator doesn't work w/ colwise()
                    //over_thr = rv.kc.pks.colwise() > rv.kc.thr.col(0);
                    check(rv.kc.thr.cols() == 1);
                    // TODO adapt
                    //auto const over_thr = Vm.col(t).array() > rv.kc.thr.array();
                    //
                    // TODO most idiomatic way to do this? or at least what works?
                    //
                    // this works, but would then need to threshold wrt 0 (or something
                    // else?)
                    // TODO delete eventually (if assertion that matches array() direct
                    // matrix vs vector inequality output above passes in all tests)
                    //thr_diff = (rv.kc.pks.colwise() - rv.kc.thr.col(0));
                    // i thought this worked at one point, but now getting:
                    // no match for operator> w/ Matrix and int (oh, the x.isApprox(y)
                    // error below just shadowed this error)
                    // (w/o the .array() on thr_diff. also doesn't work w/ array)
                    //over_thr2 = thr_diff.array() > 0;
                    // compilation error about mixing numeric types again
                    //over_thr2 = thr_diff.array() > 0;
                    // no such compilation error using auto const instead of assigning
                    // into matrix...
                    //thr_diff = (rv.kc.pks.colwise() - rv.kc.thr.col(0));
                    //auto const over_thr2 = (thr_diff.array() > 0).cast<double>();

                    // TODO do i actually need the .array() calls?
                    //auto const offset_thr = rv.kc.thr.array() - 2*spont_in.array();
                    auto const offset_thr = rv.kc.thr - 2*spont_in;
                    //thr_diff = (rv.kc.pks.colwise() - offset_thr.col(0));
                    //thr_diff = (rv.kc.pks.colwise() - offset_thr.matrix());
                    // works (compiles), w/ offset_thr def above w/ the 2 array() calls
                    //thr_diff = (rv.kc.pks.colwise() - offset_thr.matrix().col(0));
                    check(offset_thr.cols() == 1);
                    thr_diff = (rv.kc.pks.colwise() - offset_thr.col(0));
                    auto const over_thr2 = (thr_diff.array() > 0).cast<double>();

                    // https://stackoverflow.com/questions/44738315
                    // couldn't get to work. trying to figure out scalar type of array,
                    // to avoid compilation errors.
                    //rv.log(cat("decltype(over_thr)::Scalar: ", decltype(over_thr)::Scalar));
                    //rv.log(cat("decltype(over_thr2)::Scalar: ", decltype(over_thr2)::Scalar));

                    //check(over_thr.isApprox(over_thr2));
                    //

                    // TODO don't need some offset ([2x?] spont_in?) do i?

                    // TODO delete
                    rv.log("");
                    rv.log(cat("thr_diff.rows(): ", thr_diff.rows()));
                    rv.log(cat("thr_diff.cols(): ", thr_diff.cols()));
                    rv.log(cat("thr_diff.mean(): ", thr_diff.mean()));
                    rv.log(cat("thr_diff.maxCoeff(): ", thr_diff.maxCoeff()));
                    rv.log(cat("thr_diff.minCoeff(): ", thr_diff.minCoeff()));

                    // TODO delete
                    rv.log("");
                    rv.log(cat("over_thr.rows(): ", over_thr.rows()));
                    rv.log(cat("over_thr.cols(): ", over_thr.cols()));
                    // TODO TODO is this causing a segfault (just w/ `auto const` or
                    // also if size of matrix in def?)?
                    // (seems any usage of over_thr might be?)
                    //rv.log(cat("over_thr.colwise().sum(): ", over_thr.colwise().sum()));
                    //rv.log(cat("over_thr.mean(): ", over_thr.mean()));
                    //rv.log(cat("over_thr.maxCoeff(): ", over_thr.maxCoeff()));
                    //rv.log(cat("over_thr.minCoeff(): ", over_thr.minCoeff()));
                    rv.log("");
                    rv.log(cat("over_thr2.rows(): ", over_thr2.rows()));
                    rv.log(cat("over_thr2.cols(): ", over_thr2.cols()));
                    rv.log(cat("over_thr2.colwise().sum(): ", over_thr2.colwise().sum()));
                    rv.log(cat("over_thr2.mean(): ", over_thr2.mean()));
                    rv.log(cat("over_thr2.maxCoeff(): ", over_thr2.maxCoeff()));
                    rv.log(cat("over_thr2.minCoeff(): ", over_thr2.minCoeff()));
                    //
                    // TODO cast before mean to avoid warning about scalar sum op being
                    // undefined on bool? or just sum and divide by .size(), to be more
                    // explicit than casting to double? (sum also triggers the warning)
                    //sp = over_thr.mean();
                    sp = over_thr2.mean();
                    // TODO delete
                    rv.log("");
                    rv.log(cat("sp: ", sp));
                    //
                    check(sp > 0);
                    check(sp < 1);
                    // TODO TODO assert sparsity is very close to initial target (or at
                    // least above 0 and < 1?)?

                    // TODO delete
                    // TODO first assert spikes is not all 0 here (it will be,
                    // unless i do another round of sim_KC_layer calls [/similar], after
                    // setting threshold to chosen value above [down from initial
                    // value chosen to prevent spiking])? is above
                    // call set such that it is? can i fix it (disable APL some other
                    // way?)
                    //KCmean_st = spikes.rowwise().sum();
                    //
                    //KCmean_st = (KCmean_st.array() > 0.0).select(1.0, KCmean_st);
                    //sp = KCmean_st.mean();
                    // TODO assert this isn't 0 (is w/ current calculation from
                    // KCmean_st!)
                    //
                    rv.log(cat("calculated initial sparsity (response rate): ", sp));
                }
                */

                // always want this step to be at this point in the loop for the
                // hardcode_initial_sp=true case, so we don't double up at the end of
                // first pass.
                // TODO TODO or do we just want all the updates befores sim_KC_layer in
                // this case?
                if (p.kc.hardcode_initial_sp) {
                    // TODO log the special casing we are doing here
                    // TODO TODO any issue w/ generally doing these after first set of
                    // sim_KC_layer calls that have APL enabled (how else will we know
                    // what direction we should go in?)
                    scale_APLKC_weights(p, rv, sp);
                    rv.kc.tuning_iters++;
                }
            }

            /* Run through a bunch of odors to test sparsity. */
#pragma omp for
            for (unsigned i = 0; i < tlist.size(); i+=p.kc.apltune_subsample) {
                sim_KC_layer(p, rv, rv.pn.sims[tlist[i]], rv.ffapl.vm_sims[tlist[i]],
                    Vm, spikes, nves, inh, Is, claw_sims, bouton_sims
                );
                // TODO assert this is not true here? was it only because scale
                // factor was negative here? expect above, but not here, right?
                // (may still be some odors with no response at some point though...)
                if (inh.isZero()) {
                    rv.log(cat(
                        "odor ", i, " had all 0 inh (presumably no response either)!"
                    ));
                }
                //
                KCmean_st.col(i / p.kc.apltune_subsample) = spikes.rowwise().sum();
            }

#pragma omp single
            {
                KCmean_st = (KCmean_st.array() > 0.0).select(1.0, KCmean_st);
                sp = KCmean_st.mean();

                // TODO also log this relative diff, if actually any sign there is a
                // computation mismatch between here and python (is there tho?)
                // (already doing in scale_APLKC_weights?)
                sparsity_not_converged = (
                    abs(sp - p.kc.sp_target) > (p.kc.sp_acc * p.kc.sp_target)
                );
                under_max_iters = rv.kc.tuning_iters <= p.kc.max_iters;

                // TODO re-organize loop to not go back to always only calling
                // scale_APLKC_weights after first call? still think that might require
                // another set of sim_KC_layer calls though, to evaluate effect w/
                // initial APL weights... probably not worth it
                //
                // also don't want to duplicate calls in the hardcode=true case (so
                // always doing above sim_KC_layer calls there)
                if (sparsity_not_converged && !p.kc.hardcode_initial_sp) {
                    // TODO TODO why is sparsity not changing between first two calls of
                    // this (apparently)? (assume i was testing in
                    // non-hardcode-initial-sparsity case? prob in one of
                    // test_btn_expansion connectome APL cases?)
                    scale_APLKC_weights(p, rv, sp);
                    rv.kc.tuning_iters++;
                }
            }
        } while (sparsity_not_converged && under_max_iters);
#pragma omp barrier
#pragma omp single
        {
            // TODO what is point of this? does loop above give us 1 above the
            // tuning_iters we want? even if only the initial tuning, and no subsequent
            // loop iterations? why not move after parallel block (put in conditional on
            // tune_apl_weights=true) instead of in this separate single-thread block at
            // end?
            rv.kc.tuning_iters--;
        }
    }}

    // TODO delete?
    rv.log(cat("FINAL rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));
    rv.log(cat("FINAL rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));
    rv.log(cat("FINAL rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));
    rv.log(cat("FINAL rv.kc.wKCAPL mean: ", rv.kc.wKCAPL.mean()));

    if (!under_max_iters) {
        // this is just a hack to ensure we also failure in call below, consistent w/
        // call above (only need b/c the x-- above that i'm not currently sure if i can
        // remove)
        rv.kc.tuning_iters++;

        sparsity_nonconvergence_failure(p, rv);
    }

    // TODO always log tuned parameters at end (fixed_thr, wAPLKC/wKCAPL when not
    // preset, or wAPLKC_scale/wKCAPL_scale when preset)
    rv.log("done fitting sparseness");
    // TODO delete
    rv.log("");
}

void sim_ORN_layer(ModelParams const& p, RunVars const& rv, int odorid, Matrix& orn_t) {
    /* Initialize with spontaneous activity. */
    orn_t = p.orn.data.spont*p.time.row_all();
    // TODO TODO can orn_t go negative? i think i'm seeing that in some outputs???
    // is that reasonable?
    // TODO inspect some timecourses where it goes negative?
    // TODO TODO see >=0 constraining line in sim_PN_layer (?)

    /* "Odor input to ORNs" (Kennedy comment)
     * Smoothed timeseries of spont...odor rate...spont */
    Matrix   odor = orn_t + p.orn.data.delta.col(odorid)*p.time.stim.row_all();
    smoothts_exp(odor, 0.02/p.time.dt); // where does 0.02 come from!?

    double mul = p.time.dt/p.orn.taum;
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        orn_t.col(t) = orn_t.col(t-1)*(1.0-mul) + odor.col(t)*mul;
    }
}

void sim_LN_layer(ModelParams const& p, Matrix const& orn_t, Row& inhA, Row& inhB) {
    Row potential(1, p.time.steps_all()); potential.setConstant(300.0);
    Row response(1, p.time.steps_all());  response.setOnes();
    inhA.setConstant(50.0);
    inhB.setConstant(50.0);
    double inh_LN = 0.0;

    double dinhAdt, dinhBdt, dLNdt;
    double scaling = double(get_ngloms(p))/double(p.orn.n_physical_gloms);
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        dinhAdt = -inhA(t-1) + response(t-1);
        dinhBdt = -inhB(t-1) + response(t-1);
        dLNdt =
            -potential(t-1)
            +pow(orn_t.col(t-1).mean()*scaling, 3.0)/scaling/2.0*inh_LN;
        inhA(t) = inhA(t-1) + dinhAdt*p.time.dt/p.ln.tauGA;
        inhB(t) = inhB(t-1) + dinhBdt*p.time.dt/p.ln.tauGB;
        inh_LN = p.ln.inhsc/(p.ln.inhadd+inhA(t));
        potential(t) = potential(t-1) + dLNdt*p.time.dt/p.ln.taum;
        // TODO delete?
        //response(t) = potential(t) > lnp.thr ? potential(t)-lnp.thr : 0.0;
        response(t) = (potential(t)-p.ln.thr)*double(potential(t)>p.ln.thr);
    }
}
void sim_PN_layer(
    ModelParams const& p, RunVars& rv,
    Matrix const& orn_t, Row const& inhA, Row const& inhB,
    Matrix& pn_t) {
    // TODO: Verify this isn't actually making noise (both params 0? or sd at least?)?
    // It should be seed-able if it is.
    std::normal_distribution<double> noise(p.pn.noise.mean, p.pn.noise.sd);

    Column spont  = p.orn.data.spont*p.pn.inhsc/(p.orn.data.spont.sum()+p.pn.inhadd);
    pn_t          = p.orn.data.spont*p.time.row_all();
    double inh_PN = 0.0;

    Column orn_delta;
    Column dPNdt;
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        orn_delta = orn_t.col(t-1)-p.orn.data.spont;

        dPNdt = -pn_t.col(t-1) + spont;
        dPNdt +=
            200.0*((orn_delta.array()+p.pn.offset)*p.pn.tanhsc/200.0*inh_PN).matrix(
                ).unaryExpr<double(*)(double)>(&tanh);

        add_randomly([&noise](){return noise(g_randgen);}, dPNdt);

        inh_PN = p.pn.inhsc/(p.pn.inhadd+0.25*inhA(t)+0.75*inhB(t));
        pn_t.col(t) = pn_t.col(t-1) + dPNdt*p.time.dt/p.pn.taum;

        // TODO TODO why not do something like this in sim_ORN_layer case too?
        // ann also handle the 2 cases the same way?
        pn_t.col(t) = (0.0 < pn_t.col(t).array()).select(pn_t.col(t), 0.0);
    }

    // TODO delete. replace w/ old code [now above] (don't want to keep any changes
    // after 2cda53a80df [one of my last commits before Tianpei], right?)
    /*
    const int ng = get_ngloms(p);
    pn_t.setZero();
    // Baseline spontaneous activity, scaled for inhibition
    // TODO at least rename to pn_spont. store in RunVars if i end up wanting later?
    // or just calc later (i.e. in run_KC_sims)
    // TODO refactor to share w/ other pn activity calc? (should be same, right?)
    Column spont = p.orn.data.spont * p.pn.inhsc / (p.orn.data.spont.sum() + p.pn.inhadd);

    // Time-series vectors for PN<->APL interaction
    Matrix inh = Matrix::Zero(1, p.time.steps_all());
    Matrix Is = Matrix::Zero(1, p.time.steps_all());

    // TODO prob break into two separate (or only have one bool flag to enable both)
    if (!p.pn.preset_wAPLPN || !p.pn.preset_wPNAPL) {
        rv.pn.wAPLPN.setZero();
        rv.pn.wPNAPL.setZero();
    }

    Column spont_btn;       // Spontaneous activity per bouton
    Column orn_delta_btn;   // ORN delta activity per bouton
    Column dPNdt;
    // TODO delete? think i only want matrices of shape # btns in sim_KC_layer
    //if (p.pn.preset_Btn) {
    //    // total bouton count
    //    // TODO TODO TODO is this really not used anywhere else? mistake? just not
    //    // implemented yet? change that (still want to use this var?)
    //    int total_btns = rv.pn.Btn_to_pn.size();
    //    pn_t.resize(total_btns, p.time.steps_all());
    //    spont_btn.resize(total_btns, 1);

    //    // ADDED: Resize loop variables now that we know their dimension.
    //    orn_delta_btn.resize(total_btns, 1);
    //    dPNdt.resize(total_btns, 1);

    //    // fill each bouton row with its per-bouton baseline across time
    //    for (int g = 0; g < ng; ++g) {
    //        // std::vector<unsigned> of bouton indices for glom g
    //        const auto& btns = rv.pn.pn_to_Btns[g];
    //        const int Bg = (int)btns.size();
    //        // STEP 1: Calculate initialization value from RAW data.
    //        // const double per_btn_init = (Bg > 0) ? p.orn.data.spont(g) / double(Bg) : 0.0;
    //        const double per_btn_init = (Bg > 0) ? p.orn.data.spont(g): 0.0;
    //        // STEP 2: Calculate simulation baseline from the SCALED 'spont' variable.
    //        // const double per_btn_baseline = (Bg > 0) ? spont(g) / double(Bg) : 0.0;
    //        const double per_btn_baseline = (Bg > 0) ? spont(g) : 0.0;

    //        for (int k = 0; k < Bg; ++k) {
    //            const int b = btns[k];                  // bouton row index in pn_t
    //            spont_btn(b) = per_btn_baseline;
    //            // Fill the full row (1×T): scalar * time row
    //            pn_t.row(b) = per_btn_init * p.time.row_all();
    //        }
    //    }
    //} else {
    //    pn_t = p.orn.data.spont*p.time.row_all();

    //    // ADDED: Resize dPNdt for the non-bouton case.
    //    // orn_delta_btn is not used in this case, so it can remain size 0.
    //    dPNdt.resize(ng, 1);
    //}
    // TODO replace w/ old code?
    pn_t = p.orn.data.spont*p.time.row_all();
    dPNdt.resize(ng, 1);
    //

    double inh_PN = 0.0;
    Column orn_delta;

    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        // --- 4a. Select correct inputs (bouton or glomerulus) ---

        const Column* spont_ref; // Pointer to use either spont or spont_btn
        Column orn_delta;        // Will hold either per-glom or per-bouton delta

        Column orn_delta_glom = orn_t.col(t-1) - p.orn.data.spont; // ng x 1

        // TODO TODO delete this here, and only copy glomeruli -> btns at start of KC
        // simulation (where then inh can be applied as needed). does it matter
        // actually?
        //if (p.pn.preset_Btn) {
        //    spont_ref = &spont_btn;
        //    // Expand glomerulus delta to bouton delta
        //    for (int g = 0; g < ng; ++g) {
        //        const auto& btns = rv.pn.pn_to_Btns[g];
        //        // const int Bg = static_cast<int>(btns.size());
        //        // const double per_btn_delta = (Bg > 0) ? orn_delta_glom(g) / double(Bg) : 0.0;
        //        const double per_btn_delta = orn_delta_glom(g);
        //        for (int bouton_idx : btns) {
        //            orn_delta_btn(bouton_idx) = per_btn_delta;
        //        }
        //    }
        //    orn_delta = orn_delta_btn;
        //} else {
        //    spont_ref = &spont;
        //    orn_delta = orn_delta_glom;
        //}
        //spont_ref = &spont;
        // TODO replace w/ old code?
        // TODO TODO delete?
        orn_delta = orn_delta_glom;
        //

        // TODO is pn_apl_tune best name for this?
        // TODO TODO ever make sense to have this controlled separately from
        // pn_claw_to_APL=true? (is this even related to that?) use that flag (after renaming?) to control
        // this as well?
        //if (p.pn.pn_apl_tune) {
        //    // TODO TODO TODO when i'm actually implementing PN<>APL interactions (in
        //    // sim_KC_layer, do i want it to depend only on activity above baseline, or
        //    // also just raw PN activity? (may need to recalc or return/store spont_ref
        //    // if want it)
        //    // and (related) have i checked what KC outputs look like w/ const thresh,
        //    // instead of only a thresh relative to spontanenous input (do so, if
        //    // haven't)?
        //    //
        //    // PN activity above baseline (rectified)
        //    // TODO use cwiseMax(0.0) in place of select calls that are used elsewhere
        //    // for same purpose?
        //    Column pn_act = (pn_t.col(t-1) - *spont_ref).cwiseMax(0.0);

        //    // Drive from PNs to the inhibitory APL neuron
        //    double pn_apl_drive = (rv.pn.wPNAPL * pn_act).value();

        //    // TODO TODO TODO does this part make sense here? at least refactor to share
        //    // w/ APL activity code in sim_KC_layer?

        //    // Update inhibition dynamics
        //    double dIsdt = -Is(t-1) + pn_apl_drive;
        //    double dinhdt = -inh(t-1) + Is(t-1);
        //    Is(t) = Is(t-1) + dIsdt * p.time.dt / p.pn.apl_taum;
        //    inh(t) = inh(t-1) + dinhdt * p.time.dt / p.pn.tau_apl2pn;

        //    const double inh_scalar = inh(t-1);

        //    // TODO TODO TODO shouldn't PN activity here depend on APL activity more
        //    // broadly (including KC input), rather than just some similar looking "APL"
        //    // activity defined only using PN activity in here? rewrite?
        //    dPNdt = -pn_t.col(t-1) + *spont_ref + rv.pn.wAPLPN * inh_scalar;
        //} else {
        //    dPNdt = -pn_t.col(t-1) + *spont_ref; // APL effect removed
        //}

        // TODO what is unaryExpr... for?
        dPNdt += 200.0 * ((orn_delta.array() + p.pn.offset) * p.pn.tanhsc / 200.0 * inh_PN)
                           .matrix().unaryExpr<double(*)(double)>(&tanh);

        inh_PN = p.pn.inhsc / (p.pn.inhadd + 0.25 * inhA(t) + 0.75 * inhB(t));
        pn_t.col(t) = pn_t.col(t-1) + dPNdt * p.time.dt / p.pn.taum;
        pn_t.col(t) = (pn_t.col(t).array() < 0.0).select(0.0, pn_t.col(t));
    }
    */
}

// NOTE: this seems to only have ever been called from run_FFAPL_sims, which was only
// ever called manually (between run_PN_sims and run_KC_sims calls), and thus is not
// suitable for including feedback from overall APL activity (e.g. that includes input
// from KCs)
void sim_FFAPL_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t,
        Vector& ffapl_t, Vector& coef_t) {
    ffapl_t.setZero();
    coef_t.setZero();

    // TODO equiv to current calc? delete
    //Column pn_spont = p.orn.data.spont*p.pn.inhsc/(p.orn.data.spont.sum()+p.pn.inhadd);
    Column pn_spont = sample_PN_spont(p, rv);

    double (*coef_calc)(ModelParams const&, Column const&, Column const&);
    coef_calc =
        p.ffapl.coef == "gini" ? ffapl_coef_gini :
        p.ffapl.coef == "lts" ? ffapl_coef_lts :
        (abort(), nullptr);

    double dVdt;
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        coef_t(t) = coef_calc(p, pn_t.col(t-1), pn_spont);
        dVdt = -ffapl_t(t-1) + p.ffapl.w*coef_t(t)*pn_t.col(t-1).sum();
        ffapl_t(t) = ffapl_t(t-1) + dVdt*p.time.dt/p.ffapl.taum;
    }

    double spont = ffapl_t(p.time.stim.start_step()-1);
    if (p.ffapl.nneg) {
        ffapl_t = (spont < ffapl_t.array()).select(ffapl_t, spont);
    }
    if (p.ffapl.zero) {
        ffapl_t = ffapl_t.array() - spont;
    }
}

void sim_KC_layer(
    ModelParams const& p, RunVars const& rv,
    Matrix const& pn_t, Vector const& ffapl_t,
    Matrix& Vm, Matrix& spikes, Matrix& nves, Matrix& inh, Matrix& Is,
    Matrix& claw_sims, Matrix& bouton_sims) {
    /* Args:
     * - inh: APL potential timeseries
     * - Is: KC->APL synapse current (across all KCs) timeseries
     * */

    // Determine number of compartments
    // int n_compartments = rv.kc.claw_compartments.maxCoeff() + 1;
    Vm.setZero();
    spikes.setZero();
    // TODO why setOnes here?
    nves.setOnes();
    inh.setZero();
    Is.setZero();

    // TODO just set this automatically (to wPNKC.cols()) if wPNKC.cols() >
    // get_ngloms(p)?
    if (p.pn.n_total_boutons) {
        // TODO TODO necessary? or size defined correctly above? try to remove?
        //bouton_sims.resize(p.pn.n_total_boutons, p.time.steps_all());

        // TODO or init by copying from pn_sims? how to expand by correct # of pns
        // in each case? start using those ID maps here?
        // TODO TODO or can we not really copy PN sims, apart from perhaps some
        // current(?) (/last?) value? all bouton values in future will depend on
        // activity of all other KCs/boutons, including feedback from APL, right?
        // TODO TODO TODO should some parameter exist for like a bouton time constant
        // (or moving average window?), something with which to balance effect of APL vs
        // PN feedforward activity
        bouton_sims.setZero();

        // TODO (delete) need to move this bit into the loop over time though i assume?
        /*
        // TODO (delete. should be fine) or do in run_KC_sims, where we initially loop
        // over odors? (any other way to avoid duplicating some of that logic in here?)
        // (claw_sims currently ignores that issue in two calls in fit_sparseness, and
        // then also has claw_sims indexed by odor in run_KC_sims [where there is
        // actually potential to save the sims])
        for (unsigned i=0; i < rv.pn.sims.size(); i++) {
            std::vector<unsigned> btn_indices = rv.pn.pn_to_Btns[i];
            for (const unsigned& btn_idx : btn_indices) {
                // TODO will i need to go odor by odor or no? (assuming not? why does
                // fit_sparseness end up needing that? just so it can fit on only a
                // subset of odors, if requested?)
                // TODO this assignment copies by default, right? need .array() or
                // anything?
                bouton_sims[btn_idx] = pn_t[i];
            }
        }
        */
    }

    float use_ffapl = float(!p.kc.ignore_ffapl);
    if (p.kc.wPNKC_one_row_per_claw) {
        const unsigned n_claws = rv.kc.claw_to_kc.size();
        // TODO log n_claws

        // TODO try to move outside of sim_KC_layer (at least the resize()?)?
        // (still need? delete?)
        // TODO this able to correct for initial size not being specified
        // correctly? (yes, seems so. may be slower doing it in here tho)
        claw_sims.resize(n_claws, p.time.steps_all());

        claw_sims.setZero();

        if (p.kc.apl_coup_const != -1) {
            const auto& claws_by_compartment = rv.kc.compartment_to_claws;
            const unsigned num_comp = claws_by_compartment.size();
            double g_voltage_coup = p.kc.apl_coup_const;

            Eigen::VectorXd Is_prev_per_comp = Eigen::VectorXd::Zero(num_comp);
            Eigen::VectorXd Is_curr_per_comp = Is_prev_per_comp;

            // APL has its own membrane potential, which replaces 'inh' for feedback
            Eigen::VectorXd Vm_apl_prev_per_comp = Eigen::VectorXd::Zero(num_comp);
            Eigen::VectorXd Vm_apl_curr_per_comp = Vm_apl_prev_per_comp;

            // 'inh' is preserved for other uses but not for feedback in this loop
            Eigen::VectorXd inh_prev_per_comp  = Eigen::VectorXd::Zero(num_comp);
            Eigen::VectorXd inh_curr_per_comp  = inh_prev_per_comp;

            // Scalars evolve in lock-step as sums of vectors
            double inh_prev = inh_prev_per_comp.sum();
            double Is_prev  = Is_prev_per_comp.sum();
            double inh_curr = inh_prev;
            double Is_curr  = Is_prev;

            // TODO TODO shouldn't rest of values be filled in? am i missing
            // something? (they are below, right? but is this code doing anything?
            // delete?)
            inh(0,0) = inh_prev;   // log scalar aggregates (global)
            Is(0,0)  = Is_prev;

            // (rest of setup is the same)
            Column dKCdt;
            Eigen::VectorXd comp_drive(num_comp);
            Eigen::VectorXd pn_drive(p.kc.N);
            Eigen::VectorXd kc_apl_inh(p.kc.N);

            // TODO factor into generic row/col vec checking (or use appropriate eigen
            // types which guantee at compile time). currently just have both Row and
            // Column defined to be Matrix in .hpp file.
            check(rv.kc.wKCAPL.rows() > 1);
            check(rv.kc.wKCAPL.cols() == 1);

            check(rv.kc.wAPLKC.cols() > 1);
            check(rv.kc.wAPLKC.rows() == 1);

            for (unsigned t=p.time.start_step() + 1; t<p.time.steps_all(); ++t) {
                // (1) KC -> APL drive per compartment (Unchanged)
                const Eigen::VectorXd kc_activity =
                    (nves.col(t-1).array() * spikes.col(t-1).array()).matrix();

                // TODO also replace this def of claw_drive w/ call to
                // pn_to_kc_drive_at_t (adapting, if needed)
                // TODO can it also be initialized to this size in other two cases? yes,
                // right?
                Eigen::VectorXd claw_drive(rv.kc.wPNKC.rows());
                // TODO what is .noalias() doing here? would .array() also work? need it
                // at all?
                claw_drive.noalias() = rv.kc.wPNKC * pn_t.col(t);

                comp_drive.setZero();

                if (!p.kc.pn_claw_to_APL) {
                    // TODO refactor to use sum_across_...? add per-comparment handling
                    // there? separate fn for that?
                    for (unsigned comp=0; comp<num_comp; ++comp) {
                        double s = 0.0;
                        for (unsigned claw : claws_by_compartment[comp]) {
                            const unsigned kc = rv.kc.claw_to_kc[claw];
                            // TODO still redo indexing to avoid 0 if i can
                            // (use VectorXd [/Column] instead of Row vector?)
                            const double w_kc_apl = rv.kc.wKCAPL(0, claw);
                            s += w_kc_apl * kc_activity[kc];
                        }
                        comp_drive[comp] = 1e4 * s;
                    }
                } else {
                    for (unsigned comp=0; comp<num_comp; ++comp) {
                        double s = 0.0;
                        for (int claw : claws_by_compartment[comp]) {
                            const double w_kc_apl = rv.kc.wKCAPL(0, claw);
                            s += w_kc_apl * claw_drive[claw];
                        }
                        comp_drive[comp] = 0.2 * s;
                    }
                }
                // (2) APL Internal Dynamics (REVISED)
                // (2a) The synaptic current 'Is' is updated as before.
                Eigen::VectorXd dIs_comp_dt = -Is_prev_per_comp + comp_drive;

                // (2b) Calculate APL voltage change using the simplified capacitive model.
                const double inv_Cm = 1.0 / p.kc.apl_Cm;       // The single new parameter
                const double inv_taum = 1.0 / p.kc.apl_taum; // Existing parameter
                Eigen::VectorXd dVm_apl_dt = Eigen::VectorXd::Zero(num_comp);
                for (unsigned c=0; c<num_comp; ++c) {
                    // Synaptic current from KCs
                    const double I_syn = Is_prev_per_comp[c];

                    // Electrical coupling current from neighbors
                    const int L = (c - 1 + num_comp) % num_comp;
                    const int R = (c + 1) % num_comp;
                    const double I_coupling = g_voltage_coup * (Vm_apl_prev_per_comp[L] + Vm_apl_prev_per_comp[R]
                                                            - 2.0 * Vm_apl_prev_per_comp[c]);

                    // Sum the currents
                    const double I_total = I_syn + I_coupling;

                    // The change in voltage is the leak plus the integrated total current.
                    // dV/dt = -V/tau + I_total/Cm
                    dVm_apl_dt[c] = -inv_taum * Vm_apl_prev_per_comp[c] + inv_Cm * I_total;
                }

                // (2c) 'inh' variable is updated as before for other uses.
                Eigen::VectorXd dInh_comp_dt = -inh_prev_per_comp + Is_prev_per_comp;

                double dIsdt  = dIs_comp_dt.sum();
                double dinhdt = dInh_comp_dt.sum();

                // (3) APL -> KC Feedback (MODIFIED)
                pn_drive.setZero();
                kc_apl_inh.setZero();

                // use the *vector* per-compartment inhibition (previous step)
                for (unsigned comp=0; comp<num_comp; ++comp) {
                    const double apl_comp_prev = inh_prev_per_comp[comp];

                    for (unsigned claw : claws_by_compartment[comp]) {
                        const unsigned kc = rv.kc.claw_to_kc[claw];
                        pn_drive[kc] += claw_drive[claw];

                        // TODO delete need for this (seems it's effectively 1D either
                        // way. just have both branches initialize w/ consistent dim
                        // order...)
                        // TODO assert same whether we include 2nd 0 index or not (->
                        // delete)
                        const double w_apl_kc = rv.kc.wAPLKC(claw, 0);
                        // compartment-specific inhibition:
                        kc_apl_inh[kc] += w_apl_kc * apl_comp_prev;
                    }
                }

                dKCdt = (-Vm.col(t-1) + pn_drive - kc_apl_inh).array()
                        - use_ffapl * ffapl_t(t-1);
                Vm.col(t) = Vm.col(t-1) + dKCdt * (p.time.dt / p.kc.taum);

                // (5) Advance APL state variables (Unchanged from previous version)
                Is_curr_per_comp   = Is_prev_per_comp   + (p.time.dt / p.kc.tau_apl2kc) * dIs_comp_dt;
                inh_curr_per_comp  = inh_prev_per_comp  + (p.time.dt / p.kc.apl_taum)   * dInh_comp_dt;
                Vm_apl_curr_per_comp = Vm_apl_prev_per_comp + p.time.dt * dVm_apl_dt;

                Is_curr  = Is_prev  + (p.time.dt / p.kc.tau_apl2kc) * dIsdt;
                inh_curr = inh_prev + (p.time.dt / p.kc.apl_taum)   * dinhdt;

                // TODO delete commented code? (some non-commented code also unused tho?)
                // Is_curr  = Is_curr_per_comp.sum();
                // inh_curr = inh_curr_per_comp.sum();
                // TODO check equiv to a LHS w/o `0,` prefix (-> replace w/ that
                // syntax)?
                Is(0,t)  = Is_curr;
                inh(0,t) = inh_curr;

                // (6) Vesicles + thresholding (unchanged)
                nves.col(t) = nves.col(t-1);
                nves.col(t) += p.time.dt * ((1.0 - nves.col(t-1).array()).matrix()/p.kc.tau_r)
                            - (p.kc.ves_p * spikes.col(t-1).array() * nves.col(t-1).array()).matrix();

                auto const over_thr = Vm.col(t).array() > rv.kc.thr.array();
                spikes.col(t) = over_thr.select(1.0, spikes.col(t));
                Vm.col(t)     = over_thr.select(0.0, Vm.col(t));

                // TODO what is this doing?
                std::swap(Is_prev_per_comp,  Is_curr_per_comp);
                std::swap(Vm_apl_prev_per_comp, Vm_apl_curr_per_comp);
                std::swap(inh_prev_per_comp, inh_curr_per_comp);
                Is_prev  = Is_curr;
                inh_prev = inh_curr;
            }
        } else {
            Column dKCdt;
            for (unsigned t=p.time.start_step()+1; t < p.time.steps_all(); t++) {

                // TODO also declare size of vector here (currently done inside
                // pn_to_kc_drive_at_t)?
                // TODO TODO or should this be returning pn_drive? another fn to compute
                // that?
                Eigen::VectorXd claw_drive = pn_to_kc_drive_at_t(p, rv, pn_t, t,
                    bouton_sims
                );

                // Calculate the KC-level activity, a vector of size (p.kc.N, 1)
                Eigen::VectorXd kc_activity = (
                    nves.col(t-1).array() * spikes.col(t-1).array()
                ).matrix();
                // TODO double check python always sets p.kc.N to # KCs, never # claws
                // TODO add assertion (or at least log a warning) somewhere in setup
                // here, if p.kc.N >= n_claws (prob best to err, unless some flag set
                // explicitly allowing?)

                Eigen::VectorXd claw_drive_with_inh = (
                    // all of these have 1 col and #-claws rows (e.g. 9472),
                    // as does claw_sims.col(t)
                    claw_drive - rv.kc.wAPLKC * inh(t-1)
                );
                // TODO also set claw_sims in `apl_coup_const != -1` case
                // above (+ change math in same manner changed below), and also use
                // allow_net_inh_per_claw (alongside slight change to calculation, to
                // operate within each claw first) in that case
                claw_sims.col(t) = claw_drive_with_inh;

                if (!p.kc.allow_net_inh_per_claw) {
                    // there typically will be claws that would get sent negative b/c of
                    // inhibition, so we do need to clip if we want to avoid single
                    // claws contribution inhibition exceeding their excitation
                    auto const claw_drives_lt0 = claw_sims.col(t).array() < 0;
                    // replace per-claw drives to min of 0
                    claw_sims.col(t) = claw_drives_lt0.select(0.0, claw_sims.col(t));
                    // TODO make conditional (/delete)
                    check(claw_sims.col(t).minCoeff() >= 0);
                }

                // TODO move checks to top of loop (to only run once) (/ make
                // conditional)
                // TODO refactor (at least) this squeezing + dot code?
                check(rv.kc.wKCAPL.rows() == 1);
                // TODO this always true, or can it be a scalar essentially?
                check(rv.kc.wKCAPL.cols() > 1);
                double claw_apl_drive = rv.kc.wKCAPL.col(0).dot(claw_sims.col(t));

                // TODO delete (after verifying it passes -> removing .col(0) above)
                // TODO modify so we assert RHS has only one element, and then get
                // that element and check equal to LHS. currently not compiling b/c RHS
                // could be non-scalar eigen type (and may indeed have >1 element...)
                // ...
                //   libolfsysm/src/olfsysm.cpp:1866:17:   required from here
                //  libolfsysm/include/Eigen/src/Core/util/StaticAssert.h:140:29: error: static assertion failed: YOU_TRIED_CALLING_A_VECTOR_METHOD_ON_A_MATRIX
                //check(claw_apl_drive == rv.kc.wKCAPL.dot(claw_sims.col(t)));
                // still getting same error w/ these two lines
                //double claw_apl_drive2 = rv.kc.wKCAPL.dot(claw_sims.col(t));
                //check(claw_apl_drive == claw_apl_drive2);
                //

                // TODO TODO what is this 0.2 doing here? add as a parameter (/delete)?
                // TODO TODO this change behavior of my
                // allow_net_inh_per_claw=False path? add test that would catch it!
                claw_apl_drive = claw_apl_drive * 0.2;

                double dIsdt;
                if (!p.kc.pn_claw_to_APL) {
                    //double kc_apl_drive = sum_across_claws_within_each_kc(p, rv,
                    //    rv.kc.wKCAPL * kc_activity[kc]
                    //);
                    // TODO TODO probably have fn ouput vector of length # KCs, and then
                    // sum that to get scalar (may not need template type after all?)
                    double kc_apl_drive = 0.0;
                    for (unsigned claw=0; claw<n_claws; ++claw) {
                        unsigned kc = rv.kc.claw_to_kc[claw];
                        // TODO TODO need to pass a fn to sum_across.. to make this
                        // work, or can i precompute and pass something? how to expand
                        // kc_activity to # claws? separate fn for that?
                        kc_apl_drive += rv.kc.wKCAPL(claw) * kc_activity[kc];
                    }
                    // APL activity depends on KC spiking
                    dIsdt = -Is(t-1) + kc_apl_drive * 1e4;
                } else {
                    // APL activity does NOT depend on KC spiking. APL recieves direct
                    // inputs from the claws, including what might be subthreshold.
                    //
                    // TODO need to also add this branch for apl_coup_const !=
                    // -1 case above? tianpei ever implement for that case?
                    dIsdt = -Is(t-1) + claw_apl_drive;
                }

                double dinhdt = -inh(t-1) + Is(t-1);

                // TODO factor into a fn to sum over claws? (something w/ generic types
                // [how?], or at least a version both for vector and scalar
                // input/outputs?)
                // TODO TODO can i pass a function to access claw_sims like this? or
                // should i pass the data and a function to access it separately?
                // should this fn always be indexing, and thus not need an arbitrary fn
                // input (just something with .rows() == # claws?)?
                Eigen::VectorXd pn_drive = sum_across_claws_within_each_kc(p, rv,
                    claw_sims.col(t)
                );

                // NOTE: pn_drive includes inh (effect of wAPLKC) here, since calculated
                // per claw above
                dKCdt = (-Vm.col(t-1) + pn_drive).array() - use_ffapl * ffapl_t(t-1);

                Vm.col(t) = Vm.col(t-1) + dKCdt*p.time.dt/p.kc.taum;
                inh(t)    = inh(t-1)    + dinhdt*p.time.dt/p.kc.apl_taum;
                Is(t)     = Is(t-1)     + dIsdt*p.time.dt/p.kc.tau_apl2kc;

                nves.col(t) = nves.col(t-1);
                nves.col(t) += (p.time.dt *
                    ((1.0 - nves.col(t-1).array()).matrix() / p.kc.tau_r) -
                    (p.kc.ves_p*spikes.col(t-1).array() *
                     nves.col(t-1).array()).matrix()
                );

                if (p.kc.n_claws_active_to_spike < 0) {
                    //auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
                    // TODO TODO fix: (same issue if Row instead of Column)
                    // libolfsysm/include/Eigen/src/Core/util/XprHelper.h:816:96: error:
                    // static assertion failed:
                    // YOU_MIXED_DIFFERENT_NUMERIC_TYPES__YOU_NEED_TO_USE_THE_CAST_METHOD_OF_MATRIXBASE_TO_CAST_NUMERIC_TYPES_EXPLICITLY
                    //thr_comp = Vm.col(t).array() > rv.kc.thr.array();

                    // TODO TODO TODO this solution also work in newer instance of
                    // similar issue i'm having in fit_sparseness? or already tried this
                    // code there?
                    auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
                    // TODO de-dupe after fixing
                    // either go to 1 or _stay_ at 0.
                    spikes.col(t) = thr_comp.select(1.0, spikes.col(t));
                    // very abrupt repolarization!
                    Vm.col(t) = thr_comp.select(0.0, Vm.col(t));
                    //
                } else {
                    // TODO TODO what type is this? how to avoid compilation error about
                    // mixing types below?
                    auto const claw_thr_comp = claw_sims.col(t).array() > rv.kc.thr.array();

                    // NOTE: cast<double>() seems required to avoid compilation time
                    // Eigen complaint about mixing types here. Presumably the
                    // inequality above gives us something with int (unsigned?) type?
                    // TODO how to check type of above? just assign into a fixed type
                    // until it works?
                    // https://stackoverflow.com/questions/23946658
                    Eigen::VectorXd n_claws_active = sum_across_claws_within_each_kc(p,
                        rv, claw_thr_comp.cast<double>()
                    );

                    auto const thr_comp = n_claws_active.array() >= p.kc.n_claws_active_to_spike;

                    // TODO de-dupe after fixing
                    // either go to 1 or _stay_ at 0.
                    spikes.col(t) = thr_comp.select(1.0, spikes.col(t));
                    // TODO TODO just have Vm entirely undefined in this case (all 0 or
                    // something)? still used for anything (would prob be incorrect if
                    // so...)?
                    // very abrupt repolarization!
                    Vm.col(t) = thr_comp.select(0.0, Vm.col(t));
                    //

                    // TODO also want to de/re-polarize claws? make sense (w/ and w/o?)?
                }
                // TODO restore here after de-duping above (/delete)
                // either go to 1 or _stay_ at 0.
                //spikes.col(t) = thr_comp.select(1.0, spikes.col(t));
                // very abrupt repolarization!
                //Vm.col(t) = thr_comp.select(0.0, Vm.col(t));
            }
        }
    } else {
        Column dKCdt;
        // vector to store kc_apl_drive in each iteration
        for (unsigned t = p.time.start_step()+1; t < p.time.steps_all(); t++) {
            // TODO is this multiplication elementwise? is that the point if .array()?
            // (with matrices, by default, docs say it would be matrix-matrix
            // multiplication)
            Eigen::VectorXd kc_activity =
                (nves.col(t-1).array() * spikes.col(t-1).array()).matrix();

            Eigen::VectorXd kc_drive = pn_to_kc_drive_at_t(p, rv, pn_t, t, bouton_sims);

            // TODO move checks to top of loop (to only run once) (/ make conditional)
            check(rv.kc.wKCAPL.rows() == 1);
            // TODO this always true, or can it be a scalar essentially?
            check(rv.kc.wKCAPL.cols() > 1);
            const double kc_apl_drive = rv.kc.wKCAPL.col(0).dot(kc_activity);
            // TODO update all defs so wKCAPL is recognized as a vector at compile time?
            // currently this won't compile because of that.
            //const double kc_apl_drive = rv.kc.wKCAPL.dot(kc_activity);

            // use the scalar
            // TODO what is the 1e4 for?
            const double dIsdt = -Is(t-1) + kc_apl_drive * 1e4;
            double dinhdt = -inh(t-1) + Is(t-1);
            dKCdt = (
                (-Vm.col(t-1) + kc_drive -rv.kc.wAPLKC*inh(t-1)).array()
                -use_ffapl*ffapl_t(t-1)
            );

            Vm.col(t) = Vm.col(t-1) + dKCdt*p.time.dt/p.kc.taum;
            inh(t)    = inh(t-1)    + dinhdt*p.time.dt/p.kc.apl_taum;
            Is(t)     = Is(t-1)     + dIsdt*p.time.dt/p.kc.tau_apl2kc;

            nves.col(t) = nves.col(t-1);
            nves.col(t) += p.time.dt*((1.0-nves.col(t-1).array()).matrix()/p.kc.tau_r) - (p.kc.ves_p*spikes.col(t-1).array()*nves.col(t-1).array()).matrix();

            auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
            // either go to 1 or _stay_ at 0.
            spikes.col(t) = thr_comp.select(1.0, spikes.col(t));
            // TODO add assertion that checks spikes max is 1? or that unique values are
            // 0/1?

            // TODO describe how exactly this is working
            // very abrupt repolarization!
            Vm.col(t) = thr_comp.select(0.0, Vm.col(t));
        }
    }
    // TODO TODO assert nves is all 1, if ves_p == 0 (which it should be)?

    // TODO TODO TODO even if it seems true that inh/Is are all 0 on the first calls,
    // when picking threshold: *why* are they 0 there, when it doesnt seem like i was
    // always properly setting wAPLKC/wKCAPL to 0 for those calls [in case where
    // preset_*=true, i.e. use_connectome_APL_weights=True in python fit_mb_model]?
    // (is there something else that changed across the two calls in fit_sparseness?)
    // TODO delete
    /*
    rv.log(cat(
        "rv.kc.wAPLKC.isZero(): ", rv.kc.wAPLKC.isZero(),
        " (rv.kc.wAPLKC.array() == 0).all(): ", (rv.kc.wAPLKC.array() == 0.0).all(),
        "\nrv.kc.wKCAPL.isZero(): ", rv.kc.wKCAPL.isZero(),
        " (rv.kc.wKCAPL.array() == 0).all(): ", (rv.kc.wKCAPL.array() == 0.0).all(),
        "\ninh.isZero(): ", inh.isZero(),
        " (inh.array() == 0.0).all(): ", (inh.array() == 0.0).all(),
        "\nIs.isZero(): ", Is.isZero(),
        " (Is.array() == 0.0).all(): ", (Is.array() == 0.0).all()
    ));
    */
    //
}

void run_ORN_LN_sims(ModelParams const& p, RunVars& rv) {
    std::vector<unsigned> simlist = get_simlist(p);
#pragma omp parallel
    {
        Matrix orn_t(get_ngloms(p), p.time.steps_all());
        Row inhA(1, p.time.steps_all());
        Row inhB(1, p.time.steps_all());
#pragma omp for
        for (unsigned j = 0; j < simlist.size(); j++) {
            unsigned i = simlist[j];
            sim_ORN_layer(p, rv, i, orn_t);
            sim_LN_layer(p, orn_t, inhA, inhB);
#pragma omp critical
            {
                rv.orn.sims[i] = orn_t;
                rv.ln.inhA.sims[i] = inhA;
                rv.ln.inhB.sims[i] = inhB;
            }
        }
    }
}

void run_PN_sims(ModelParams const& p, RunVars& rv) {
    std::vector<unsigned> simlist = get_simlist(p);
#pragma omp parallel for
    for (unsigned j = 0; j < simlist.size(); j++) {
        unsigned i = simlist[j];
        sim_PN_layer(
                p, rv,
                rv.orn.sims[i], rv.ln.inhA.sims[i], rv.ln.inhB.sims[i],
                rv.pn.sims[i]);
    }
}

void run_FFAPL_sims(ModelParams const& p, RunVars& rv) {
    std::vector simlist = get_simlist(p);
#pragma omp parallel for
    for (unsigned j = 0; j < simlist.size(); j++) {
        unsigned i = simlist[j];
        sim_FFAPL_layer(
                p, rv,
                rv.pn.sims[i],
                rv.ffapl.vm_sims[i], rv.ffapl.coef_sims[i]);
    }
}

void run_KC_sims(ModelParams const& p, RunVars& rv, bool regen) {
    // TODO only do once? how? delete?
    // these should be defined in Eigen/src/Core/util/Macros.h
    // https://stackoverflow.com/questions/21497064
    // TODO factor into fn also printing OpenMP version into?
    // WORLD=3 MAJOR=3 MINOR=7 (world == major?)
    /*
    rv.log(cat(
        "EIGEN_WORLD_VERSION: ", EIGEN_WORLD_VERSION,
        " EIGEN_MAJOR_VERSION: ", EIGEN_MAJOR_VERSION,
        " EIGEN_MINOR_VERSION: ", EIGEN_MINOR_VERSION
    ));
    */

    #ifdef _OPENMP
        // TODO also log total # of threads?
        rv.log("using OpenMP multithreading");
    #else
        rv.log("running single threaded only");
    #endif

    // TODO delete
    rv.log(cat("p.pn.n_total_boutons=", p.pn.n_total_boutons));
    rv.log(cat("p.pn.preset_Btn=", p.pn.preset_Btn));
    rv.log(cat("rv.kc.wPNKC.rows()=", rv.kc.wPNKC.rows()));
    rv.log(cat("rv.kc.wPNKC.cols()=", rv.kc.wPNKC.cols()));
    rv.log("");
    //

    if (p.kc.n_claws_active_to_spike != -1) {
        check(p.kc.n_claws_active_to_spike > 0);
        check(p.kc.wPNKC_one_row_per_claw);
        // don't support homeostatic thresholds (so neither "hstatic" or "mixed")
        check( (p.kc.thr_type == "uniform") || (p.kc.thr_type == "fixed") );

        // not yet implemented for this code path
        check(p.kc.apl_coup_const == -1);

        // TODO also check that n_claws_active_to_spike is at least <= max # claws?
        // could go further and check at least the target frac of KCs have >= this many
        // claws (but may be easier to handle this checking when actually computing
        // claw threshold)
    }

    // checking we aren't requesting features not yet implemented in the compartmented
    // APL code.
    if (p.kc.apl_coup_const != -1) {
        check(!p.kc.save_claw_sims);
        // TODO TODO has tianpei fixed that in the code he messaged me about in early
        // 2026?
        // NOTE: currently just forcing to true (with a warning) in python fit_mb_model,
        // if allow_net_inh_per_claw=false (the default) when `APL_coup_const != -1`
        // NOTE: will have to change from default parameter on this one, in order to use
        // current comparmented APL code
        //
        // just b/c not implemented in that branch yet
        check(p.kc.allow_net_inh_per_claw);
    }

    if (regen) {
        if (p.kc.wPNKC_one_row_per_claw) {
            // TODO use this in other places i'm currently defining # claws some other
            // way?
            // TODO move this def outside of the `if (regen) { ... }` conditional?
            rv.kc.nclaws_total = rv.kc.claw_to_kc.size();
            // TODO accurate? delete
            rv.log(cat("rv.kc.nclaws_total: ", rv.kc.nclaws_total));

            // TODO delete (+ replace all usage of one with the other. prob just delete
            // rv.kc.nclaws_total)
            check(rv.kc.nclaws_total == p.kc.kc_ids.size());
            //

            // would almost certainly be a mistake. unlikely to ever want to test
            // 1-claw-per-KC (or some other case where some KCs have no claws, and total
            // sums up to same as # of KCs)
            check(rv.kc.nclaws_total > p.kc.N);
        } else {
            // TODO move this to default (-> delete this else)?
            rv.kc.nclaws_total = 0;
        }

        rv.log("generating new KC replicate");
        build_wPNKC(p, rv);
        // TODO want to add optional flag to allow fit_sparseness to run w/o re-gening
        // wPNKC? likely not relevant if i only need multiple run_KC_sims calls for
        // deterministic wPNKC (e.g. from hemibrain)? would just want to be able to get
        // rv.kc.pks and then pick thresholds based on that in python, with another
        // run_KC_sims call after
        fit_sparseness(p, rv);

        // TODO delete? redundant w/ check currently at start of fit_sparseness, at
        // least if it's true that `claw_compartments.size() == claw_to_kc.size()`
        // (add assertion for this latter thing, in fit_sparseness?)?
        // (may want to move / duplicate all checks at start of fit_sparesness to end of
        // that fn tho, only if there's a change they could be resized somewhere in
        // fit_sparesness)
        if (p.kc.wPNKC_one_row_per_claw) {
            check(rv.kc.wAPLKC.rows() == rv.kc.claw_compartments.size());
        } else {
            check(rv.kc.wAPLKC.rows() == p.kc.N);
        }
    }

    std::vector<unsigned> simlist = get_simlist(p);

#pragma omp parallel
    {
        Matrix Vm_here;
        if (!p.kc.save_vm_sims) {
            Vm_here = Matrix(p.kc.N, p.time.steps_all());
        }

        Matrix spikes_here;
        if (!p.kc.save_spike_recordings) {
            spikes_here = Matrix(p.kc.N, p.time.steps_all());
        }

        Matrix nves_here;
        if (!p.kc.save_nves_sims) {
            nves_here = Matrix(p.kc.N, p.time.steps_all());
        }
        Matrix inh_here;
        if (!p.kc.save_inh_sims) {
            inh_here = Matrix(1, p.time.steps_all());
        }

        Matrix Is_here;
        if (!p.kc.save_Is_sims) {
            Is_here = Matrix(1, p.time.steps_all());
        }

        Matrix claw_here;
        if (!p.kc.save_claw_sims) {
            // TODO (delete? am currently uncondtionally resizing at start of
            // sim_KC_layer tho, and i think that might have been requried...)
            // get # claws (matter? will it always be successfully resized later?)
            claw_here = Matrix(rv.kc.nclaws_total, p.time.steps_all());
        }

        Matrix respcol;
        Matrix respcol_bin;
#pragma omp for
        for (unsigned j = 0; j < simlist.size(); j++) {
            unsigned i = simlist[j];
            Matrix& Vm_link = p.kc.save_vm_sims
                ? rv.kc.vm_sims.at(i)
                : Vm_here;
            Matrix& spikes_link = p.kc.save_spike_recordings
                ? rv.kc.spike_recordings.at(i)
                : spikes_here;
            Matrix& nves_link = p.kc.save_nves_sims
                ? rv.kc.nves_sims.at(i)
                : nves_here;
            Matrix& inh_link = p.kc.save_inh_sims
                ? rv.kc.inh_sims.at(i)
                : inh_here;
            Matrix& Is_link = p.kc.save_Is_sims
                ? rv.kc.Is_sims.at(i)
                : Is_here;
            Matrix& claw_link = p.kc.save_claw_sims
                ? rv.kc.claw_sims.at(i)
                : claw_here;

            // TODO TODO TODO finish implementing these
            // TODO also want to allow disabling saving of these? could be a factor of
            // 10 or so higher than size of PN sims...
            Matrix& bouton_link = rv.pn.bouton_sims.at(i);

            sim_KC_layer(
                p, rv, rv.pn.sims[i], rv.ffapl.vm_sims[i], Vm_link, spikes_link,
                nves_link, inh_link, Is_link, claw_link, bouton_link
            );
            respcol = spikes_link.rowwise().sum();
            respcol_bin = (respcol.array() > 0.0).select(1.0, respcol);

#pragma omp critical
            rv.kc.responses.col(i) = respcol_bin;
            rv.kc.spike_counts.col(i) = respcol;
        }
    } // The parallel region ends here.

    // **ALL of the following code has been moved here, outside the parallel region,
    // to ensure it runs only after all threads have completed.**

    double final_sp = rv.kc.responses.mean();
    if (rv.kc.responses.hasNaN()) {
        rv.log("Warning: The rv.kc.responses matrix contains NaN values.");

        for (int r = 0; r < rv.kc.responses.rows(); ++r) {
            for (int c = 0; c < rv.kc.responses.cols(); ++c) {
                if (std::isnan(rv.kc.responses(r, c))) {
                    rv.log(cat("NaN found at row ", r, ", column ", c));
                }
            }
        }
    }
    // TODO TODO check no NaN in responses? (-> delete loop printing them above)
    // (equiv to some NaN check on an input? just check that?)

    rv.log(cat("Post-sim global sparsity (C++): ", final_sp));
    // TODO delete (/put behind some kind of verbose/debug flag)
    /*
    const unsigned n_kc    = rv.kc.responses.rows();
    const unsigned n_odors = rv.kc.responses.cols();

    // Sum of all responses (active KC–odor pairs)
    const double total_active = rv.kc.responses.sum(); // 0/1 matrix → sum is a count
    rv.log(cat("Total active KC-odor pairs: ", total_active, " / ",
               double(n_kc) * double(n_odors)));

    // Per-odor response (number of active KCs and fraction per odor)
    for (unsigned odor=0; odor<n_odors; ++odor) {
        const double col_sum  = rv.kc.responses.col(odor).sum();
        const double col_frac = col_sum / double(n_kc);  // per-odor sparsity
        rv.log(cat("Odor ", odor, ": active KCs = ", col_sum,
                   " (sparsity = ", col_frac, ")"));
    }
    */
}

void remove_before(unsigned step, Matrix& timecourse) {
    Matrix intermediate = timecourse.block(
            0,                 step,
            timecourse.rows(), timecourse.cols()-step);
    timecourse = intermediate;
}
// TODO TODO use this before saving / retrieving in mb_model?
void remove_all_pretime(ModelParams const& p, RunVars& r) {
    auto cut = [&p](Matrix& m) {
        remove_before(p.time.start_step(), m);
    };
    // TODO loop over n odors for all of this? should be same for each loop right?
    // assert that?
    // TODO delete eventually
    unsigned n_odors = get_nodors(p);
    check(r.orn.sims.size() == n_odors);
    check(r.ln.inhA.sims.size() == n_odors);
    check(r.ln.inhB.sims.size() == n_odors);
    check(r.pn.sims.size() == n_odors);
    if (p.pn.n_total_boutons > 0) {
        check(r.pn.bouton_sims.size() == n_odors);
    }
    if (p.kc.save_vm_sims) {
        check(r.kc.vm_sims.size() == n_odors);
    }
    if (p.kc.save_spike_recordings == n_odors) {
        check(r.kc.spike_recordings.size() == n_odors);
    }
    if (p.kc.save_nves_sims == n_odors) {
        check(r.kc.nves_sims.size() == n_odors);
    }
    if (p.kc.save_inh_sims == n_odors) {
        check(r.kc.inh_sims.size() == n_odors);
    }
    if (p.kc.save_Is_sims == n_odors) {
        check(r.kc.Is_sims.size() == n_odors);
    }
    if (p.kc.save_claw_sims == n_odors) {
        check(r.kc.claw_sims.size() == n_odors);
    }
    //
#pragma omp parallel
    {
        // ORN
#pragma omp for
        for (unsigned i = 0; i < n_odors; i++) {
            cut(r.orn.sims[i]);
        }
        // LN
#pragma omp for
        for (unsigned i = 0; i < n_odors; i++) {
            cut(r.ln.inhA.sims[i]);
        }
#pragma omp for
        for (unsigned i = 0; i < n_odors; i++) {
            cut(r.ln.inhB.sims[i]);
        }
        // PN
#pragma omp for
        for (unsigned i = 0; i < n_odors; i++) {
            cut(r.pn.sims[i]);
            if (p.pn.n_total_boutons > 0) {
                cut(r.pn.bouton_sims[i]);
            }
        }
#pragma omp for
        for (unsigned i = 0; i < n_odors; i++) {
            // TODO only do each of these, conditional on whether it's being saved?
            if (p.kc.save_vm_sims) {
                cut(r.kc.vm_sims[i]);
            }
            if (p.kc.save_spike_recordings) {
                cut(r.kc.spike_recordings[i]);
            }
            if (p.kc.save_nves_sims) {
                cut(r.kc.nves_sims[i]);
            }
            if (p.kc.save_inh_sims) {
                cut(r.kc.inh_sims[i]);
            }
            if (p.kc.save_Is_sims) {
                cut(r.kc.Is_sims[i]);
            }
            if (p.kc.save_claw_sims) {
                cut(r.kc.claw_sims[i]);
            }
        }
    }
}

// TODO TODO implement (along w/ change to np::to_npy fn to support std::vector<Matrix>)
// saving of all into one file per variable, rather than one file per variable per odor
// TODO proper way to pass directory in C++? will assume for now directory already
// exists
// TODO TODO add (optional?) bool arg to call remove_all_pretime first?
void save_odor_dynamics(ModelParams const& p, RunVars& r, std::string path, unsigned i) {
    // TODO proper way to append filesep to path (assuming it doesn't already have it on
    // input. what if it does?)?
    // TODO also put odor index somewhere? subdir? (assuming i can make a fn to save
    // everything after testing  this as an intermediate. will try to delete fn with
    // this name)
    np::to_npy(r.orn.sims[i], path + "/orn_sims");
    np::to_npy(r.ln.inhA.sims[i], path + "/inhA_sims");
    np::to_npy(r.ln.inhB.sims[i], path + "/inhB_sims");
    np::to_npy(r.pn.sims[i], path + "/pn_sims");

    if (p.pn.n_total_boutons > 0) {
        np::to_npy(r.pn.bouton_sims[i], path + "/bouton_sims");
    }
    if (p.kc.save_vm_sims) {
        np::to_npy(r.kc.vm_sims[i], path + "/vm_sims");
    }
    if (p.kc.save_spike_recordings) {
        np::to_npy(r.kc.spike_recordings[i], path + "/spike_recordings");
    }
    if (p.kc.save_nves_sims) {
        np::to_npy(r.kc.nves_sims[i], path + "/nves_sims");
    }
    if (p.kc.save_inh_sims) {
        np::to_npy(r.kc.inh_sims[i], path + "/inh_sims");
    }
    if (p.kc.save_Is_sims) {
        np::to_npy(r.kc.Is_sims[i], path + "/Is_sims");
    }
    if (p.kc.save_claw_sims) {
        np::to_npy(r.kc.claw_sims[i], path + "/claw_sims");
    }
}

std::vector<unsigned> get_simlist(ModelParams const& p) {
    if (p.sim_only.empty()) {
        std::vector<unsigned> ret(get_nodors(p));
        std::iota(std::begin(ret), std::end(ret), 0);
        return ret;
    } else { // Corrected: Use curly braces { } for the else block
        return p.sim_only;
    }
}

