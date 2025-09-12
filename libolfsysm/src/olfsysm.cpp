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
#include <cassert>
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

/* So code can be compiled single threaded, to support debugging.
 * Only other OMP references should be in the preprocessor directives, which I think can
 * just be ignored (though that will generate compilation warning, which is good).
 * https://stackoverflow.com/questions/7847900 */
#ifdef _OPENMP
   #include <omp.h>
#else
   #define omp_get_thread_num() 0
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

    p.kc.N                     = 2000;
    p.kc.nclaws                = 6;
    p.kc.uniform_pns           = false;
    p.kc.pn_drop_prop          = 0.0;
    p.kc.preset_wPNKC          = false;
    p.kc.seed                  = 0;
    p.kc.tune_apl_weights      = true;
    p.kc.preset_wAPLKC         = false;
    p.kc.preset_wKCAPL         = false;
    p.kc.ignore_ffapl          = false;
    p.kc.fixed_thr             = 0;
    p.kc.add_fixed_thr_to_spont= false;
    p.kc.use_fixed_thr         = false;
    p.kc.use_vector_thr        = false;
    p.kc.use_homeostatic_thrs  = true;
    p.kc.thr_type              = "";
    p.kc.sp_target             = 0.1;
    p.kc.sp_factor_pre_APL     = 2.0;
    p.kc.sp_acc                = 0.1;
    p.kc.sp_lr_coeff           = 10.0;

    p.kc.max_iters             = 10;
    p.kc.apltune_subsample     = 1;

    // TODO doc how each of these are diff (w/ units if i can). not currently mentioned
    // in .hpp file
    p.kc.taum                  = 0.01;
    p.kc.apl_taum              = 0.05;
    p.kc.tau_apl2kc            = 0.01;

    p.kc.tau_r                 = 1.0;
    // olfsysm.hpp says that setting this to 0 should disable synaptic depression
    // (tau_r above is another parameter for synaptic depression)
    p.kc.ves_p                 = 0.0;

    p.kc.save_vm_sims          = false;
    p.kc.save_spike_recordings = false;
    p.kc.save_nves_sims        = false;
    p.kc.save_inh_sims         = false;
    p.kc.save_Is_sims          = false;

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

/* Sample spontaneous PN output from odor 0. */
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
    sims(get_nodors(p), Matrix(get_ngloms(p), p.time.steps_all())) {
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
        ? int(p.kc.kc_ids.size())
        : int(p.kc.N),
        get_ngloms(p)
    ),

    wAPLKC( p.kc.wPNKC_one_row_per_claw ? int(p.kc.kc_ids.size()) : int(p.kc.N), 1 ),
    wKCAPL( 1, p.kc.wPNKC_one_row_per_claw ? int(p.kc.kc_ids.size()) : int(p.kc.N) ),

    wAPLKC_scale(1.0),
    wKCAPL_scale(1.0),

    thr(p.kc.N, 1),
    responses(p.kc.N, get_nodors(p)),
    spike_counts(p.kc.N, get_nodors(p)),

    vm_sims(p.kc.save_vm_sims ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    spike_recordings(p.kc.save_spike_recordings ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    nves_sims(p.kc.save_nves_sims ? get_nodors(p) : 0,
            Matrix(p.kc.N, p.time.steps_all())),
    inh_sims(p.kc.save_inh_sims ? get_nodors(p) : 0,
            Matrix(1, p.time.steps_all())),
    Is_sims(p.kc.save_Is_sims ? get_nodors(p) : 0,
            Matrix(1, p.time.steps_all())),
    tuning_iters(0)
{
    if (p.kc.wPNKC_one_row_per_claw) {
        const auto& raw = p.kc.kc_ids;  // One body ID per claw
        claw_to_kc.resize(raw.size());

        std::unordered_map<unsigned, int> id2idx;
        int nextIndex = 0;

        for (size_t i = 0; i < raw.size(); ++i) {
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
        assert(nextIndex == int(p.kc.N) && "Number of unique KC IDs must equal p.kc.N");
    } else {
        claw_to_kc.resize(0);  // For clarity, make sure it's empty
    }
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
        rv.log("preset_wPNKC");
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
Column sample_PN_spont(ModelParams const& p, RunVars const& rv) {
    /* Sample from halfway between time start and stim start to stim start. */
    unsigned sp_t1 =
        p.time.start_step()
        + unsigned((p.time.stim.start-p.time.start)/(2*p.time.dt));
    unsigned sp_t2 =
        p.time.start_step()
        + unsigned((p.time.stim.start-p.time.start)/(p.time.dt));
    return rv.pn.sims[0].block(0,sp_t1,get_ngloms(p),sp_t2-sp_t1).rowwise().mean();
}
Column choose_KC_thresh_uniform(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in) {
    unsigned tlist_sz = KCpks.cols();
    KCpks.resize(1, KCpks.size());                     // flatten
    std::sort(KCpks.data(), KCpks.data()+KCpks.size(),
            [](double a, double b){return a>b;});      // dec. order
    // TODO TODO log what we would get if we used values +/- 1 from the index used for
    // KCpks? (to try to figure out limits of precision in sparsity achievable through
    // setting threshold alone)
    double thr_const = KCpks(std::min(
        int(p.kc.sp_target * p.kc.sp_factor_pre_APL * double(p.kc.N*tlist_sz)),
        int(p.kc.N*tlist_sz)-1));
    return thr_const + spont_in.array()*2.0;
}
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

 
void fit_sparseness(ModelParams const& p, RunVars& rv) {
    rv.log("fitting sparseness");

    std::vector<unsigned> tlist = p.kc.tune_from;
    if (!tlist.size()) {
        for (unsigned i = 0; i < get_nodors(p); i++) tlist.push_back(i);
    }

    // fix? I don't need to re-size this right? it should be sized up correctly already? 
    // only in the situation when there's no preset_wAPLKC? 
    unsigned num_claws = rv.kc.claw_to_kc.size();
    if(!p.kc.preset_wAPLKC){
        if(!p.kc.wPNKC_one_row_per_claw){
            // if(rv.kc.wAPLKC.cols() != p.kc.N){
            //     rv.log(cat("rv.kc.wAPLKC.cols(): ", rv.kc.wAPLKC.cols()));
            //     rv.kc.wAPLKC.resize(p.kc.N,1);
            //     rv.kc.wKCAPL.resize(1,p.kc.N);
            // }    
            rv.log(cat("rv.kc.wAPLKC.rows():", rv.kc.wAPLKC.rows()));
            rv.log(cat("rv.kc.wKCAPL.cols():", rv.kc.wKCAPL.cols()));
        }else {
            // rv.log(cat("rv.kc.wAPLKC.cols():", rv.kc.wAPLKC.cols()));
            // rv.log(cat("rv.kc.wKCAPL.rows():", rv.kc.wKCAPL.rows()));
            // if(rv.kc.wAPLKC.cols() != num_claws){
            //     rv.kc.wAPLKC.resize(num_claws,1);
            //     rv.kc.wKCAPL.resize(1,num_claws);
            // }
            rv.log(cat("rv.kc.wAPLKC.rows():", rv.kc.wAPLKC.rows()));
            rv.log(cat("rv.kc.wKCAPL.cols():", rv.kc.wKCAPL.cols()));
        }
    } else {
        rv.log("p.kc.preset_wAPLKC is true");
    }
    int wAPLKC_nan_count = 0;
    // Check for NaN values in wAPLKC
    for (int i = 0; i < rv.kc.wAPLKC.rows(); ++i) {
        for (int j = 0; j < rv.kc.wAPLKC.cols(); ++j) {
            if (std::isnan(rv.kc.wAPLKC(i, j))) {
                wAPLKC_nan_count++;
            }
        }
    }

    int wKCAPL_nan_count = 0;
    // Check for NaN values in wKCAPL
    for (int i = 0; i < rv.kc.wKCAPL.rows(); ++i) {
        for (int j = 0; j < rv.kc.wKCAPL.cols(); ++j) {
            if (std::isnan(rv.kc.wKCAPL(i, j))) {
                wKCAPL_nan_count++;
            }
        }
    }
    /* Calculate spontaneous input to KCs. */
    // TODO log stuff about PN spont to figure out if part of that isn't init'd
    // properly?

    Column spont_in_ini = rv.kc.wPNKC * sample_PN_spont(p, rv);
    Column spont_in;
    if (p.kc.wPNKC_one_row_per_claw) {
        // Reduce per-claw -> per-KC by summation
        spont_in.resize(p.kc.N, 1);
        spont_in.setZero();

        // Fast path using claw_to_kc (length = num_claws)
        const auto& claw_to_kc = rv.kc.claw_to_kc;  // std::vector<unsigned>
        for (int claw = 0; claw < claw_to_kc.size(); ++claw) {
            unsigned kc = claw_to_kc[claw];
            // guard (in case of bad mapping)
            if (kc < (unsigned)spont_in.size()) {
                spont_in(kc) += spont_in_ini((Eigen::Index)claw);
            }
        }
    } else {
        spont_in = spont_in_ini;
    }
    rv.kc.spont_in = spont_in;
    {
        const auto& M = rv.kc.spont_in;          // Eigen::MatrixXd or VectorXd
        const double mean = M.array().mean();
        const double minv = M.minCoeff();
        const double maxv = M.maxCoeff();

        // Population variance
        const double var_pop = (M.array() - mean).square().mean();
        const double sd_pop  = std::sqrt(std::max(0.0, var_pop));

        // Sample variance (use this if you prefer N-1 in the denominator)
        const Eigen::Index N = M.size();
        const double var_samp = (N > 1)
            ? (M.array() - mean).square().sum() / double(N - 1)
            : 0.0;
        const double sd_samp = std::sqrt(std::max(0.0, var_samp));

        rv.log(cat("spont_in mean: ", mean));
        rv.log(cat("spont_in min: ",  minv));
        rv.log(cat("spont_in max: ",  maxv));
        rv.log(cat("spont_in sd(pop): ",  sd_pop));
        rv.log(cat("spont_in sd(sample): ", sd_samp));
    }

    Column wAPLKC_unscaled(p.kc.N, 1);
    Row wKCAPL_unscaled(1, p.kc.N);
    if (p.kc.wPNKC_one_row_per_claw) {
        wAPLKC_unscaled.resize(num_claws, 1);
        wKCAPL_unscaled.resize(1, num_claws);
    }

    if (p.kc.preset_wAPLKC) {
        // TODO delete
        rv.log(cat("INITIAL rv.kc.wAPLKC.mean(): ", rv.kc.wAPLKC.mean()));
        {   // sample standard deviation (ddof=1)
            const auto A = rv.kc.wAPLKC.array();
            const Eigen::Index n = rv.kc.wAPLKC.size();
            const double mu  = A.mean();
            const double var = (A - mu).square().sum() / std::max<Eigen::Index>(1, n - 1);
            const double sd  = std::sqrt(std::max(0.0, var));
            rv.log(cat("INITIAL rv.kc.wAPLKC sd(): ", sd));
        }        // should be a deep copy
        wAPLKC_unscaled = rv.kc.wAPLKC;

        // TODO delete
        rv.log(cat("INITIAL wAPLKC_unscaled.mean(): ", wAPLKC_unscaled.mean()));
    } else {
        rv.log(cat("INITIAL rv.kc.wAPLKC.mean(): ", rv.kc.wAPLKC.mean()));
        {   // sample standard deviation (ddof=1)
            const auto A = rv.kc.wAPLKC.array();
            const Eigen::Index n = rv.kc.wAPLKC.size();
            const double mu  = A.mean();
            const double var = (A - mu).square().sum() / std::max<Eigen::Index>(1, n - 1);
            const double sd  = std::sqrt(std::max(0.0, var));
            rv.log(cat("INITIAL rv.kc.wAPLKC sd(): ", sd));
        }        // should be a deep copy
    }
    if (p.kc.preset_wKCAPL) {
        // TODO delete
        rv.log(cat("INITIAL rv.kc.wKCAPL.mean(): ", rv.kc.wKCAPL.mean()));
        {   // sample standard deviation (ddof=1)
            const auto A = rv.kc.wKCAPL.array();
            const Eigen::Index n = rv.kc.wKCAPL.size();
            const double mu  = A.mean();
            const double var = (A - mu).square().sum() / std::max<Eigen::Index>(1, n - 1);
            const double sd  = std::sqrt(std::max(0.0, var));
            rv.log(cat("INITIAL rv.kc.wKCAPL sd(): ", sd));
        }
        wKCAPL_unscaled = rv.kc.wKCAPL;

        // TODO delete
        rv.log(cat("INITIAL wKCAPL_unscaled.mean(): ", wKCAPL_unscaled.mean()));
    } else {
        rv.log(cat("INITIAL rv.kc.wKCAPL.mean(): ", rv.kc.wKCAPL.mean()));
    }
    
    /* Set starting values for the things we'll tune. */
    // TODO matter? seems to be overwritten below in this case anyway...
    // (and put inside this conditional to avoid overwriting values set in python, via
    // pybind11)
    if (p.kc.tune_apl_weights) {
        if (!p.kc.preset_wAPLKC) {
            rv.kc.wAPLKC.setZero();
        }
        if (!p.kc.preset_wKCAPL) {
            if (p.kc.wPNKC_one_row_per_claw) {
                double preset_wKCAPL_base = 1.0/float(p.kc.N);
                for (Eigen::Index i_c = 0; i_c < rv.kc.claw_to_kc.size(); ++i_c) {
                    unsigned kc = rv.kc.claw_to_kc[i_c];
                    const std::size_t cnt = rv.kc.kc_to_claws[kc].size(); // claws of this KC
                    const double val = preset_wKCAPL_base / static_cast<double>(cnt ? cnt : 1);
                    rv.kc.wKCAPL(i_c, 0) = val;  // row vector
                }
            } else {
                rv.kc.wKCAPL.setConstant(1.0/float(p.kc.N));
            }
        }
    }
    // TODO check that, in NOT p.kc.tune_apl_weights case, wAPLKC and wKCAPL are
    // appropriately initialized? maybe also in preset_wAPLKC/preset_wKCAPL = true
    // cases above?

    if (!p.kc.use_vector_thr) {
        if (!p.kc.use_fixed_thr) {
            rv.kc.thr.setConstant(1e5); // higher than will ever be reached
        }
        else {
            rv.log(cat("using FIXED threshold: ", p.kc.fixed_thr));
            // TODO would it ever make sense to have add_fixed_thr_to_spont=False?
            // when? in any cases i use? doc
            if (p.kc.add_fixed_thr_to_spont) {
                // TODO delete + replace w/ similar commented line below
                // (after confirming the 2 things w/ factor 2 cancel out...)
                rv.log("adding fixed threshold to 2 * spontaneous PN input to each KC");
                //rv.log("adding fixed threshold to spontaneous PN input to each KC");
                // TODO TODO what are units of spont_in? doc these as units of fixed_thr
                rv.kc.thr = p.kc.fixed_thr + spont_in.array()*2.0;
            } else {
                rv.kc.thr.setConstant(p.kc.fixed_thr);
            }
        }
    } else {
        rv.log("using prespecified vector KC thresholds");
        // TODO even want to allow `add_fixed_thr_to_spont = False`? don't think it's
        // useful now
        if (p.kc.add_fixed_thr_to_spont) {
            rv.log("adding threshold to 2 * spontaneous PN input to each KC");

            // TODO delete
            // TODO do i need .array() here? also, i assuming changing <x>.array() also
            // changes values in <x> (assuming it's a Matrix/similar)?
            rv.log(cat("(before adding spont) rv.kc.thr.mean(): ", rv.kc.thr.mean()));

            // TODO this line working as intended? (do need LHS .array() to avoid err,
            // at least w/ RHS as it is here)
            rv.kc.thr.array() += spont_in.array()*2.0;

            // TODO delete
            // TODO do i need .array() here?
            rv.log(cat("(after adding spont) rv.kc.thr.mean(): ", rv.kc.thr.mean()));
        }
    }

    // TODO delete print
    rv.log(cat("p.kc.N at KCpks declaration, ", p.kc.N));
    /* Used for measuring KC voltage; defined here to make it shared across all
     * threads.*/
    Matrix KCpks(p.kc.N, tlist.size()); KCpks.setZero();

    /* Used to store odor response data during APL tuning. */
    Matrix KCmean_st(p.kc.N, 1+ ((tlist.size() - 1) / p.kc.apltune_subsample));

    // TODO TODO should this not be computed on first iteration?
    // (would need to do in a way that preserves behavior in hemibrain-repro test, or at
    // least add this as a param i can set from python to preserve that behavior)
    // NOTE: currently need to keep this initial value as-is, in order to reproduce (at
    // least) the hemibrain paper responses exactly.
    /* Used to store the current sparsity.
     * Initially set to the below value because, given default model
     * parameters, it causes tuning to complete in just one iteration. */
    double sp = 0.0789;

    /* Used to count number of times looped; the 'learning rate' is decreased
     * as 1/sqrt(count) with each iteration. */
    rv.kc.tuning_iters = 0;

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
        (abort(), TTINVALID);

    /* Break up into threads. */
#pragma omp parallel
    {
        /* Output matrices for the KC simulation. */
        Matrix Vm(p.kc.N, p.time.steps_all());
        Matrix spikes(p.kc.N, p.time.steps_all());
        Matrix nves(p.kc.N, p.time.steps_all());
        Row inh(1, p.time.steps_all());
        Row Is(1, p.time.steps_all());

        // TODO delete (assuming i want this for use_vector_thr. why don't i for
        // TTFIXED?)
        // if (thrtype != TTFIXED && !p.kc.use_vector_thr) {
        if (thrtype != TTFIXED && !p.kc.use_vector_thr) {
#pragma omp single
            {
                // TODO print str value for thrtype instead? (may need to add something
                // to invert mapping above. seems like some cases above currently don't
                // use the existing string p.kc.thr_type [but that could be changed?])
                rv.log(cat("choosing thresholds from spontaneous input (thrtype=",
                           thrtype, ")"));
            }

            // TODO TODO maybe i still want to sim_KC_layer in use_vector_thr case
            // (just not use it to pick a thr)?

            /* Measure voltages achieved by the KCs, and choose a threshold
             * based on that. */
#pragma omp for
            for (unsigned i = 0; i < tlist.size(); i++) {
                sim_KC_layer(p, rv,
                        rv.pn.sims[tlist[i]], rv.ffapl.vm_sims[tlist[i]],
                        Vm, spikes, nves, inh, Is);
#pragma omp critical
                KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*2.0;
            }

#pragma omp single
            {
                // TODO TODO need to redefine these after end of fit_sparseness
                // (so they are actually accurate and useful in mb_model's use to
                // compute per-subtype thresholds) (currently just hardcoding thresholds
                // rather than trying to compute them from pks in python)
                rv.kc.pks = KCpks;
                /*for (unsigned w = 0; w < rv.kc.pks.rows(); w++) {
                    for (unsigned z = 0; z < rv.kc.pks.cols(); z++) {
                        if (rv.kc.pks(w,z) < -1e20) abort();
                    }
                }*/

                // TODO TODO make a new variable, like rv.kc.pks, but only set at the
                // end (so as to also include the APL's influence). store the same peak
                // KC Vms (or whatever exact quantity pks is)? (same thing comment above
                // is asking for, just into a new variable)

                /* Finish picking thresholds. */
                rv.kc.thr =
                    (thrtype == TTHSTATIC ? choose_KC_thresh_homeostatic :
                     thrtype == TTMIXED ? choose_KC_thresh_mixed :
                     choose_KC_thresh_uniform)
                    (p, KCpks, spont_in);
                // TODO TODO compute + log sparsity here? (from KCpks)
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
        if (!p.kc.tune_apl_weights && p.kc.preset_wAPLKC) {
            // TODO delete
            rv.log(cat("FIXED rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

            rv.kc.wAPLKC = rv.kc.wAPLKC_scale * wAPLKC_unscaled;
        }
        if (!p.kc.tune_apl_weights && p.kc.preset_wKCAPL) {
            // TODO delete
            rv.log(cat("FIXED rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));

            rv.kc.wKCAPL = rv.kc.wKCAPL_scale * wKCAPL_unscaled;
        }
        }

        // TODO TODO in use_vector_thr=True case, want to at least log/save the
        // mean response rate before APL (esp if rv.kc.thr not set appropriately there,
        // which maybe could have been used in python to compute that?)

        // TODO if `!tune_apl_weights` just return here, so i can de-ident code below?
        // or does some or it need to run?
        /* Enter this region only if APL use is enabled; if disabled, just exit
         * (at this point APL->KC weights are set to 0). */
        if (p.kc.tune_apl_weights) {
#pragma omp single
        {
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
            if (!p.kc.preset_wAPLKC) {
                // e.g. 3 w/ sp_target=0.1
                if(p.kc.wPNKC_one_row_per_claw){
                    const double base = 2.0 * ceil(-log(p.kc.sp_target));
                    for (Eigen::Index claw = 0; claw < rv.kc.claw_to_kc.size(); ++claw) {
                        unsigned kc = rv.kc.claw_to_kc[claw];
                        const std::size_t cnt = rv.kc.kc_to_claws[kc].size(); // claws of this KC
                        const double val = base / static_cast<double>(cnt ? cnt : 1);
                        rv.kc.wAPLKC(claw, 0) = val;  // row vector
                    }
                    // rv.kc.wAPLKC.setConstant(2*ceil(-log(p.kc.sp_target)));
                } else {
                    rv.kc.wAPLKC.setConstant(2*ceil(-log(p.kc.sp_target)));
                }  
                rv.log(cat("setConst initial rv.kc.wAPLKC sum: ", rv.kc.wAPLKC.sum()));
                rv.log(cat("setConst initial rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));
            } else {
                rv.kc.wAPLKC_scale = 2*ceil(-log(p.kc.sp_target));
                // TODO delete
                rv.log(cat("INITIAL rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

                rv.kc.wAPLKC = rv.kc.wAPLKC_scale * wAPLKC_unscaled;
            }
            if (!p.kc.preset_wKCAPL) {
                if(p.kc.wPNKC_one_row_per_claw){
                    const double base = 2*ceil(-log(p.kc.sp_target)) / double(p.kc.N);
                    for (Eigen::Index claw = 0; claw < rv.kc.claw_to_kc.size(); ++claw) {
                        unsigned kc = rv.kc.claw_to_kc[claw];
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
                // if(p.kc.wPNKC_one_row_per_claw){
                //     rv.kc.wKCAPL_scale = 2*ceil(-log(p.kc.sp_target)) / double(rv.kc.claw_to_kc.size());
                // } else {
                //     rv.kc.wKCAPL_scale = 2*ceil(-log(p.kc.sp_target)) / double(p.kc.N);
                // }
                // TODO delete
                // this definition is a big problem. how would the math work to accomplish this? 
                rv.log(cat("INITIAL rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));

                rv.kc.wKCAPL = rv.kc.wKCAPL_scale * wKCAPL_unscaled;
            }
            // TODO have code fail (terminate w/o achieving target sp) [or backtrack
            // somehow] if count of either changes (don't want to add 0s)
            int n_wAPLKC_lte0_initial = (rv.kc.wAPLKC.array() <= 0.0).count();
            int n_wKCAPL_lte0_initial = (rv.kc.wKCAPL.array() <= 0.0).count();
            rv.log(cat("n_wAPLKC_lte0_initial: ", n_wAPLKC_lte0_initial));
            rv.log(cat("n_wKCAPL_lte0_initial: ", n_wKCAPL_lte0_initial));
        }

        /* Continue tuning until we reach the desired sparsity. */
        do {
            //rv.log(cat("** t", omp_get_thread_num(), " @ top"));
#pragma omp barrier

#pragma omp single
            {
                /* Modify the APL<->KC weights in order to move in the
                 * direction of the target sparsity. */
                double lr = p.kc.sp_lr_coeff / sqrt(double(rv.kc.tuning_iters));
                double delta = (sp - p.kc.sp_target) * lr / p.kc.sp_target;
                // TODO log initial value of delta?

                if (!p.kc.preset_wAPLKC) {
                    // TODO why using .array() for +=, but not for direct assignment
                    // operations? is .array() actually necessary in this case?
                    // what does .array() do?
                    if(p.kc.wPNKC_one_row_per_claw){
                        double change = delta ;
                        for (Eigen::Index claw = 0; claw < rv.kc.claw_to_kc.size(); ++claw) {
                            unsigned kc = rv.kc.claw_to_kc[claw];
                            const std::size_t cnt = rv.kc.kc_to_claws[kc].size(); // claws of this KC
                            const double val = change / static_cast<double>(cnt ? cnt : 1);
                            rv.kc.wAPLKC(claw, 0) += val;  // row vector
                        }
                        // ? this as well? should we update different claws differently? 
                    } else {
                        rv.kc.wAPLKC.array() += delta;
                    }
                    rv.log(cat("rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));
                } else {
                    rv.kc.wAPLKC_scale += delta;

                    // TODO delete?
                    rv.log(cat("rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));

                    rv.kc.wAPLKC = rv.kc.wAPLKC_scale * wAPLKC_unscaled;
                }

                if (!p.kc.preset_wKCAPL) {
                    if(p.kc.wPNKC_one_row_per_claw){
                        double change = delta / double(p.kc.N);
                        for (Eigen::Index claw = 0; claw < rv.kc.claw_to_kc.size(); ++claw) {
                            unsigned kc = rv.kc.claw_to_kc[claw];
                            const std::size_t cnt = rv.kc.kc_to_claws[kc].size(); // claws of this KC
                            const double val = change / static_cast<double>(cnt ? cnt : 1);
                            rv.kc.wKCAPL(0, claw) += val;  // row vector
                        }
                        // ? this as well? should we update different claws differently? 
                    } else {
                        rv.kc.wKCAPL.array() += delta / double(p.kc.N);
                    }
                    rv.log(cat("rv.kc.wKCAPL mean: ", rv.kc.wKCAPL.mean()));
                } else {
                    rv.kc.wKCAPL_scale += delta / double(p.kc.N);
                    // if(p.kc.wPNKC_one_row_per_claw){
                    //     rv.kc.wKCAPL_scale += delta / double(rv.kc.claw_to_kc.size());
                    // } else {
                    //     rv.kc.wKCAPL_scale += delta / double(p.kc.N);
                    // }

                    // TODO delete?
                    rv.log(cat("rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));

                    rv.kc.wKCAPL = rv.kc.wKCAPL_scale * wKCAPL_unscaled;
                }

                // TODO TODO probably want to abort (so we can change tuning params and
                // re-run) rather than clip values (which would break overall shape of
                // vector(s) from connectome). or otherwise take steps to avoid this
                // state (would probably be better if we didn't have to abort).
                // (could give people a message to choose different step size param)
                /* If we learn too fast in the negative direction we could end
                 * up with negative weights. */
                if (delta < 0.0) {
                    if (!p.kc.preset_wAPLKC) {
                        int n_wAPLKC_lt0 = (rv.kc.wAPLKC.array() < 0.0).count();
                        rv.log(cat("n_wAPLKC_lt0: ", n_wAPLKC_lt0));

                        // TODO TODO at least log that this is happening (doesn't
                        // already mean we have any negative weights, just b/c we are in
                        // this block tho... would need to know if there are any < 0)?
                        rv.kc.wAPLKC = (rv.kc.wAPLKC.array() < 0.0).select(
                                0.0, rv.kc.wAPLKC);
                    }

                    if (!p.kc.preset_wKCAPL) {
                        int n_wKCAPL_lt0 = (rv.kc.wKCAPL.array() < 0.0).count();
                        rv.log(cat("n_wKCAPL_lt0: ", n_wKCAPL_lt0));

                        rv.kc.wKCAPL = (rv.kc.wKCAPL.array() < 0.0).select(
                                0.0, rv.kc.wKCAPL);
                    }
                }

                rv.log(cat( "* i=", rv.kc.tuning_iters,
                            ", sp=", sp,
                            ", wAPLKC_delta=", delta,
                            ", lr=", lr));

                // TODO delete
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

                rv.kc.tuning_iters++;
            }

            //rv.log(cat("** t", omp_get_thread_num(), " @ before testing"));
            /* Run through a bunch of odors to test sparsity. */
#pragma omp for
            for (unsigned i = 0; i < tlist.size(); i+=p.kc.apltune_subsample) {
                sim_KC_layer(p, rv,
                        rv.pn.sims[tlist[i]], rv.ffapl.vm_sims[tlist[i]],
                        Vm, spikes, nves, inh, Is);
                KCmean_st.col(i / p.kc.apltune_subsample) = spikes.rowwise().sum();

//#pragma omp critical
                // TODO delete?
                ////KCpks.col(i) = Vm.rowwise().maxCoeff(); // - spont_in*2.0;
                // TODO probably restore
                //KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*2.0;
                ////KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*10.0;
            }
            //rv.log(cat("** t", omp_get_thread_num(), " @ after testing"));

#pragma omp single
            {
                // TODO delete
                //rv.log(cat("KCpks.mean(): ", KCpks.mean()));
                //rv.log(cat("spont_in.mean(): ", spont_in.mean()));
                //
                // TODO restore? (+ fix surrounding) (or probably better set, set
                // post-APL peaks into new rv.kc variable...)
                // don't think i could use same way as i do for prior pks [which I use
                // in python to set thresholds, in a similar manner to how they are used
                // in here] tho, so might be pointless.
                // more complicated by this point, since also depend on activity of all
                // other KCs, so don't think i can as easily use to set e.g. a single
                // KC's APL weights.
                //rv.kc.pks = KCpks;

                KCmean_st = (KCmean_st.array() > 0.0).select(1.0, KCmean_st);
                sp = KCmean_st.mean();
            }

            // TODO TODO why have multiple threads each printing these, if always the
            // same across each (actually true? indicate a bug?)
            rv.log(cat("** t", omp_get_thread_num(), " @ before bottom cond [",
                        "sp=", sp,
                        ", i=", rv.kc.tuning_iters,
                        ", tgt=", p.kc.sp_target,
                        ", acc=", p.kc.sp_acc,
                        ", I=", p.kc.max_iters,
                        "]"));
        } while ((abs(sp - p.kc.sp_target) > (p.kc.sp_acc * p.kc.sp_target))
                && (rv.kc.tuning_iters <= p.kc.max_iters));
        //rv.log(cat("** t", omp_get_thread_num(), " @ exit"));
#pragma omp barrier
#pragma omp single
        {
            rv.kc.tuning_iters--;
        }
    }}
    // TODO delete?
    rv.log(cat("FINAL rv.kc.wAPLKC_scale: ", rv.kc.wAPLKC_scale));
    rv.log(cat("FINAL rv.kc.wKCAPL_scale: ", rv.kc.wKCAPL_scale));
    rv.log(cat("FINAL rv.kc.wAPLKC mean: ", rv.kc.wAPLKC.mean()));
    rv.log(cat("FINAL rv.kc.wKCAPL mean: ", rv.kc.wKCAPL.mean()));

    // TODO always log tuned parameters at end (fixed_thr, wAPLKC/wKCAPL when not
    // preset, or wAPLKC_scale/wKCAPL_scale when preset)
    rv.log("done fitting sparseness");
}

void sim_ORN_layer(
        ModelParams const& p, RunVars const& rv,
        int odorid,
        Matrix& orn_t) {
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
void sim_LN_layer(
        ModelParams const& p,
        Matrix const& orn_t,
        Row& inhA, Row& inhB) {
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
        //response(t) = potential(t) > lnp.thr ? potential(t)-lnp.thr : 0.0;
        response(t) = (potential(t)-p.ln.thr)*double(potential(t)>p.ln.thr);
    }
}
void sim_PN_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& orn_t, Row const& inhA, Row const& inhB,
        Matrix& pn_t) {
    // TODO verify this isn't actually making noise (both params 0? or sd at least?)?
    // it should be seed-able if it is
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
            200.0*((orn_delta.array()+p.pn.offset)*p.pn.tanhsc/200.0*inh_PN).matrix().unaryExpr<double(*)(double)>(&tanh);
        add_randomly([&noise](){return noise(g_randgen);}, dPNdt);

        inh_PN = p.pn.inhsc/(p.pn.inhadd+0.25*inhA(t)+0.75*inhB(t));
        pn_t.col(t) = pn_t.col(t-1) + dPNdt*p.time.dt/p.pn.taum;

        // TODO TODO why not do something like this in sim_ORN_layer case too?
        // ann also handle the 2 cases the same way?
        pn_t.col(t) = (0.0 < pn_t.col(t).array()).select(pn_t.col(t), 0.0);
    }
}
void sim_FFAPL_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t,
        Vector& ffapl_t, Vector& coef_t) {
    ffapl_t.setZero();
    coef_t.setZero();

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
    Matrix& Vm, Matrix& spikes, Matrix& nves, Matrix& inh, Matrix& Is)
{
    // Determine number of compartments
    // int n_compartments = rv.kc.claw_compartments.maxCoeff() + 1;

    Vm.setZero();
    spikes.setZero();
    nves.setOnes();
    inh.setZero();
    Is.setZero();

    float use_ffapl = float(!p.kc.ignore_ffapl);
    if (p.kc.wPNKC_one_row_per_claw) {
        Column dKCdt;
        double total_claw_drive = 0.0;
        double total_pn_drive = 0.0;
        double total_kc_apl_inh = 0.0;
        for (unsigned t = p.time.start_step()+1; t < p.time.steps_all(); t++) {
            // Calculate the KC-level activity, a vector of size (p.kc.N, 1)
            Eigen::VectorXd kc_activity = (nves.col(t-1).array() * spikes.col(t-1).array()).matrix();

            // Sum the weighted activity of all KCs to get a single APL input value.
            // This resolves the dimension mismatch.
            double kc_apl_drive = 0.0;
            const Eigen::Index n_claws = rv.kc.claw_to_kc.size();
            for (Eigen::Index claw = 0; claw < n_claws; ++claw) {
                unsigned kc = rv.kc.claw_to_kc[claw];
                kc_apl_drive += rv.kc.wKCAPL(claw, 0) * kc_activity[kc];
            }

            double dIsdt = -Is(t-1) + kc_apl_drive * 1e4;

            double dinhdt = -inh(t-1) + Is(t-1);
            // claw-level drive: one entrty per claw
            // rv.kc.wPNKC: a matrix of size (nClaws x nGlos)
            // pn_t.col(t): a vector of size (nGloms) givine the PN activity at time step t.
            // multiplication: standard matrix-vector, a length-nClaws VectorXd
            Eigen::VectorXd claw_drive = rv.kc.wPNKC * pn_t.col(t);       // size = nClaws

            // collapse to true KC-level drive
            // initialize KC-level accumulator
            // pn_drive is a placeholder for the summed drive each KC will recieve
            // p.kc.N is the number of KCs

            Eigen::VectorXd pn_drive = Eigen::VectorXd::Zero(p.kc.N);      // size = nKCs
            for (Eigen::Index claw = 0; claw < n_claws; ++claw) {
                unsigned kc = rv.kc.claw_to_kc[claw]; // already 0..N-1
                pn_drive[kc] += claw_drive[claw];
            }

            // --- FIX: Map the APL inhibition from claw level to KC level ---
            Eigen::VectorXd kc_apl_inh = Eigen::VectorXd::Zero(p.kc.N); // size = nKCs
            for (Eigen::Index claw = 0; claw < n_claws; ++claw) {
                unsigned kc = rv.kc.claw_to_kc[claw];
                // The APL inhibition is weighted by the APL->KC weight
                // and applied to the corresponding KC.
                kc_apl_inh[kc] += rv.kc.wAPLKC(claw, 0) * inh(t - 1);
            }

            total_claw_drive += claw_drive.mean();
            total_pn_drive += pn_drive.mean();
            total_kc_apl_inh += kc_apl_inh.mean();
            // --- Now use the correctly sized KC-level inhibition ---
            dKCdt =
                (-Vm.col(t-1)
                + pn_drive
                - kc_apl_inh).array() // Now this term has the correct size
                - use_ffapl * ffapl_t(t-1);

            Vm.col(t) = Vm.col(t-1) + dKCdt*p.time.dt/p.kc.taum;
            inh(t)    = inh(t-1)    + dinhdt*p.time.dt/p.kc.apl_taum;
            Is(t)     = Is(t-1)     + dIsdt*p.time.dt/p.kc.tau_apl2kc;

            nves.col(t) = nves.col(t-1);
            nves.col(t) += p.time.dt*((1.0-nves.col(t-1).array()).matrix()/p.kc.tau_r) - (p.kc.ves_p*spikes.col(t-1).array()*nves.col(t-1).array()).matrix();

            auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
            spikes.col(t) = thr_comp.select(1.0, spikes.col(t)); // either go to 1 or _stay_ at 0.
            Vm.col(t) = thr_comp.select(0.0, Vm.col(t)); // very abrupt repolarization!
        }
    } else {
        Column dKCdt;
        Eigen::VectorXd kc_apl_drive_ts;
        const unsigned t0 = p.time.start_step() + 1;
        const unsigned tN = p.time.steps_all();
        const Eigen::Index T = static_cast<Eigen::Index>(tN - t0);
        kc_apl_drive_ts.resize(T);
        kc_apl_drive_ts.setZero();  
        // vector to store kc_apl_drive in each iteration 
        for (unsigned t = p.time.start_step()+1; t < p.time.steps_all(); t++) {
            // if(t == p.time.steps_all()-1){
            //     rv.log(cat("last step: nves.col(t-1): ", nves.col(t-1).mean()));
            //     rv.log(cat("last step: spikes.col(t-1): ", spikes.col(t-1).mean()));
            // } 
            Eigen::VectorXd kc_activity =
                (nves.col(t-1).array() * spikes.col(t-1).array()).matrix();
            // 1xN * Nx1 -> 1x1, then extract the (0,0) scalar
            const double kc_apl_drive = (rv.kc.wKCAPL * kc_activity)(0,0);
            kc_apl_drive_ts(static_cast<Eigen::Index>(t - t0)) = kc_apl_drive;
            // use the scalar
            const double dIsdt = -Is(t-1) + kc_apl_drive * 1e4;
            double dinhdt = -inh(t-1) + Is(t-1);
            // store kc_apl_drive into the vecotr
            dKCdt =
                (-Vm.col(t-1)
                +rv.kc.wPNKC*pn_t.col(t)
                -rv.kc.wAPLKC*inh(t-1)).array()
                -use_ffapl*ffapl_t(t-1);

            Vm.col(t) = Vm.col(t-1) + dKCdt*p.time.dt/p.kc.taum;
            inh(t)    = inh(t-1)    + dinhdt*p.time.dt/p.kc.apl_taum;
            Is(t)     = Is(t-1)     + dIsdt*p.time.dt/p.kc.tau_apl2kc;

            nves.col(t) = nves.col(t-1);
            nves.col(t) += p.time.dt*((1.0-nves.col(t-1).array()).matrix()/p.kc.tau_r) - (p.kc.ves_p*spikes.col(t-1).array()*nves.col(t-1).array()).matrix();

            auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
            //Eigen::Index n_spiking = thr_comp.count();   // number of KCs above threshold
            
            spikes.col(t) = thr_comp.select(1.0, spikes.col(t)); // either go to 1 or _stay_ at 0.
            Vm.col(t) = thr_comp.select(0.0, Vm.col(t)); // very abrupt repolarization!
        }
        //rv.log(cat("kc_apl_drive mean: ", kc_apl_drive_ts.mean()));
        // rv.log(cat("VM mean: ", Vm.mean()));
        // rv.log(cat("After sim_KC_layer: ", "wAPLKC mean: ", rv.kc.wAPLKC.mean(), ", ", "Vm mean: ", Vm.mean(), ", ", "Spikes mean: ", spikes.mean()));
    }
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

            /*
            sim_ORN_layer(p, rv, i, rv.orn.sims[i]);
            sim_LN_layer(
                    p, rv.orn.sims[i],
                    rv.ln.inhA.sims[i], rv.ln.inhB.sims[i]);
                    */
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
    if (regen) {
        rv.log("generating new KC replicate");
        // TODO want to add optional flag to allow fit_sparseness to run w/o re-gening
        // wPNKC? likely not relevant if i only need multiple run_KC_sims calls for
        // deterministic wPNKC (e.g. from hemibrain)? would just want to be able to get
        // rv.kc.pks and then pick thresholds based on that in python, with another
        // run_KC_sims call after
        rv.kc.nclaws_total = rv.kc.claw_to_kc.size();
        build_wPNKC(p, rv);

        // If KC_row = True, wAPLKC and wKCAPL will have length of number of KCs, otherwise they will have the length of number of claws
        // Number of unique weights = number of compartments
        // const size_t n = std::min<size_t>(10, p.kc.kc_ids.size());

        fit_sparseness(p, rv);
        if (p.kc.wPNKC_one_row_per_claw){
            assert(rv.kc.wAPLKC.rows() == int(rv.kc.claw_compartments.size()));
        } else {
            assert(rv.kc.wAPLKC.rows() == int(p.kc.N));
        }
    }

    std::vector<unsigned> simlist = get_simlist(p);
    rv.log(cat("simlist size in run_KC_sims, ", simlist.size()));
    rv.log(cat("size of p.kc.N in run_KC_sims, ", p.kc.N));

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

    // TODO doc why we now have to do this (/ delete)
    rv.kc.responses.resize(p.kc.N, get_nodors(p));
    rv.kc.spike_counts.resize(p.kc.N, get_nodors(p));

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
        // TODO TODO where does this one get saved to?
        Matrix& Is_link = p.kc.save_Is_sims
            ? rv.kc.Is_sims.at(i)
            : Is_here;
        // rv.log("after constructing all the matrices");
        sim_KC_layer(
            p, rv,
            rv.pn.sims[i], rv.ffapl.vm_sims[i],
            Vm_link, spikes_link, nves_link, inh_link, Is_link);
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

        // You can also iterate and print them for more detailed debugging
        for (int r = 0; r < rv.kc.responses.rows(); ++r) {
            for (int c = 0; c < rv.kc.responses.cols(); ++c) {
                if (std::isnan(rv.kc.responses(r, c))) {
                    rv.log(cat("NaN found at row ", r, ", column ", c));
                }
            }
        }
    }
    rv.log(cat("Post-sim global sparsity (C++): ", final_sp));
    const Eigen::Index n_kc    = rv.kc.responses.rows();
    const Eigen::Index n_odors = rv.kc.responses.cols();

    // Sum of all responses (active KC–odor pairs)
    const double total_active = rv.kc.responses.sum(); // 0/1 matrix → sum is a count
    rv.log(cat("Total active KC-odor pairs: ", total_active, " / ",
               double(n_kc) * double(n_odors)));

    // Per-odor response (number of active KCs and fraction per odor)
    for (Eigen::Index odor = 0; odor < n_odors; ++odor) {
        const double col_sum  = rv.kc.responses.col(odor).sum();
        const double col_frac = col_sum / double(n_kc);  // per-odor sparsity
        rv.log(cat("Odor ", odor, ": active KCs = ", col_sum,
                   " (sparsity = ", col_frac, ")"));
    }
}

void remove_before(unsigned step, Matrix& timecourse) {
    Matrix intermediate = timecourse.block(
            0,                 step,
            timecourse.rows(), timecourse.cols()-step);
    timecourse = intermediate;
}
void remove_all_pretime(ModelParams const& p, RunVars& r) {
    auto cut = [&p](Matrix& m) {
        remove_before(p.time.start_step(), m);
    };
#pragma omp parallel
    {
        // ORN
#pragma omp for
        for (unsigned i = 0; i < r.orn.sims.size(); i++) {
            cut(r.orn.sims[i]);
        }
        // LN
#pragma omp for
        for (unsigned i = 0; i < r.ln.inhA.sims.size(); i++) {
            cut(r.ln.inhA.sims[i]);
        }
#pragma omp for
        for (unsigned i = 0; i < r.ln.inhB.sims.size(); i++) {
            cut(r.ln.inhB.sims[i]);
        }
        // PN
#pragma omp for
        for (unsigned i = 0; i < r.pn.sims.size(); i++) {
            cut(r.pn.sims[i]);
        }
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

