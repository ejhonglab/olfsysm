#ifndef OLFSYSM_H_
#define OLFSYSM_H_

#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/Sparse"

/* Used for thread-safe logging. */
class Logger {
private:
    mutable std::ofstream fout;   // must be mutable for writing in const context
    mutable std::mutex mtx;       // must be mutable to allow locking in const context
    // TODO need this mutable? assume not. should just be set at top.
    // redirect call also just called at top and it uses mutex though... idk
    bool _tee;
public:
    Logger();
    /* Throw an error. */
    Logger(Logger const& other);

    /* Log a message. */
    void operator()(std::string const& msg) const;

    /* Log a blank line. */
    void operator()() const;

    /* Begin appending output to the given file. */
    void redirect(std::string const& path);

    /* Also write to stdout (in addition to file from any redirect calls, whether
     * before/after). */
    void tee();

    /* Turn off writing to stdout */
    void tee_off();

    /* Shut off output to file from redirect. Does not affect stdout output from tee().
     * */
    void disable();

    /* If >0, uses this color code in a control sequence prefix ("\x1b[<COLOR-CODE>m")
     * to each line printed to tee output (currently stdout), with a corresponding
     * control code to terminate the colored text at the end of each log output
     * ("\x1b[0m").
     *
     * Some valid color codes can be taken from python termcolor.COLORED dict
     * (name->color code). Current contents of that variable for me:
     * {'black': 30,
     *  'grey': 30,
     *  'red': 31,
     *  'green': 32,
     *  'yellow': 33,
     *  'blue': 34,
     *  'magenta': 35,
     *  'cyan': 36,
     *  'light_grey': 37,
     *  'dark_grey': 90,
     *  'light_red': 91,
     *  'light_green': 92,
     *  'light_yellow': 93,
     *  'light_blue': 94,
     *  'light_magenta': 95,
     *  'light_cyan': 96,
     *  'white': 97}
     * */
    int tee_color;
};

/* Use to show intent. */
using Matrix = Eigen::MatrixXd;
using Row    = Matrix;
// TODO change to vectors? why both matrices? i assume neither is enforced to actually
// have shape 1 in right place anywhere? how could i enforce that?
using Column = Matrix;
using Vector = Matrix;
using SparseMatrix = Eigen::SparseMatrix<double>;

/* Contain all model parameters; never contains data generated during
 * modeling! */
struct ModelParams {
    /* Timeline params. */
    struct Time {
        /* Time to start simulating ORN/LN/PN layers (give them time to
         * settle). */
        double pre_start;

        /* Start/end of KC simulation. */
        double start;
        double end;

        /* Start/end of stimulus presentation. */
        struct Stim {
            double start;
            double end;

            /* Calculate the pretime-relative stimulus start step. */
            unsigned start_step() const;

            /* Calculate the pretime-relative stimulus end step. */
            unsigned end_step() const;

            /* Get a row of length time.steps_all() with ones wherever the
             * stimulus is present, and zeros wherever it is not. */
            Row row_all() const;

            /* Internal only! */
            Time& _owner;
            Stim(Time&);
        } stim;

        /* Simulation timestep. */
        double dt;

        /* Calculate the pretime-relative start step. */
        unsigned start_step() const;

        /* Calculate the total number of timesteps (pre_start to end). */
        unsigned steps_all() const;

        /* Calculate the number of "real" timesteps (start to end). */
        unsigned steps() const;

        /* Get a row of ones with length steps_all(). */
        Row row_all() const;

        Time();
        Time(Time const&);
    } time;

    /* ORN params. */
    struct ORN {
        /* Membrane time constant. */
        double taum;

        /* The number of gloms in the physical system; used to scale input to
         * LNs. */
        unsigned n_physical_gloms;

        /* ORN spike-rate info (the model input). Not set by DEFAULT_PARAMS! */
        struct Data {
            /* Spontaneous rates; n_gloms x 1.*/
            Column spont;
            /* Firing rate changes in response to odors; n_gloms x n_odors. */
            Column delta;
        } data;
    } orn;

    /* LN params. */
    struct LN {
        /* Time constants. */
        double taum;
        double tauGA;
        double tauGB;

        /* Firing threshold. */
        double thr;

        /* Inhibition calculation params. */
        double inhsc;
        double inhadd;
    } ln;

    /* PN params. */
    struct PN {
        /* Time constant. */
        double taum;

        /* Inhibition calculation params. */
        double offset;
        double tanhsc;
        double inhsc;
        double inhadd;

        /* Gaussian noise parameters. */
        struct Noise {
            double mean;
            double sd;
        } noise;

        /* bouton related below */

        /* The total # of boutons. Should be >0 (and > #-glomeruli, but < #-claws) if
         * using PN<>APL weights (which should be of length #-boutons), and should
         * otherwise be 0. */
        // TODO rename, either this or nclaws_total (that even used?), to be consistent?
        unsigned n_total_boutons;

        /* Set to true if using pre-loaded APL->bouton / bouton->APL weight vectors (as
         * kc.preset_wAPLKC). Requires n_total_boutons > 0 and
         * kc.wPNKC_one_row_per_claw=true. */
        bool preset_wAPLPN;
        bool preset_wPNAPL;

        /* Multiplied into the normalized wAPLPN / wPNAPL vectors (as
         * kc.wAPLKC_pretune_scale). Default 1. */
        double wAPLPN_pretune_scale;
        double wPNAPL_pretune_scale;

        /* Rate-based (Tsodyks-Markram mean-field) vesicle depletion at each PN bouton.
         * If bouton_ves_p > 0, each bouton b has an available-pool fraction n_b in
         * [0, 1], with forward Euler steps of
         *   dn_b/dt = (1 - n_b) / bouton_tau_r - bouton_ves_p * r_b(t-1) * n_b,
         * where r_b is the bouton's release-driving rate: its rate after any APL>bouton
         * inhibition (preset_wAPLPN) and clipping at 0, before depletion. Each odor
         * starts from the steady state for the bouton's spontaneous rate r_spont_b
         * (sampled as for rv.kc.spont_in):
         *   n_b(0) = 1 / (1 + bouton_ves_p * bouton_tau_r * r_spont_b).
         * Each bouton's output is n_b(t) * r_b(t): rv.pn.bouton_sims hold this depleted
         * output, so the claw drive (wPNKC * bouton_sims), the bouton>APL drive
         * (preset_wPNAPL) and saved bouton_sims all see it. The spontaneous input to
         * KCs (rv.kc.spont_in, which thresholds are relative to) uses the same
         * transfer, wPNKC * (n_b(0) .* r_spont). n_b is clamped to [0, 1] after each
         * step (only binding if dt * (1 / bouton_tau_r + bouton_ves_p * r) > 1, where
         * forward Euler is unstable anyway). Requires n_total_boutons > 0.
         * bouton_ves_p (release fraction per spike, as rates are in Hz): default 0 (off;
         * outputs unchanged). bouton_tau_r (pool recovery time constant, in seconds):
         * default 0.2. */
        double bouton_ves_p;
        double bouton_tau_r;

        // TODO delete (/use)
        //double apl_taum;
        //double tau_apl2pn;
        //
    } pn;

    /* KC params. */
    struct KC {
        /* The number of KCs. */
        unsigned N;

        /* The number of claws assigned to each KC. */
        unsigned nclaws;

        /* Whether to use uniform PN choice, or use observational data. */
        bool uniform_pns;
        /* Weighted PN distribution data; required if uniform_pns is false. */
        Row cxn_distrib;
        /* The proportion of PN connections that should be (stochastically)
         * dropped during wPNKC generation. This is useful for simulating what
         * the draw of a reduced system would look like if it were taken by
         * truncating the connectivity matrix of the complete system. Takes
         * values in the interval [0,1). */
        double pn_drop_prop;

        /* Set to true if using a pre-loaded KC-PN connectivity matrix.
         * Overrides `uniform_pns`. */
        bool preset_wPNKC;

        /* RNG seed to be used for KC-PN connectivity matrix generation. If
         * seed=0, then a seed is generated by a std::random_device. */
        unsigned seed;

        /* Multiplicative current weights assigned to each PN. If left empty,
         * then this is treated as a row of ones. */
        // TODO use this for subtype specific weights? or do odor inputs overwrite this?
        Eigen::VectorXd currents;

        // TODO TODO reword matt's doc. not true. (APL still used, weights just
        // don't change)
        /* Whether to simulate the APL at all.*/
        bool tune_apl_weights;

        /* Set to true if using a pre-loaded APL->KC weight vector (e.g. connectome
         * synapse counts), set into rv.kc.wAPLKC before fit_sparseness. Its relative
         * values are kept, and its overall scale (rv.kc.wAPLKC_scale) is tuned (or
         * fixed, if tune_apl_weights=false). If false, APL->KC weights are uniform if
         * tuned, or used as set in rv.kc.wAPLKC otherwise. See `init_apl_weights` in
         * olfsysm.cpp for how vectors are normalized. */
        bool preset_wAPLKC;
        /* As preset_wAPLKC, but for the KC->APL weight vector. */
        bool preset_wKCAPL;

        /* If true (default), preset APL weight vectors are normalized when
         * fit_sparseness starts (see `init_apl_weights` in olfsysm.cpp). Set false to
         * use them as-is (e.g. if already normalized). */
        bool normalize_apl_weights;

        /* Only relevant for normalizing preset KC<>APL weights when
         * wPNKC_one_row_per_claw=true. If false (default), per-claw weights are scaled
         * to have a mean of 1 across claws. If true, they are scaled so that each KC's
         * total (summed over its claws) has a mean of 1 across KCs, as for
         * one-row-per-KC weights. */
        bool apl_weights_kc_mean;

        /* Multiplied into the normalized wAPLKC / wKCAPL vectors (preset or uniform)
         * before tuning. Since tuning scales all APL weights together, these set the
         * strength of each set of weights relative to the others. Default 1. */
        double wAPLKC_pretune_scale;
        double wKCAPL_pretune_scale;

        /* If false, APL activity will depend on KC spiking, and then a KC spiking will
         * cause all of its claws to provide input to the APL (just multiplied by
         * wKCAPL).
         *
         * If true, APL activity will NOT depend on KC spiking, and APL will get input
         * (which can be subthreshold) directly from each claw (though any APL
         * inhibition [-> rectification] to each claw will be applied first).
         *
         * Only relevant if wPNKC_one_row_per_claw=true. */
        bool pn_claw_to_apl;

        bool claw_dynamics;
        double claw_tau;

        /* Number of APL compartments ("units", e.g. spatial regions of the APL arbor in
         * the calyx). If 1 (default), the APL is a single compartment. If >1, each unit
         * has its own potential, gets input from and inhibits claws (and boutons)
         * according to rv.kc.w[APLKC|KCAPL]_units (and rv.pn.w[APLPN|PNAPL]_units), and
         * is electrotonically coupled to other units via rv.kc.apl_unit_coupling (see
         * sim_KC_layer). Requires wPNKC_one_row_per_claw=true. */
        unsigned n_apl_units;

        /* If true, KCs also inhibit each other directly (APL-independent lateral
         * inhibition), with weights rv.kc.wKCKC. Each KC's synaptic output (its spikes,
         * times its vesicle depletion factor nves, as for KC>APL synapses) is low-pass
         * filtered into an activity `a` (in Hz), with time constant kc_kc_tau:
         *   da/dt = (-a + nves(t-1) * spikes(t-1) / p.time.dt) / kc_kc_tau
         * (so each spike adds 1/kc_kc_tau to `a`, then decays, and `a` integrates to
         * the number of spikes; for constant firing at r Hz, `a` -> r). rv.kc.wKCKC * a
         * is then subtracted from each KC's dVm/dt (at the soma, i.e. after summing
         * claw inputs, if wPNKC_one_row_per_claw=true), in the same units as the
         * PN>KC drive. Not supported with n_claws_active_to_spike > 0 (spiking would
         * not depend on Vm). Default false (outputs unchanged). */
        bool use_kc_kc_inh;
        /* Time constant (seconds) of the filtered KC activity driving KC-KC inhibition.
         * Only relevant if use_kc_kc_inh=true. Default 0.05. */
        double kc_kc_tau;

        /* Per-claw saturation of the claw>soma transfer: the linear-summation form of
         * the n_claws_active_to_spike coincidence rule (bounds how much any one claw can
         * contribute to its KC). If > 0, each claw's drive x (claw_sims: after APL
         * inhibition and the clipping at 0) enters the sum across claws as
         *   claw_sat * tanh(x / claw_sat)   (claw_sat_hard=false, default), or
         *   min(x, claw_sat)                (claw_sat_hard=true),
         * both for the spontaneous input (rv.kc.spont_in, so thresholds are relative to
         * the saturated resting drive) and in sim_KC_layer. claw_sims themselves (and
         * the claw>APL drive, if pn_claw_to_apl=true) are left unsaturated, as under
         * n_claws_active_to_spike > 0. Requires wPNKC_one_row_per_claw=true; not
         * supported with n_claws_active_to_spike > 0 (spiking would not depend on the
         * summed drive). Default 0 (off; outputs unchanged). */
        double claw_sat;
        bool claw_sat_hard;
        /* Per-claw threshold (offset) of the claw>soma transfer, applied before claw_sat:
         * each claw's drive x enters as max(x - claw_thr, 0) (then saturated, if
         * claw_sat > 0). With claw_thr at the per-claw threshold of a
         * n_claws_active_to_spike model and a claw_sat of the same order, the summed
         * drive approximates that rule's count of active claws (a rectified, bounded
         * claw rather than a thresholded binary one). Same scope / requirements as
         * claw_sat. Default 0 (off; outputs unchanged). */
        double claw_thr;

        /* Ignore the FFAPL during KC simulation, even if run_FFAPL_sims has
         * been called. */
        bool ignore_ffapl;

        /* Optionally set a fixed KC firing threshold, instead of using the
         * normally generated thresholds. */
        double fixed_thr;

        /* If this is -1 fixed_thr (+ related) are for per-KC spike thresholds, but if
         * this is positive, then that all threshold parameters define per-claw
         * thresholds instead, and this many claws will need to exceed threshold to
         * cause their KC to spike. Only relevant if wPNKC_one_row_per_claw=true.
         *
         * NOTE: automated threshold picking not currently implemented when this is >0.
         * Had manually hardcoded particular threshold until response rate was within
         * tolerance, when testing this code so far.
         * */
        int n_claws_active_to_spike;

        // TODO doc this
        bool add_fixed_thr_to_spont;
        bool use_fixed_thr;
        /* If True, set rv.kc.thr directly */
        bool use_vector_thr;

        /* Use homeostatic (instead of uniform) KC thresholding.
         * Is overridden by use_fixed_thr. */
        bool use_homeostatic_thrs;

        /* One of "uniform", "hstatic", "mixed", or "fixed" if set.
         * If set, this option overrides other flags. Otherwise, the
         * thresholding type is decided by the use_fixed_thr and
         * use_homeostatic_thrs flags.
         * Uniform: traditional thresholding where all thresholds are the same,
         *   +/- spontaneous activity.
         * Hstatic: homeostatic thresholding where each KC is given its own
         *   threshold based on lifetime input recieved.
         * Mixed: the average of uniform and hstatic.
         * Fixed: all thrs are set to a fixed value (fixed_thr). */
        std::string thr_type;

        /* The target sparsity. */
        double sp_target;

        /* (sp_target * sp_factor_pre_apl) is sparsity (>sp_target) achieved by setting
         * KC spike thresholds alone, then sparsity is brought down to sp_target by
         * tuning APL. Must be >1, but small enough that (sp_target *
         * sp_factor_pre_apl) <= 1.0 */
        double sp_factor_pre_apl;

        /* Specifies the fraction +/- of the given target that is considered an
         * acceptable sparsity. For threshold tuning (only relevant if
         * `n_spikes_for_response > 1`) or APL weight tuning, respectively. */
        double thr_sp_acc;
        double sp_acc;

        /* Only used for `n_spikes_for_response > 1` case */
        double thr_sp_lr_coeff;

        // TODO change doc? it's 1/sqrt(n) not 1/n^2, right?
        /* Changes the scaling of the ~1/(n^2) tuning step-size curve. */
        double sp_lr_coeff;

        /* ~1/n instead of default ~1/sqrt(n) tuning step-size curve, if true */
        bool linear_lr_falloff;

        /* If we [over/under]shoot target sparsity, start a binary search, rather than
         * hoping whatever step size falloff strategy will get us to convergence in a
         * timely manner. */
        bool binary_search_on_overshoot;

        unsigned n_spikes_for_response;

        // See comment in .cpp file.
        bool hardcode_initial_sp;

        /* The maximum number of tuning iterations that should be done before
         * aborting. Must be >=1. */
        unsigned max_iters;

        /* List of (0-based!) IDs of odors that should be used for APL/sparsity
         * tuning. If the list is empty, then it will be ignored and instead
         * all odors will be used. */
        std::vector<unsigned> tune_from;

        /* X where every Xth odor in tune_from is used to estimate sparsity
         * during APL tuning. */
        unsigned apltune_subsample;

        /* Time constants. */
        double taum;
        double apl_taum;
        double tau_apl2kc;

        /* Synaptic depression params; see Hennig 2013 equation 3. Synaptic
         * depression can be disabled by setting ves_p = 0. */
        // TODO TODO say for which synapse this applies to (PN>KC? KC>APL?)
        double tau_r;
        double ves_p;

        // TODO maybe it's important that we actually get some baseline KC
        // spiking (at the slow ~0.1Hz or whatever is observed), so that single spikes
        // matter less in response phase. turner/bazhenov/laurent 2007 define response
        // as KCs spike rate exceeding baseline by 3.5 SD, at any time in the 2s after
        // odor onset. they also estimated lower sparsity 6+-5% ([1, 11%] or 5% of .06?
        // former i assume?). they see 14/71 = 19% silent cells, across their 25 odors.
        // all their odors at -2 (1:100) in pfo, and 1:10 air dilution (same air
        // dilution as ours). each KC was tested w/ a random 10/25 odors, on average
        //
        // also from turner/bazhenov/laurent 2007:
        // "a'/b' KCs were more broadly tuned than the other two types"
        // "Overall, a trend of decreasing responsiveness could be detected from a'/b'
        // to a/b to y KCs, with statistically significant differences only between y
        // and a'/b' KCs"
        //
        // "In addition to being more broadly tuned, a'/b' KCs had the highest baseline
        // firing rate and the most vigorous responses to odors: they fired 4.9+/-3.0
        // spikes during an odor response, significantly more than a/b KCs 2.2+/-1.2"
        //
        // (when KCs were held at 58+/-2 mV, if that matters)
        // "EPSP rise time was short (~2ms)"
        // with Vhold = -60mV voltage clamp:
        // "EPSP 10-90% rise times were 2.1+/-0.5ms, and decays were well fit with a
        // single exponential (11.5+/-5.3ms)"
        // NOTE: some comments are about EPSC [=currents] and some are about EPSP
        // [=potentials]
        // "EPSC rise time was 0.9+/-0.4ms, and decay time constant was 2.8+/-1.2ms"
        // "Although EPSP kinetics were fast, the membrane time constant, measured at
        // the soma by hyperpolarizing current injection, was very long (200 ms).  These
        // observations suggest that EPSP kinetics are determined mostly by synaptic
        // (and possibly, voltage-gated) conductances in the dendrites."
        // "The total number of PN outputs in the mushroom body is the product of three
        // quantities: the number of PNs projecting to the mushroom body; the number of
        // synaptic boutons per PN; and the number release sites per terminal."
        // TODO TODO do we have the # release sites from connectome data? should i have
        // wPNKC entries by that for each bouton? is this one of those things prat had
        // been trying to get across for a while?
        //
        // "model unitary EPSP amplitude was chosen to be either the mean (1.4 mV) or
        // the median (1.2 mV) of the experimental distribution"
        // NOTE: these EPSP amplitudes are as measured at the soma. i assume they should
        // be considerably larger in the boutons? any rough way to estimate, other than
        // doing simulation? i suppose it doesn't matter, since i'm not implementing
        // that spatial decay from claw->KC, it probably actually makes more sense to
        // use the value measured at the soma
        //
        // "The difference between resting potential and spike threshold is shown for 17
        // KCs in Fig. 3G (21.5+/-5.6mV). The average holding potential for the KCs was
        // -57.8mV; thus we set the voltage threshold for our model KCs at -36.3mV."
        //
        // NOTE: they report a respones rate more like 6%, so maybe that's a target
        // sparsity i should try?
        //
        // TODO TODO which 8 odors do they use at hallem concentrations (fig 5)? what
        // response rates / spiking do they estimate for each (and how do my current
        // outputs compare?)
        //
        // NOTE: their baseline a'/b' firing rate is ~0.25-0.3 Hz it seems, with other
        // two classes essentially at 0 (maybe 2E is just one example cell of each
        // though? what's average, if they report?)
        //
        // TODO TODO can i recreate their fig 5 A, w/ angular separation between odors
        // in ORN (Hallem) and KC space (w/ my model KCs)?
        // "The angular separation between vectors representing two different odors was
        // calculated as 1-cos(alpha), where alpha is the angle between the vectors"
        //
        // TODO TODO do they ever report IPSC/P amplitudes? clear at all from some of
        // their plots?
        //
        // TODO TODO what do i need to do to get the offset responses they see e.g. in
        // 1E, to 1-hexanol (or similar offset responses remy sees). just inhibition
        // that decays faster than the PN>KC synapse and compartment dynamics? or must
        // it be feedforward inhibition (e.g. from inhibitory PNs / LH)? it's not just
        // a single spike, so to the extent that a spike resets claw potentials, ig
        // there is still some transmitter at the synapse / some channels open? add
        // something to reflect that? and actually, those cells do have a small onset
        // response too (1-2 spikes), so that behavior would also be acceptable

        /* Output options. */
        bool save_vm_sims;
        bool save_spike_recordings;
        bool save_nves_sims;
        bool save_inh_sims;
        bool save_Is_sims;
        bool save_claw_sims;
        /* Saves the potential of each APL unit (to rv.kc.apl_unit_inh_sims). Only
         * relevant if n_apl_units > 1. */
        bool save_apl_unit_sims;

        // TODO delete kc_ids, and replace w/ setting N appropriately (or new N-like var
        // just for total_n_claws)
        // (only ever used for .size(), and doesn't even have IDs set in same form as in
        // other KC<>CLAW ID maps)
        std::vector<long long> kc_ids;

        bool wPNKC_one_row_per_claw;
        bool allow_net_inh_per_claw;
    } kc;

    /* Feedforward APL params. */
    struct FFAPL {
        /* Time constants. */
        double taum;

        /* PN->APL synaptic strength. */
        // TODO (delete. FFAPL stuff is all in a step prior to KC simulation. probably
        // don't want that, and would rather have PN<>APL interactions integrated as
        // part of same step as KC<>APL simulations) if going to start by trying to
        // adapt FFAPL for use w/ new connectome wPNAPL weights, change this to vector?
        // prob won't be able to use FFAPL to do all of what i want, since it was only
        // ever simulated in a separate step before run_KC_sims, it seems
        double w;

        /* The input into the APL is calculated as
         *   w * (summed output of PNs) * (coef)
         * where coef is some function of the firing rate distribution of PNs:
         * - "gini"
         * - "lts" (lifetime sparseness) (<- current default) */
        std::string coef;

        /* Whether to set the spontaneous FFAPL output to zero. */
        bool zero;

        /* Whether to stop the FFAPL output from dropping below spont. */
        bool nneg;

        /* coef = [1 - a*G]+, where G is the Gini coefficient of the
         * population of firing rates and a is declared below.
         * See: https://en.wikipedia.org/wiki/Gini_coefficient */
        struct Gini {
            /* Coefficient on G. */
            double a;

            // TODO is there a G variable actually defined somewhere? what is G?
            /* How to compute G. Options:
             * - "=": use PN firing rates directly
             * - "-spont": subtract spontaneous PN firing rates first
             * - "/spont": divide by spontaneous rates first
             * - "(-s)/s": use (firing rate - spont rate)/spont rate */
            std::string source;
        } gini;

        /* coef = m+L(1-m), where L is the instantaneous lifetime
         * sparseness of the odor. Takes values in the range [1,m]. */
        struct LTS {
            double m; /* default: 1.5 */
        } lts;
    } ffapl;

    /* Only (re?)simulate the given odors. If empty, simulate everything. */
    std::vector<unsigned> sim_only;
};
// this is referenced in python bindings, despite only being defined in .cpp
extern ModelParams const DEFAULT_PARAMS;

/* Variables and storage space that is useful to each run.
 * Matrices that are not used (e.g., KC-related matrices when KC simulation is
 * disabled) are never allocated because of Eigen's lazy evalulation system. */
struct RunVars {
    /* ORN-related variables. */
    struct ORN {
        /* Simulation results. */
        std::vector<Matrix> sims;

        /* Initialize matrices with the correct sizes and quantities. */
        ORN(ModelParams const&);
    } orn;

    /* LN-related variables. */
    struct LN {
        struct InhA {
            /* InhA timecourses. */
            std::vector<Vector> sims;
        } inhA;
        struct InhB {
            /* InhB timecourses. */
            std::vector<Vector> sims;
        } inhB;

        /* Initialize matrices with the correct sizes and quantities. */
        LN(ModelParams const&);
    } ln;

    /* PN-related variables. */
    struct PN {
        /* APL to bouton weights*/
        Column wAPLPN;

        /* bouton to APL weights*/
        Row wPNAPL;

        /* As the corresponding variables in rv.kc (only used if
         * p.pn.preset_w[APLPN|PNAPL]=true). */
        double wAPLPN_scale;
        double wPNAPL_scale;

        Column wAPLPN_unscaled;
        Row wPNAPL_unscaled;

        /* As rv.kc.w[APLKC|KCAPL]_units, but for boutons: shapes (#-boutons,
         * p.kc.n_apl_units) and (p.kc.n_apl_units, #-boutons). */
        SparseMatrix wAPLPN_units;
        SparseMatrix wPNAPL_units;

        std::vector<Matrix> sims;

        std::vector<Matrix> bouton_sims;

        // TODO TODO so do i not need a pn_ids (/ bouton_ids), like the kc_ids he uses
        // for that other case?
        // TODO convert type of either this (or claw_to_kc, which is currently
        // Eigen::VectorXi), to be consistent
        // TODO TODO check what happens if we try to set w/ signed values from python.
        // ideally would want setting w/ signed values to fail in python (tho python
        // should be validating all these anyway) (previously all these were
        // vector<int>, but want unsigned)
        std::vector<unsigned> Btn_to_pn;

        std::vector<std::vector<unsigned>> pn_to_Btns;
        /* Initialize matrices with the correct sizes and quantities. */
        PN(ModelParams const&);
    } pn;

    /* FFAPL-related variables. */
    struct FFAPL {
        /* Membrane voltage timecourses. */
        std::vector<Vector> vm_sims;

        /* Coef timecourses (see ModelParams::FFAPL for description). */
        std::vector<Vector> coef_sims;

        /* Initialize memory, zero-fill vm_sims. */
        FFAPL(ModelParams const&);
    } ffapl;

    /* KC-related variables. */
    struct KC {
        /* A->B connectivity matrices. */
        Matrix wPNKC;
        Column wAPLKC;
        Row    wKCAPL;

        /* wAPLKC / wKCAPL are set to these scalars times the w[APLKC|KCAPL]_unscaled
         * vectors. Tuning changes the scalars, keeping the relative values within each
         * vector (and any 0 entries) the same. If not tuning (tune_apl_weights=false),
         * these can be set before fit_sparseness, to fix preset weights at that
         * scale. */
        double wAPLKC_scale;
        double wKCAPL_scale;

        /* Normalized (or uniform) weight vectors, set at the start of fit_sparseness
         * (see `init_apl_weights` in olfsysm.cpp). */
        Column wAPLKC_unscaled;
        Row wKCAPL_unscaled;

        /* Only used if p.kc.n_apl_units > 1. Fraction of each claw's APL>claw synapses
         * from each APL unit (shape (#-claws, p.kc.n_apl_units). each row sums to 1, or
         * 0 for claws without any). wAPLKC is total weight onto each claw, so APL unit u
         * inhibits claw c with weight wAPLKC(c) * wAPLKC_units(c, u). */
        SparseMatrix wAPLKC_units;
        /* As wAPLKC_units, but fraction of each claw's claw>APL synapses onto each APL
         * unit, and transposed (shape (p.kc.n_apl_units, #-claws). columns sum to 1). */
        SparseMatrix wKCAPL_units;

        /* Only used if p.kc.n_apl_units > 1. Symmetric (and 0 on the diagonal) matrix of
         * shape (p.kc.n_apl_units, p.kc.n_apl_units), with the coupling conductance
         * between each pair of APL units, relative to the leak conductance of each
         * unit. */
        SparseMatrix apl_unit_coupling;

        /* Only used if p.kc.use_kc_kc_inh=true. KC->KC inhibitory weights, of shape
         * (p.kc.N, p.kc.N), with element (post, pre) the weight of inhibition onto KC
         * `post` from KC `pre` (including any overall scale). Must be non-negative.
         * See p.kc.use_kc_kc_inh for how it enters the KC dynamics. */
        SparseMatrix wKCKC;

        /* Spontaneous input each KC receives. Threshold typically added to this. */
        Column spont_in;

        /* Peak membrane potentials achieved on the training set before
         * applying firing thresholds. */
        Matrix pks;

        /* Firing thresholds. Of length # KCs, unless `n_claws_active_to_spike > 0`, in
         * which case it will be length # claws. */
        Column thr;

        /* Binary (KC, odor) response information. */
        Matrix responses;

        /* Like responses, but counting the number of spikes. */
        Matrix spike_counts;

        /* Membrane voltage timeseries (KCs x timesteps) for each odor. */
        std::vector<Matrix> vm_sims;

        /* Spike recordings (KCs x timesteps) for each odor. */
        std::vector<Matrix> spike_recordings;

        /* Timeseries of the vesicle depletion factor for each odor. */
        std::vector<Matrix> nves_sims;

        /* Timeseries of APL potential for each odor. Each element is a Row of length
         * #-timepoints [shape=(1, #-timepoints)]. If p.kc.n_apl_units > 1, this is the
         * mean potential across units (see apl_unit_inh_sims for each unit's). */
        std::vector<Matrix> inh_sims;

        /* Timeseries of APL potential of each unit, for each odor (each element of
         * shape (p.kc.n_apl_units, #-timepoints)). Only saved if
         * p.kc.save_apl_unit_sims=true and p.kc.n_apl_units > 1. */
        std::vector<Matrix> apl_unit_inh_sims;

        /* Timeseries of KC->APL synapse current for each odor (summed across all KCs,
         * and across APL units, if p.kc.n_apl_units > 1). Same shape as inh_sims
         * elements. */
        std::vector<Matrix> Is_sims;

        /* Timeseries of KC>APL & PN>APL "current" inputs to APL for each odor (summed
         * across APL units, if p.kc.n_apl_units > 1).
         *
         * Will only be used if we are using bouton PN<>APL weights
         * (via pn.preset_w[APLPN|PNAPL]=true and pn.n_total_boutons > 0), because
         * otherwise this would be the same as inh_sims below).
         *
         * Same shape as inh_sims elements. */
        // TODO TODO rename dIsdt_*, rather than Is_* (since that's what these are!)
        std::vector<Matrix> Is_from_kcs;
        std::vector<Matrix> Is_from_pns;

        /* Contribution each claw makes to its KC's membrane potential.
         *
         * Each KC's Vm is defined by simply adding these across all its claws, so could
         * interpret these as in units of volts.
         *
         * Only relevant if wPNKC_one_row_per_claw=true. */
        std::vector<Matrix> claw_sims;

        /* The number of iterations done during threshold tuning.
         * (only relevant for `n_spikes_for_response > 1` case) */
        unsigned thr_tuning_iters;

        /* The number of iterations done during APL tuning (each an update of the APL
         * weights, followed by a simulation to check sparsity). At least 1 if
         * tune_apl_weights=true, even if initial weights were already within
         * tolerance. */
        unsigned tuning_iters;

        /* Initialize matrices with the correct sizes and quantities. */
        KC(ModelParams const&);

        /*Vector of the KC associated with each claw*/
        // TODO TODO doc better
        // TODO change type to Matrix (/ Column/Row, whichever appropriate. matter?)?
        // TODO TODO make consistent type w/ same things for PN<>BOUTON
        // and is both this and kc_to_claws used? delete any unused
        // TODO also specify unsigned type? some other code currently assumes that, i
        // assume w/o issue?
        Eigen::VectorXi claw_to_kc;

        /*map of claws to their kc*/
        // TODO TODO doc better
        std::vector<std::vector<unsigned>> kc_to_claws;

        // TODO doc better (+ use this more consistently)
        unsigned nclaws_total;

        /* Starts false and set true if fit_sparseness terminates with sparsity in
         * tolerance, within max_iters iterations. Mainly to check so we can print
         * certain debug information only on calls after tuning has completed. */
        bool tuning_successful;

        // to access in python, to shortcut tuning on future calls
        double thr_sp_lr_coeff_to_tune_in_one_iter;
        double sp_lr_coeff_to_tune_in_one_iter;
    } kc;

    /* Logger for this run. */
    Logger log;

    /* Info from the model parameters is needed to correctly initialize matrix
     * sizes.*/
    RunVars(ModelParams const&);

    // TODO move to mp? or only for things not expected to change w/in run (or after rv
    // init?)? i do feel like i change a few of them in some cases though?
    bool verbose;

    /* Set true after model properly initialized, and set false if model put into a
     * state where future simulation calls would produce invalid output (e.g. if
     * remove_all_pretime called). */
    bool ready;
};

/* Load HC data from file. */
void load_hc_data(ModelParams& p, std::string const& fpath);

/* Choose between the above functions appropriately. */
void build_wPNKC(ModelParams const& p, RunVars& rv);

/* Set KC spike thresholds, and tune APL<->KC weights until reaching the
 * desired sparsity. */

void fit_sparseness(ModelParams const& p, RunVars& rv);

/* Model ORN response for one odor. */
void sim_ORN_layer(
        ModelParams const& p, RunVars const& rv,
        int odorid,
        Matrix& orn_t);

/* Model LN response to one odor. */
void sim_LN_layer(
        ModelParams const& p,
        Matrix const& orn_t,
        Row& inhA, Row& inhB);

/* Model PN response to one odor. */
void sim_PN_layer(
        ModelParams const& p, RunVars& rv,
        Matrix const& orn_t, Row const& inhA, Row const& inhB,
        Matrix& pn_t);

/* Model feedforward APL response to one odor. */
void sim_FFAPL_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t,
        Vector& coef_t, Vector& ffapl_t);

/* Model KC response to one odor. */
void sim_KC_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t, Vector const& ffapl_t,
        Matrix& Vm, Matrix& spikes, Matrix& nves, Row& inh, Row& Is, Matrix& claw_sims,
        Matrix& Is_from_kcs,
        Matrix& Is_from_pns,
        Matrix& bouton_sims);

/* Run ORN and LN sims for all odors. */
void run_ORN_LN_sims(ModelParams const& p, RunVars& rv);

/* Run PN sims for all odors. */
void run_PN_sims(ModelParams const& p, RunVars& rv);

/* Run feedforward APL sims for all odors. */
void run_FFAPL_sims(ModelParams const& p, RunVars& rv);

/* Regenerate PN->KC connectivity, re-tune thresholds and APL, and run KC sims
 * for all odors.
 * Connectivity regeneration can be turned off by passing regen=false. */
void run_KC_sims(ModelParams const& p, RunVars& rv, bool regen=true);

/* Frees memory for all dynamics from time.pre_start to time.start (deleting values) */
void remove_all_pretime(ModelParams const& p, RunVars& r);

#endif
