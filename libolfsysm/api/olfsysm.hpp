#ifndef OLFSYSM_H_
#define OLFSYSM_H_

#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <fstream>
#include "Eigen/Dense"

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

    /* Also write to stdout, in addition to file from previous redirect call. */
    void tee();

    /* Shut off output. */
    void disable();
};

/* Use to show intent. */
using Matrix = Eigen::MatrixXd;
using Row    = Matrix;
// TODO change to vectors? why both matrices? i assume neither is enforced to actually
// have shape 1 in right place anywhere? how could i enforce that?
using Column = Matrix;
using Vector = Matrix;

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

        // TODO TODO TODO add parameter to control strength of APL<>PN stuff relative to
        // strength of KC<>APL stuff (or just handle via scale factor between these
        // weights in python? at least make sure python tries a sweep that includes
        // varying PN>APL separately from APL>PN?)
        bool preset_wAPLPN;
        bool preset_wPNAPL;

        // TODO delete? (currently unused, but may want to try disabling PN<>APL stuff
        // in tuning?)
        //bool pn_apl_tune;

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

        // TODO TODO doc how these will interact w/ tune_apl_weights
        /* Set to true if using a pre-loaded APL->KC weight vector. */
        bool preset_wAPLKC;
        /* Set to true if using a pre-loaded KC->APL weight vector. */
        bool preset_wKCAPL;

        /* If false, APL activity will depend on KC spiking, and then a KC spiking will
         * cause all of its claws to provide input to the APL (just multiplied by
         * wKCAPL).
         *
         * If true, APL activity will NOT depend on KC spiking, and APL will get input
         * (which can be subthreshold) directly from each claw (though any APL
         * inhibition [-> rectification] to each claw will be applied first).
         *
         * Only relevant if wPNKC_one_row_per_claw=true. */
        bool pn_claw_to_APL;

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
         * Not yet implemented for `apl_coup_const != -1` case.
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

        /* (sp_target * sp_factor_pre_APL) is sparsity (>sp_target) achieved by setting
         * KC spike thresholds alone, then sparsity is brought down to sp_target by
         * tuning APL. Must be >1, but small enough that (sp_target *
         * sp_factor_pre_APL) <= 1.0 */
        double sp_factor_pre_APL;

        /* Specifies the fraction +/- of the given target that is considered an
         * acceptable sparsity. */
        double sp_acc;

        /* Changes the scaling of the ~1/(n^2) tuning step-size curve. */
        double sp_lr_coeff;

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
        double apl_Cm;
        double apl_taum;
        double tau_apl2kc;

        /* APL compartment coupling constants*/
        double apl_coup_const;
        int comp_num;

        /* Synaptic depression params; see Hennig 2013 equation 3. Synaptic
         * depression can be disabled by setting ves_p = 0. */
        double tau_r;
        double ves_p;

        /* Output options. */
        bool save_vm_sims;
        bool save_spike_recordings;
        bool save_nves_sims;
        bool save_inh_sims;
        bool save_Is_sims;
        bool save_claw_sims;

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

        double wAPLPN_scale;
        double wPNAPL_scale;

        Column wAPLPN_unscaled;
        Row wPNAPL_unscaled;

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

        /* Only used if respective flag preset_w[APLKC|KCAPL] is true, where then these
         * scalars are tuned rather than wAPLKC/wKCAPL themselves.
         *
         * In these preset cases, wAPLKC/wKCAPL are vectors, whose relative
         * relationships we don't want to accidentally change by clipping some entries
         * to 0.
         *
         * Also want to keep 0 entries as 0 at output of scaling, rather than adding
         * constant across all elements of these vectors. */
        double wAPLKC_scale;
        double wKCAPL_scale;

        Column wAPLKC_unscaled;
        Row wKCAPL_unscaled;

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

        /* Timeseries of APL potential for each odor. */
        std::vector<Row> inh_sims;

        // TODO clarify in doc that it is always just a single number (sum? mean?)
        // across all KCs (it is, right?)
        /* Timeseries of KC->APL synapse current for each odor. */
        std::vector<Row> Is_sims;

        /* Contribution each claw makes to its KC's membrane potential.
         *
         * Each KC's Vm is defined by simply adding these across all its claws, so could
         * interpret these as in units of volts.
         *
         * Only relevant if wPNKC_one_row_per_claw=true. */
        std::vector<Matrix> claw_sims;

        /* The number of iterations done during APL tuning. */
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

        /*Vector of the compartment associated with each claw*/
        // TODO doc better
        // TODO change type to Matrix (/ Column/Row, whichever appropriate. matter?)?
        Eigen::VectorXi claw_compartments;

        /*map of claws to their compartment */
        // TODO doc better
        std::vector<std::vector<unsigned>> compartment_to_claws;

        // TODO doc better (+ use this more consistently)
        unsigned nclaws_total;

        // TODO delete?
        // for debugging weight scaling
        // TODO TODO also add one for claw>apl (no spiking required)? or just use kc
        // ones for that too? (latter, probably)
        std::vector<Eigen::VectorXd> odor_stats;
        // TODO delete. couldn't get to work (read only compile error when trying to set
        // based on index in sim_KC_layer, adding odor_index param [passing loop vars to
        // each call])
        //std::vector<double> max_kc_apl_drive;
        //std::vector<double> avg_kc_apl_drive;
        //std::vector<double> max_bouton_apl_drive;
        //std::vector<double> avg_bouton_apl_drive;
    } kc;

    /* Logger for this run. */
    Logger log;

    /* Info from the model parameters is needed to correctly initialize matrix
     * sizes.*/
    RunVars(ModelParams const&);
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
        // TODO TODO replace odor_index w/ passing in reference to a vector to put all
        // odor stats into? (seems we can't set std::vector elements by index b/c read
        // only compile error. not sure 100% why...)
        //Matrix& bouton_sims, unsigned odor_index);
        Matrix& bouton_sims, Eigen::VectorXd& odor_stats);

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
