#ifndef MODEL_H_
#define MODEL_H_

#include <vector>
#include <string>
#include "Eigen/Dense"

/* Constants relevant to HC data loading. */
unsigned const N_HC_ODORS  = 110; // all original HC odors
unsigned const N_ODORS_ALL = 186; // all odors in the hc data file
unsigned const N_ODORS = N_HC_ODORS;
unsigned const N_GLOMS_ALL = 51;  // all physical gloms
unsigned const N_HC_GLOMS  = 23;  // all good HC gloms
unsigned const N_GLOMS = N_GLOMS_ALL;

/* Use to show intent. */
using Matrix = Eigen::MatrixXd;
using Row    = Matrix;
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

        /* Path to the HC data file. */
        std::string hcdata_path;
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
    } pn;

    /* KC params. */
    struct KC {
        /* The number of KCs. */
        unsigned N;

        /* The number of claws assigned to each KC. */
        unsigned nclaws;
        /* Whether to use uniform PN choice, or use observational data. */
        bool uniform_pns;

        /* Whether to simulate the APL at all.*/
        bool enable_apl;

        /* Optionally set a fixed KC firing threshold, instead of using the
         * normally generated thresholds. */
        double fixed_thr;
        bool use_fixed_thr;

        /* The target sparsity. */
        double sp_target;

        /* Specifies the fraction +/- of the given target that is considered an
         * acceptable sparsity. */
        double sp_acc;

        /* Changes the scaling of the ~1/(n^2) tuning step-size curve. */
        double sp_lr_coeff;

        /* The maximum number of tuning iterations that should be done before
         * aborting. Must be >=1. */
        unsigned max_iters;

        /* Time constants. */
        double taum;
        double apl_taum;
        double tau_apl2kc;
    } kc;
};
extern ModelParams const DEFAULT_PARAMS;

/* Variables and storage space that is useful to each run.
 * Matrices that are not used (e.g., KC-related matrices when KC simulation is
 * disabled) are never allocated because of Eigen's lazy evalulation system. */
struct RunVars {
    /* ORN-related variables. */
    struct ORN {
        /* Loaded ORN firing rates (spont+delta). */
        Matrix rates;
        /* Spontaneous rates. */
        Column spont;
        /* Firing rate changes in response to odors. */
        Column delta;

        /* Simulation results. */
        std::vector<Matrix> sims;

        /* Initialize matrices with the correct sizes and quantities. */
        ORN(ModelParams const&);
    } orn;

    /* LN-related variables. */
    struct LN {
        struct {
            /* InhA timecourses. */
            std::vector<Vector> sims;
        } inhA;
        struct {
            /* InhB timecourses. */
            std::vector<Vector> sims;
        } inhB;

        /* Initialize matrices with the correct sizes and quantities. */
        LN(ModelParams const&);
    } ln;

    /* PN-related variables. */
    struct PN {
        std::vector<Matrix> sims;

        /* Initialize matrices with the correct sizes and quantities. */
        PN(ModelParams const&);
    } pn;

    /* KC-related variables. */
    struct KC {
        /* A->B connectivity matrices. */
        Matrix wPNKC;
        Column wAPLKC;
        Row    wKCAPL;

        /* Firing thresholds. */
        Column thr;

        /* Binary (KC, odor) response information. */
        Matrix responses;

        /* The number of iterations done during APL tuning. */
        unsigned tuning_iters;

        /* Initialize matrices with the correct sizes and quantities. */
        KC(ModelParams const&);
    } kc;

    /* Info from the model parameters is needed to correctly initialize matrix
     * sizes.*/
    RunVars(ModelParams const&);
};

/* Load HC data from file. */
void load_hc_data(ModelParams const& p, RunVars& rv);

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
        ModelParams const& p, RunVars const& rv,
        Matrix const& orn_t, Row const& inhA, Row const& inhB, 
        Matrix& pn_t);

/* Model KC response to one odor. */
void sim_KC_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t,
        Matrix& Vm, Matrix& spikes);

/* Run ORN and LN sims for all odors. */
void run_ORN_LN_sims(ModelParams const& p, RunVars& rv);

/* Run PN sims for all odors. */
void run_PN_sims(ModelParams const& p, RunVars& rv);

/* Regenerate PN->KC connectivity, re-tune thresholds and APL, and run KC sims
 * for all odors.
 * Connectivity regeneration can be turned off by passing regen=false. */
void run_KC_sims(ModelParams const& p, RunVars& rv, bool regen=true);

#endif
