#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <random>
#include <iostream>
#include "Eigen/Dense"

/* Use to show intent. */
using Matrix = Eigen::MatrixXd;
using Row    = Matrix;
using Column = Matrix;
using Vector = Matrix;

/* Constants relevant to HC data loading. */
unsigned const N_HC_ODORS  = 110; // all original HC odors
unsigned const N_ODORS_ALL = 186; // all odors in the hc data file
unsigned const N_ODORS = N_HC_ODORS;
unsigned const N_GLOMS_ALL = 51;  // all physical gloms
unsigned const N_HC_GLOMS  = 23;  // all good HC gloms
unsigned const N_GLOMS = N_GLOMS_ALL;

/* The ID associated with each HC glom, in the order that they are listed as
 * columns in the HC data file.
 * len = N_HC_GLOMS+1 = 24. */
unsigned const HC_GLOMNUMS[] = {
    6, 16, 45, 11, 7, 19, 4,
    123456,  // UNUSED!! (8TH GLOM)
    38, 5, 44, 20, 28, 32, 21,
    14, 23, 39, 33, 22, 47, 15,
    27, 48};

/* A transform that zeros rows corresponding to non-HC gloms. */
Matrix const ZERO_NONHC_GLOMS = [](){
    Matrix ret(N_GLOMS_ALL, N_GLOMS_ALL);
    ret.setZero();
    for (unsigned i = 0; i < N_HC_GLOMS+1; i++) {
        if (i == 7) continue; // skip the 8th glom; it's bad!
        unsigned gn = HC_GLOMNUMS[i];
        ret(gn, gn) = 1.0;
    }
    return ret;
}();

/* A distribution describing the frequency with wich each HC glom should be
 * connected to when creating PN->KC connectivity matrices. */
std::discrete_distribution<int> HC_GLOM_CXN_DISTRIB {
    2.0, 24.0, 4.0, 30.0, 33.0, 8.0, 0.0,
    0.0, // no #8!
    29.0, 6.0, 2.0, 4.0, 21.0, 18.0, 4.0,
    12.0, 21.0, 10.0, 27.0, 4.0, 26.0, 7.0,
    26.0, 24.0
};

/* Contain all model parameters; never contains data generated during
 * modeling! */
extern "C" struct ModelParams {
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
    } pn;

    /* KC params. */
    struct KC {
        /* Whether to model KCs at all. */
        bool enable;

        /* The number of KCs. */
        unsigned N;

        /* The number of claws assigned to each KC. */
        unsigned nclaws;

        /* The target sparsity. */
        double sp_target;

        /* Specifies the fraction +/- of the given target that is considered an
         * acceptable sparsity. */
        double sp_acc;

        /* Time constants. */
        double taum;
        double apl_taum;
        double tau_apl2kc;
    } kc;
};

ModelParams const DEFAULT_PARAMS = []() {
    ModelParams p;

    p.time.pre_start  = -2.0;
    p.time.start      = -0.5;
    p.time.end        = 0.75;
    p.time.stim.start = 0.0;
    p.time.stim.end   = 0.5;
    p.time.dt         = 0.5e-3;

    p.orn.taum = 0.01;
    p.orn.hcdata_path = "hc_data.csv";

    p.ln.taum   = 0.01;
    p.ln.tauGA  = 0.1;
    p.ln.tauGB  = 0.4;
    p.ln.thr    = 1.0;
    p.ln.inhsc  = 500.0;
    p.ln.inhadd = 200.0;

    p.pn.taum   = 0.01;
    p.pn.offset = 2.9410;
    p.pn.tanhsc = 5.3395;
    p.pn.inhsc  = 368.6631;
    p.pn.inhadd = 31.4088;

    p.kc.enable     = true;
    p.kc.N          = 2000;
    p.kc.nclaws     = 6;
    p.kc.sp_target  = 0.1;
    p.kc.sp_acc     = 0.1;
    p.kc.taum       = 0.01;
    p.kc.apl_taum   = 0.05;
    p.kc.tau_apl2kc = 0.01;

    return p;
}();


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

        /* Initialize matrices with the correct sizes and quantities. */
        KC(ModelParams const&);
    } kc;

    /* Info from the model parameters is needed to correctly initialize matrix
     * sizes.*/
    RunVars(ModelParams const&);
};

template<class M>
void dump(M val, std::string const& name) {
    std::ofstream fout(name);
    for (int i = 0; i < val.rows(); i++) {
        for (int k = 0; k < val.cols()-1; k++) {
            fout << val(i, k) << ',';
        }
        fout << val(i, val.cols()-1) << std::endl;
    }
    fout.close();
}

/* (utility) Split a string by commas, and fill vec with the segments.
 * vec must be sized correctly! */
void split_regular_csv(std::string const& str, std::vector<std::string>& vec);

/* Load HC data from file. */
void load_hc_data(ModelParams const& p, RunVars& rv);

/* The exponential ('e') part of the smoothts MATLAB function included in the
 * Kennedy source.
 * Instead of returning the smoothed matrix, it smooths it in-place. */
void smoothts_exp(Matrix& vin, double wsize);

/* Randomly generate the wPNKC connectivity matrix. Glom choice is WEIGHTED by
 * HC_GLOM_CXN_DISTRIB (above). */
void build_wPNKC(ModelParams const& p, RunVars& rv);

/* Sample spontaneous PN output from odor 0. */
Column sample_PN_spont(ModelParams const& p, RunVars const& rv);

/* Decide a KC threshold column from KC membrane voltage data. */
Column choose_KC_thresh(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in);

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

/*******************************************************************************
********************************************************************************
*********************                                      *********************
*********************            IMPLEMENTATIONS           *********************
*********************                                      *********************
********************************************************************************
*******************************************************************************/
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

RunVars::RunVars(ModelParams const& p) : orn(p), ln(p), pn(p), kc(p) {
}
RunVars::ORN::ORN(ModelParams const& p) :
    rates(N_GLOMS, N_ODORS),
    spont(N_GLOMS, 1),
    delta(N_GLOMS, N_ODORS),
    sims(N_ODORS, Matrix(N_GLOMS, p.time.steps_all())) {
}
RunVars::LN::LN(ModelParams const& p) :
    inhA{std::vector<Vector>(N_HC_ODORS, Row(1, p.time.steps_all()))},
    inhB{std::vector<Vector>(N_HC_ODORS, Row(1, p.time.steps_all()))} {
}
RunVars::PN::PN(ModelParams const& p) :
    sims(N_ODORS, Matrix(N_GLOMS, p.time.steps_all())) {
}
RunVars::KC::KC(ModelParams const& p) :
    wPNKC(p.kc.N, N_GLOMS),
    wAPLKC(p.kc.N, 1),
    wKCAPL(1, p.kc.N),
    thr(p.kc.N, 1),
    responses(p.kc.N, N_ODORS) {
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

void load_hc_data(ModelParams const& p, RunVars& run) {
    run.orn.rates.setZero();
    run.orn.spont.setZero();

    std::ifstream fin(p.orn.hcdata_path);
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
        split_regular_csv(line, segs);
        for (unsigned glom = 0; glom < N_HC_GLOMS+1; glom++) {
            /* Ignore the 8th glom column (Kennedy does this). */
            if (glom == 7) continue;

            /* At this point we're actually storing deltas. */
            run.orn.rates(HC_GLOMNUMS[glom], odor) = std::stod(segs[glom+2]);
            run.orn.delta(HC_GLOMNUMS[glom], odor) = std::stod(segs[glom+2]);
        }
    }

    /* Load the spontaneous rates line. */
    std::getline(fin, line);
    split_regular_csv(line, segs);
    for (unsigned glom = 0; glom < N_HC_GLOMS+1; glom++) {
        if (glom == 7) continue;
        run.orn.spont(HC_GLOMNUMS[glom]) = std::stod(segs[glom+2]);
    }

    /* Convert deltas into absolute rates. */
    run.orn.rates.colwise() += run.orn.spont.col(0);
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

void build_wPNKC(ModelParams const& p, RunVars& r) {
    std::random_device randdev;
    std::default_random_engine randgen{randdev()};

    /* Draw PN connections from realistic connection distribution data (see
     * above). */
    r.kc.wPNKC.setZero();
    for (unsigned kc = 0; kc < p.kc.N; kc++) {
        for (unsigned claw = 0; claw < p.kc.nclaws; claw++) {
            r.kc.wPNKC(kc, HC_GLOMNUMS[HC_GLOM_CXN_DISTRIB(randgen)]) += 1.0;
        }
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
    return rv.pn.sims[0].block(0,sp_t1,N_GLOMS,sp_t2-sp_t1).rowwise().mean();
}
Column choose_KC_thresh(
        ModelParams const& p, Matrix& KCpks, Column const& spont_in) {
    KCpks.resize(1, KCpks.size());                     // flatten
    std::sort(KCpks.data(), KCpks.data()+KCpks.size(),
            [](double a, double b){return a>b;});      // dec. order
    double thr_const = KCpks(std::min(
                int(p.kc.sp_target*2.0*double(p.kc.N*N_ODORS)),
                int(p.kc.N*N_ODORS)-1));
    return thr_const + spont_in.array()*2.0;

}
void fit_sparseness(ModelParams const& p, RunVars& rv) {
    /* Set starting values for the things we'll tune. */
    rv.kc.wAPLKC.setZero();
    rv.kc.wKCAPL.setConstant(1.0/float(p.kc.N));
    rv.kc.thr.setConstant(1e5); // higher than will ever be reached (for now)

    /* Calculate spontaneous input to KCs. */
    Column spont_in = rv.kc.wPNKC * sample_PN_spont(p, rv);

    /* Used for measuring KC voltage; defined here to make it shared across all
     * threads.*/
    Matrix KCpks(p.kc.N, N_ODORS); KCpks.setZero();

    /* Used to store odor response data during APL tuning. */
    Matrix KCmean_st(p.kc.N, 1+((N_ODORS-1)/3));
    /* Used to store the current sparsity.
     * Initially set to the below value because, given default model
     * parameters, it causes tuning to complete in just one iteration. */
    double sp = 0.0789;
    /* Used to count number of times looped; the 'learning rate' is decreased
     * as 1/sqrt(count) with each iteration. */
    double count = 1.0;

    /* Break up into threads. */
#pragma omp parallel
    { 
        /* Output matrices for the KC simulation. */
        Matrix Vm(p.kc.N, p.time.steps_all());
        Matrix spikes(p.kc.N, p.time.steps_all());

        /* Measure voltages achieved by the KCs, and choose a threshold based
         * on that. */
#pragma omp for
        for (unsigned i = 0; i < N_ODORS; i++) {
            sim_KC_layer(p, rv, rv.pn.sims[i], Vm, spikes);
            KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*2.0;
        }
#pragma omp barrier
#pragma omp master
        {
            /* Finish picking thresholds. */
            rv.kc.thr = choose_KC_thresh(p, KCpks, spont_in);

            /* Starting values for to-be-tuned APL<->KC weights. */
            rv.kc.wAPLKC.setConstant(
                    2*ceil(-log(p.kc.sp_target)));
            rv.kc.wKCAPL.setConstant(
                    2*ceil(-log(p.kc.sp_target))/double(p.kc.N));
        }
#pragma omp barrier

        /* Continue tuning until we reach the desired sparsity. */
        do {
#pragma omp master
            {
                /* Modify the APL<->KC weights in order to move in the
                 * direction of the target sparsity. */
                double lr = 10.0/sqrt(count);
                double delta = (sp-p.kc.sp_target)*lr/p.kc.sp_target;
                rv.kc.wAPLKC.array() += delta;
                rv.kc.wKCAPL.array() += delta/double(p.kc.N);

                count += 1.0;
            }
#pragma omp barrier

            /* Run through a bunch of odors to test sparsity. */
#pragma omp for
            for (unsigned i = 0; i < N_ODORS; i+=3) {
                sim_KC_layer(p, rv, rv.pn.sims[i], Vm, spikes);
                KCmean_st.col(i/3) = spikes.rowwise().sum();
            }
#pragma omp barrier
#pragma omp master
            {
                KCmean_st = (KCmean_st.array() > 0.0).select(1.0, KCmean_st);
                sp = KCmean_st.mean();
                std::cout << sp << std::endl;
            }
#pragma omp barrier
        } while (abs(sp-p.kc.sp_target)>(p.kc.sp_acc*p.kc.sp_target));
    }
    std::cout << count << std::endl;
}

void sim_ORN_layer(
        ModelParams const& p, RunVars const& rv,
        int odorid,
        Matrix& orn_t) {
    /* Initialize with spontaneous activity. */
    orn_t = rv.orn.spont*p.time.row_all();

    /* "Odor input to ORNs" (Kennedy comment)
     * Smoothed timeseries of spont...odor rate...spont */
    Matrix odor = orn_t + rv.orn.delta.col(odorid)*p.time.stim.row_all();
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
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        dinhAdt = -inhA(t-1) + response(t-1);
        dinhBdt = -inhB(t-1) + response(t-1);
        dLNdt =
            -potential(t-1)
            +pow(orn_t.col(t-1).mean(), 3.0)*51.0/23.0/2.0*inh_LN;
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
    Column spont  = rv.orn.spont*p.pn.inhsc/(rv.orn.spont.sum()+p.pn.inhadd);
    pn_t          = rv.orn.spont*p.time.row_all();
    double inh_PN = 0.0;

    Column orn_delta;
    Column dPNdt;
    for (unsigned t = 1; t < p.time.steps_all(); t++) {
        orn_delta = orn_t.col(t-1)-rv.orn.spont;
        dPNdt = -pn_t.col(t-1) + spont;
        dPNdt += 
            200.0*((orn_delta.array()+p.pn.offset)*p.pn.tanhsc/200.0*inh_PN).matrix().unaryExpr<double(*)(double)>(&tanh);
        inh_PN = p.pn.inhsc/(p.pn.inhadd+0.25*inhA(t)+0.75*inhB(t));
        pn_t.col(t) = pn_t.col(t-1) + dPNdt*p.time.dt/p.pn.taum;
        pn_t.col(t) = (0.0 < pn_t.col(t).array()).select(pn_t.col(t), 0.0);
    }

    /* Zero non-HC gloms (they are just noise, not from odor...) */
    pn_t = ZERO_NONHC_GLOMS * pn_t;
}
void sim_KC_layer(
        ModelParams const& p, RunVars const& rv,
        Matrix const& pn_t,
        Matrix& Vm, Matrix& spikes) { 
    Vm.setZero();
    spikes.setZero();
    Row inh(1, p.time.steps_all()); inh.setZero();
    Row Is(1, p.time.steps_all());  Is.setZero();

    Column dKCdt;
    for (unsigned t = p.time.start_step()+1; t < p.time.steps_all(); t++) {
        double dIsdt = -Is(t-1) + (rv.kc.wKCAPL*spikes.col(t-1))(0,0)*1e4;
        double dinhdt = -inh(t-1) + Is(t-1);

        dKCdt = 
            -Vm.col(t-1)
            +rv.kc.wPNKC*pn_t.col(t)
            -rv.kc.wAPLKC*inh(t-1);
        Vm.col(t) = Vm.col(t-1) + dKCdt*p.time.dt/p.kc.taum;
        inh(t)    = inh(t-1)    + dinhdt*p.time.dt/p.kc.apl_taum;
        Is(t)     = Is(t-1)     + dIsdt*p.time.dt/p.kc.tau_apl2kc;

        auto const thr_comp = Vm.col(t).array() > rv.kc.thr.array();
        spikes.col(t) = thr_comp.select(1.0, spikes.col(t)); // either go to 1 or _stay_ at 0.
        Vm.col(t) = thr_comp.select(0.0, Vm.col(t)); // very abrupt repolarization!
    }
}

void run_ORN_LN_sims(ModelParams const& p, RunVars& rv) {
#pragma omp parallel for
    for (unsigned i = 0; i < N_ODORS; i++) {
        sim_ORN_layer(p, rv, i, rv.orn.sims[i]);
        sim_LN_layer(
                p, rv.orn.sims[i],
                rv.ln.inhA.sims[i], rv.ln.inhB.sims[i]);
    }
}
void run_PN_sims(ModelParams const& p, RunVars& rv) {
#pragma omp parallel for
    for (unsigned i = 0; i < N_ODORS; i++) {
        sim_PN_layer(
                p, rv,
                rv.orn.sims[i], rv.ln.inhA.sims[i], rv.ln.inhB.sims[i],
                rv.pn.sims[i]);
    }
}
void run_KC_sims(ModelParams const& p, RunVars& rv, bool regen) {
    if (regen) {
        build_wPNKC(p, rv);
        fit_sparseness(p, rv);
    }

#pragma omp parallel
    {
        Matrix Vm(p.kc.N, p.time.steps_all());
        Matrix spikes(p.kc.N, p.time.steps_all());
        Matrix respcol;
#pragma omp for
        for (unsigned i = 0; i < N_ODORS; i++) {
            sim_KC_layer(
                    p, rv,
                    rv.pn.sims[i],
                    Vm, spikes);
            respcol = spikes.rowwise().sum();
            respcol = (respcol.array() > 0.0).select(1.0, respcol);
            rv.kc.responses.col(i) = respcol;
        }
    }
}

/*
extern "C" double *model_kc_responses_raw(int nclaws, int& n_kcs, int& n_odors) {
    Matrix result = model_kc_responses(nclaws);
    n_kcs = result.rows();
    n_odors = result.cols();
    double *raw = new double[n_kcs*n_odors];
#pragma omp parallel for
    for (int r = 0; r < n_kcs; r++) {
        for (int c = 0; c < n_odors; c++) {
            raw[(r*n_odors)+c] = result(r, c);
        }
    }
    return raw;
}
*/

void dump(Matrix const& m, std::string const& p) {
    std::ofstream fout(p);
    for (unsigned i = 0; i < m.rows(); i++) {
        for (unsigned j = 0; j < m.cols()-1; j++) {
            fout << m(i,j) << ',';
        }
        fout << m(i,m.cols()-1) << std::endl;
    }
    fout.close();
}

int main() {
    ModelParams mp = DEFAULT_PARAMS;
    RunVars rv(mp);
    load_hc_data(mp, rv);
    run_ORN_LN_sims(mp, rv);
    run_PN_sims(mp, rv);

    run_KC_sims(mp, rv);

    // dump(rv.kc.wPNKC, "o/wPNKC.csv");
    // dump(rv.kc.wKCAPL, "o/wKCAPL.csv");
    // dump(rv.kc.wAPLKC, "o/wAPLKC.csv");
    // dump(rv.kc.responses, "o/responses.csv");
    // dump(rv.kc.thr, "o/thr.csv");
    // dump(rv.orn.sims[0], "o/ornt1.csv");
    // dump(rv.ln.inhA.sims[0], "o/inhA1.csv");
    // dump(rv.ln.inhB.sims[0], "o/inhB1.csv");
    // dump(rv.pn.sims[0], "o/pnt1.csv");

    return 0;
}

