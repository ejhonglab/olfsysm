#include <math.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>
#include "Eigen/Dense"

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

using Matrix = Eigen::MatrixXd;
using Row = Matrix;
using Column = Matrix;
using Vector = Matrix;

int const N_ODORS = 186;
int const N_HC_ODORS = 110;
int const N_GLOMS = 51;
int const N_KCS = 2000;

int const HC_GLOMNUMS[] = {
    6, 16, 45, 11, 7, 19, 4,
    -123456,  // UNUSED!! (8TH GLOM)
    38, 5, 44, 20, 28, 32, 21,
    14, 23, 39, 33, 22, 47, 15,
    27, 48};
int const HC_NGLOMS = 23;
Matrix const ZERO_NONHC_GLOMS = [](){
    Matrix ret(N_GLOMS, N_GLOMS);
    ret.setZero();
    for (int i = 0; i < HC_NGLOMS+1; i++) {
        if (i == 7) continue;
        int gn = HC_GLOMNUMS[i];
        ret(gn, gn) = 1.0;
    }
    return ret;
}();
std::discrete_distribution<int> HC_GLOM_CXN_DISTRIB = {
    2, 24, 4, 30, 33, 8, 0,
    0, // no #8!
    29, 6, 2, 4, 21, 18, 4,
    12, 21, 10, 27, 4, 26, 7,
    26, 24
};


double const PRETIME_START = -2;
double const TIME_START = -0.5;
double const TIME_END = 0.75;
double const STIM_START = 0.0;
double const STIM_END = 0.5;
double const DT = 0.5e-3;
int const N_TIMESTEPS = int((TIME_END-PRETIME_START)/DT);
int const TIME_START_STEP = int((TIME_START-PRETIME_START)/DT);
int const STIM_START_STEP = int((STIM_START-PRETIME_START)/DT);
int const STIM_END_STEP = int((STIM_END-PRETIME_START)/DT);

int const KC_TBIN_SIZE = 10;
int const N_KC_TBINS = 1+(N_TIMESTEPS/KC_TBIN_SIZE);


Row const TIME = [](){
    Row ret(1, N_TIMESTEPS);
    ret.setOnes();
    return ret;
}();
Row const STIM = [](){
    Row ret(1, N_TIMESTEPS);
    ret.setZero();
    ret.block(0, STIM_START_STEP, 1, STIM_END_STEP-STIM_START_STEP).setOnes();
    return ret;
}();

struct ORNParams {
    double taum;                    // membrane time constant?
    Matrix rates;  // firing rates
    Column spont; // column vec of spontaneous firing rates

    ORNParams() :
        taum(0.01),
        rates(N_GLOMS, N_ODORS),
        spont(N_GLOMS, 1)
    {}
};

struct Run {
    Matrix orn_t;
    Row    inhA, inhB;
    Matrix pn_t;
    Matrix kc_spikes;

    Run() :
        orn_t(N_GLOMS, N_TIMESTEPS),
        inhA(1, N_TIMESTEPS),
        inhB(1, N_TIMESTEPS),
        pn_t(N_GLOMS, N_TIMESTEPS),
        kc_spikes(N_KCS, N_TIMESTEPS)
    {}
};

struct LNParams {
    double taum;
    double tauGA;
    double tauGB;
    double thr;
    double inhsc;
    double inhadd;

    LNParams() :
        taum(0.01),
        tauGA(0.1),
        tauGB(0.4),
        thr(1.0),
        inhsc(500.0),
        inhadd(200.0)
    {}
};

struct PNParams {
    double offset;
    double tanhsc;
    double inhsc;
    double inhadd;
    double taum;

    PNParams() :
        offset(2.9410),
        tanhsc(5.3395),
        inhsc(368.6631),
        inhadd(31.4088),
        taum(0.01)
    {}
};

struct KCParams {
    double taum;       // KC time constant
    double apl_taum;   // APL (inh) taum
    double tau_apl2kc; // synaptic time constant
    Column thr;        // KC firing thresholds
    Matrix wPNKC;      // PN->KC weights
    Column wAPLKC;     // APL->KC weights
    Row    wKCAPL;     // KC->APL weights

    KCParams() :
        taum(0.01),
        apl_taum(0.05),
        tau_apl2kc(0.01),
        thr(N_KCS, 1),
        wPNKC(N_KCS, N_GLOMS),
        wAPLKC(N_KCS, 1),
        wKCAPL(1, N_KCS)
    {}
};

class LoadBar {
    bool done;
    std::string msg;

public:
    LoadBar(std::string const& message) : done(true) {
#if 0
        reset(message);
#endif
    }
    void reset(std::string const& message) {
#if 0
        finish();
        done = false;

        msg = message;
        while (msg.size() < 43) {
            msg += " ";
        }
        std::cout << msg << "[     ]\r" << std::flush;
#endif
    };

    void operator()(int step, int total) {
#if 0
        std::cout << msg << "[";
        int nticks = (5*step)/total; // steps of 20%
        for (int i = 0; i < nticks; i++)   std::cout << '#';
        for (int i = 0; i < 5-nticks; i++) std::cout << ' ';
        std::cout << "]\r" << std::flush;
#endif
    }
    void finish() {
#if 0
        if (done) return;
        std::cout << msg << "[#####]\n";
        done = true;
#endif
    }
    ~LoadBar() {
#if 0
        finish();
#endif
    }
};

/* (utility) Split a string by commas, and fill vec with the segments.
 * vec must be sized correctly! */
void split_regular_csv(std::string const& str, std::vector<std::string>& vec);

/* Load the csv at fpath into data. (All values should be numbers). */ 
void load_csv(std::string const& fpath, Matrix& data);

/* Load HC data from file. */
void load_hc_data(ORNParams& ornp);

/* The exponential ('e') part of the smoothts MATLAB function included in the
 * Kennedy source.
 * Instead of returning the smoothed matrix, it smooths it in-place. */
template<class Mtx>
void smoothts_exp(Mtx& vin, double wsize) {
    double extarg = wsize;
    if (wsize > 1.0) {
        extarg = 2.0/(wsize+1.0);
    }
    for (int i = 1; i < vin.cols(); i++) {
        vin.col(i) = extarg*vin.col(i) + (1-extarg)*vin.col(i-1);
    }
}

void build_wPNKC(KCParams& kcp, int nclaws);
void fit_sparseness(std::vector<Run> const& runs, KCParams& kcp, double target);
void tune_KC_thresholds(std::vector<Run> const& runs, KCParams& kcp, double target);
void tune_APL(std::vector<Run> const& runs, KCParams& kcp, double target);

void sim_ORN_layer(
        int odor,              // in
        ORNParams const& ornp, // in
        Matrix& orn_t);        // out

void sim_LN_layer(
        LNParams const& lnp, // in
        Matrix const& orn_t, // in
        Row& inhA,           // out
        Row& inhB);          // out

void sim_PN_layer(
        PNParams const& pnp, // in
        Matrix const& orn_t, // in
        Matrix const& inhA,  // in
        Matrix const& inhB,  // in
        Matrix pn_t);        // out

void sim_KC_layer(
        KCParams const& kcp, // in
        Matrix const& pn_t,  // in
        Matrix& Vm,          // out
        Matrix& spikes);     // out

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

void load_csv(std::string const& fpath, Matrix& data) {
    std::ifstream fin(fpath);
    std::vector<std::string> splits(data.cols());
    std::string line;
    for (int row = 0; row < data.rows(); row++) {
        std::getline(fin, line);
        split_regular_csv(line, splits);
        for (int col = 0; col < data.cols(); col++) {
            data(row, col) = std::stod(splits[col]);
        }
    }
    fin.close();
}

void load_hc_data(ORNParams& ornp) {
    ornp.rates.setZero();
    ornp.spont.setZero();

    std::ifstream fin("hc_data.csv");
    std::string line;
    std::vector<std::string> segs(HC_NGLOMS+2+1); // there are 2 ID columns
    // discard the first two (header) lines
    std::getline(fin, line);
    std::getline(fin, line);
    for (int odor = 0; odor < N_ODORS; odor++) {
        std::getline(fin, line);
        split_regular_csv(line, segs);
        for (int glom = 0; glom < HC_NGLOMS+1; glom++) {
            if (glom == 7) continue; // skip 8th glom??
            ornp.rates(HC_GLOMNUMS[glom], odor) = std::stod(segs[glom+2]);
        }
    }
    // last line is spontaneous
    std::getline(fin, line);
    split_regular_csv(line, segs);
    for (int glom = 0; glom < HC_NGLOMS+1; glom++) {
        if (glom == 7) continue; // skip 8th glom???
        ornp.spont(HC_GLOMNUMS[glom], 0) = std::stod(segs[glom+2]);
    }
}

void build_wPNKC(KCParams& kcp, int nclaws) {
    std::default_random_engine randgen;

    // draw from glom distribution
    kcp.wPNKC.setZero();
    Eigen::VectorXf choices(nclaws);
    for (int kc = 0; kc < N_KCS; kc++) {
        for (int claw = 0; claw < nclaws; claw++) {
            kcp.wPNKC(kc, HC_GLOMNUMS[HC_GLOM_CXN_DISTRIB(randgen)]) += 1.0;
        }
    }
}

void fit_sparseness(std::vector<Run> const& runs, KCParams& kcp, double target) {
    tune_KC_thresholds(runs, kcp, target);
    tune_APL(runs, kcp, target);
}

void tune_KC_thresholds(std::vector<Run> const& runs, KCParams& kcp, double target) {
    LoadBar lb("Tuning KC thresholds");

    // calculate spont. input to KCs
    int sp_t1 = TIME_START_STEP + int((STIM_START-TIME_START)/(2*DT));
    int sp_t2 = TIME_START_STEP + int((STIM_START-TIME_START)/DT);
    Column pn_spont = runs[0].pn_t.block(0,sp_t1,N_GLOMS,sp_t2-sp_t1).rowwise().mean();
    Column spont_in = kcp.wPNKC*pn_spont;

    // set default (untuned) weights
    kcp.wAPLKC.setZero();
    kcp.wKCAPL.setConstant(1.0/float(N_KCS));
    kcp.thr.setConstant(1e5); // something that will never be reached while we just test Vm

    // our own recording
    Matrix KCpks(N_KCS, runs.size()); KCpks.setZero();

    // set KC thresholds
#pragma omp parallel
    {
        Matrix Vm(N_KCS, N_TIMESTEPS);
        Matrix spikes(N_KCS, N_TIMESTEPS);
#pragma omp for
        for (int i = 0; i < runs.size(); i++) {
            sim_KC_layer(kcp, runs[i].pn_t, Vm, spikes);
            KCpks.col(i) = Vm.rowwise().maxCoeff() - spont_in*2.0;
            lb(i+1, runs.size());
        }
    }
    KCpks.resize(1, KCpks.size()); // flatten
    std::sort(KCpks.data(), KCpks.data()+KCpks.size(),
            [](double a, double b){return a>b;}); // sort in dec. oder
    double thr_const = KCpks(std::min(
                int(target*2.0*double(N_KCS)*double(runs.size())),
                int(N_KCS*runs.size())-1));
    kcp.thr = (thr_const + spont_in.array()*2.0).matrix();
}

void tune_APL(std::vector<Run> const& runs, KCParams& kcp, double target) {
    // tune APL<->KC weights.
    LoadBar lb("Tuning APL<->KC weights");

    kcp.wAPLKC.setConstant(2*ceil(-log(target)));
    kcp.wKCAPL.setConstant(2*ceil(-log(target))/N_KCS);

    double sp = 0.0;
    int count = 0;
    Matrix KCmean_st(N_KCS, 1+((runs.size()-1)/3));
    while (abs(sp-target)>(0.1*target) && count++<50) {
        KCmean_st.setZero();
#pragma omp parallel
        {
        Matrix Vm(N_KCS, N_TIMESTEPS);
        Matrix spikes(N_KCS, N_TIMESTEPS);
#pragma omp for
            for (int i = 0; i < runs.size(); i+=3) {
                sim_KC_layer(kcp, runs[i].pn_t, Vm, spikes);
                KCmean_st.col(i/3) = spikes.rowwise().sum();
                lb(i, runs.size());
            }
        }
        KCmean_st = (KCmean_st.array() > 0.0).select(1.0, KCmean_st);
        sp = KCmean_st.mean();
        double learning_rate = 7.0/sqrt(double(count));
        kcp.wAPLKC.array() += (sp-target)*learning_rate/target;
        kcp.wKCAPL.array() += (sp-target)*learning_rate/target/double(N_KCS);
    }
}

void sim_ORN_layer(int odorid, ORNParams const& ornp, Matrix& orn_t) {
    orn_t = ornp.spont*TIME;
    // "odor input to ORNs"
    Matrix odor = orn_t + ornp.rates.col(odorid)*STIM;
    smoothts_exp(odor, 0.02/DT); // where does 0.02 come from!?

    //Column dORNdt;
    double mul = DT/ornp.taum;
    for (int t = 1; t < N_TIMESTEPS; t++) {
        // dORNdt = -orn_t.col(t-1) + odor.col(t);
        // orn_t.col(t) = orn_t.col(t-1) + dORNdt*DT/ornp.taum;
        orn_t.col(t) = orn_t.col(t-1)*(1.0-mul) + odor.col(t)*mul;
    }
}


void sim_LN_layer(LNParams const& lnp, Matrix const& orn_t, Row& inhA, Row& inhB) {
    Row potential(1, N_TIMESTEPS); potential.setConstant(300.0);
    Row response(1, N_TIMESTEPS);  response.setOnes();
    inhA.setConstant(50.0);
    inhB.setConstant(50.0);
    double inh_LN = 0.0;

    double dinhAdt, dinhBdt, dLNdt;
    for (int t = 1; t < N_TIMESTEPS; t++) {
        dinhAdt = -inhA(t-1) + response(t-1);
        dinhBdt = -inhB(t-1) + response(t-1);
        dLNdt =
            -potential(t-1)
            +pow(orn_t.col(t-1).mean(), 3.0)*51.0/23.0/2.0*inh_LN;
        inhA(t) = inhA(t-1) + dinhAdt*DT/lnp.tauGA;
        inhB(t) = inhB(t-1) + dinhBdt*DT/lnp.tauGB;
        inh_LN = lnp.inhsc/(lnp.inhadd+inhA(t));
        potential(t) = potential(t-1) + dLNdt*DT/lnp.taum;
        //response(t) = potential(t) > lnp.thr ? potential(t)-lnp.thr : 0.0;
        response(t) = (potential(t)-lnp.thr)*double(potential(t)>lnp.thr);
    }
}


void sim_PN_layer(
        PNParams const& pnp, ORNParams const& ornp,
        Matrix const& orn_t, Row const& inhA, Row const& inhB, 
        Matrix& pn_t) {                                        
    Column spont  = ornp.spont*pnp.inhsc/(ornp.spont.sum()+pnp.inhadd);
    pn_t          = spont*TIME;
    double inh_PN = 0.0;

    Column orn_delta;
    Column dPNdt;
    for (int t = 1; t < N_TIMESTEPS; t++) {
        orn_delta = orn_t.col(t-1)-ornp.spont;
        dPNdt = -pn_t.col(t-1) + spont;
        dPNdt += 
            200.0*((orn_delta.array()+pnp.offset)*pnp.tanhsc/200.0*inh_PN).matrix().unaryExpr<double(*)(double)>(&tanh);
        inh_PN = pnp.inhsc/(pnp.inhadd+0.25*inhA(t)+0.75*inhB(t));
        pn_t.col(t) = pn_t.col(t-1) + dPNdt*DT/pnp.taum;
        pn_t.col(t) = (0.0 < pn_t.col(t).array()).select(pn_t.col(t), 0.0);
    }

    // zero non-HC gloms (they are just noise, not from odor...)
    pn_t = ZERO_NONHC_GLOMS * pn_t;
}


void sim_KC_layer(
        KCParams const& kcp, Matrix const& pn_t,                                 
        Matrix& Vm, Matrix& spikes) { 
    Vm.setZero();
    spikes.setZero();
    Row inh(1, N_TIMESTEPS); inh.setZero();
    Row Is(1, N_TIMESTEPS); Is.setZero();

    Column dKCdt;
    for (int t = TIME_START_STEP+1; t < N_TIMESTEPS; t++) {
        double dIsdt = -Is(t-1) + (kcp.wKCAPL*spikes.col(t-1))(0,0)*1e4;
        double dinhdt = -inh(t-1) + Is(t-1);

        dKCdt = 
            -Vm.col(t-1)
            +kcp.wPNKC*pn_t.col(t)
            -kcp.wAPLKC*inh(t-1);
        Vm.col(t) = Vm.col(t-1) + dKCdt*DT/kcp.taum;
        inh(t)    = inh(t-1) + dinhdt*DT/kcp.apl_taum;
        Is(t)     = Is(t-1) + dIsdt*DT/kcp.tau_apl2kc;

        auto const thr_comp = Vm.col(t).array() > kcp.thr.array();
        spikes.col(t) = thr_comp.select(1.0, spikes.col(t)); // either go to 1 or _stay_ at 0.
        Vm.col(t) = thr_comp.select(0.0, Vm.col(t)); // very abrupt repolarization!
    }
}

Matrix model_kc_responses(int nclaws) {
    std::vector<Run> runs(110);

    ORNParams ornp;
    load_hc_data(ornp);

    LNParams lnp;
    PNParams pnp;

    LoadBar lb("Simulating ORN->LN->PN layers");
#pragma omp parallel for
    for (int i = 0; i < runs.size(); i++) {
        Run& r = runs[i];
        sim_ORN_layer(i, ornp, r.orn_t);
        sim_LN_layer(lnp, r.orn_t, r.inhA, r.inhB);
        sim_PN_layer(pnp, ornp, r.orn_t, r.inhA, r.inhB, r.pn_t);
        lb(i+1, runs.size());
    }
    lb.finish();

    KCParams kcp;
    build_wPNKC(kcp, nclaws);
    fit_sparseness(runs, kcp, 0.1);

    lb.reset("Simulating final KC responses");
    Matrix response(N_KCS, runs.size());
    Matrix Vm(N_KCS, N_TIMESTEPS);
    Column nspikes_each(N_KCS, 1);
    for (int i = 0; i < runs.size(); i++) {
        sim_KC_layer(kcp, runs[i].pn_t, Vm, runs[0].kc_spikes);
        nspikes_each = runs[0].kc_spikes.rowwise().sum();
        response.col(i) = (nspikes_each.array()>0.0).select(1.0, nspikes_each);
        lb(i, runs.size());
    }

    return response;

}

int main() {
    Matrix data = model_kc_responses(6);
    dump(data, "kc_response.csv");

    return 0;
}

