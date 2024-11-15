#include <iostream>
#include <cmath>
#include <cstdlib>
#include <fstream>

using namespace std;

int test(int a)
{
    static int first_time = 1;
    static ofstream fdata, fsite, farate;

    if (first_time) {
        fdata.open("corr.txt");
        if (fdata.bad()) {
            cout << "Failed to open correlator file\n" << flush;
        }
        farate.open("acceptance.txt");
        if (farate.bad()) {
            cout << "Failed to open acceptance rate file\n" << flush;
        }
        fsite.open("site.txt");
        if (fsite.bad()) {
            cout << "Failed to open sites data file\n" << flush;
        }
        first_time = 0;
    }

    // simulation parameters
    int THERM = 1000000; // number of thermalization steps
    int SWEEPS = 1000000; // number of generation steps
    int GAP = 100; // interval between measurements
    double DELTA = 0.5; // random shift range
    double shift, u; // random shift, random number value
    double tot = 0.0;
    int accept = 0, nocalls = 0; // for acceptance rate

    // physics parameters
    int T = 64; // number of time slices
    double omega = 1.0; // frequency omega
    double m = 1.0; // mass m
    double dS; // change in action
    double site[T], old_site[T], new_site[T];

    // observables
    double corr[64] = { }; // to store correlator data
    double corr_sq[64] = { }, stderr[64] = { };
    double xsq = 0.0, xsq_sq = 0.0;
    double xval = 0.0, xval_sq = 0.0;
    double stderr_xval, stderr_xsq;
    int tau; // to choose a random site

    // write out initially
    cout << "MCMC for Simple Harmonic Oscillator" << endl;
    cout << "Mass m = " << m << endl;
    cout << "Frequency omega = " << omega << endl;

    // initialize observables, etc.
    for (int t = 0; t < T; t++) {
        site[t] = (drand48() - 0.5);
        old_site[t] = 0.0;
        new_site[t] = 0.0;
    }

    // begin thermalization MC sweeps
    for (int i = 1; i <= THERM; i++) {
        for (int t = 0; t < T; t++) {
            tau = int(T * drand48());

            old_site[tau] = site[tau];
            shift = 2.0 * DELTA * (drand48() - 0.5);
            new_site[tau] = site[tau] + shift;

            if (tau != (T - 1)) {
                dS = (pow(site[tau + 1] - new_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[tau + 1] + new_site[tau], 2.0))
                    - (pow(site[tau + 1] - old_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[tau + 1] + old_site[tau], 2.0));
                dS = (m / 2.0) * dS;
            } else {
                dS = (pow(site[0] - new_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[0] + new_site[tau], 2.0))
                    - (pow(site[0] - old_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[0] + old_site[tau], 2.0));
                dS = (m / 2.0) * dS;
            }

            u = drand48();
            if (u < exp(-dS)) {
                site[tau] = new_site[tau];
                accept++;
                cout << "ACCEPTED with dS of " << dS << endl;
            } else {
                site[tau] = old_site[tau];
                cout << "REJECTED with dS of " << dS << endl;
            }
        }
    }

    // begin generation MC steps
    for (int i = 1; i <= SWEEPS; i++) {
        for (int t = 0; t < T; t++) {
            nocalls++;
            if ((nocalls % 100 == 0) && (!first_time)) {
                cout << "Acceptance rate " << double(accept) / double(nocalls) << "\n" << flush;
                farate << double(accept) / double(nocalls) << endl;
                nocalls = 0;
                accept = 0;
            }

            tau = int(T * drand48());
            old_site[tau] = site[tau];
            shift = 2.0 * DELTA * (drand48() - 0.5);
            new_site[tau] = site[tau] + shift;

            if (tau != (T - 1)) {
                dS = (pow(site[tau + 1] - new_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[tau + 1] + new_site[tau], 2.0))
                    - (pow(site[tau + 1] - old_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[tau + 1] + old_site[tau], 2.0));
                dS = (m / 2.0) * dS;
            } else {
                dS = (pow(site[0] - new_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[0] + new_site[tau], 2.0))
                    - (pow(site[0] - old_site[tau], 2.0)
                    + 0.25 * omega * omega * pow(site[0] + old_site[tau], 2.0));
                dS = (m / 2.0) * dS;
            }

            u = drand48();
            if (u < exp(-dS)) {
                site[tau] = new_site[tau];
                accept++;
            } else {
                site[tau] = old_site[tau];
            }
        }
    }

    return 0;
}

int main()
{
    int a = 10;
    int main(a);

    return 0;
}
