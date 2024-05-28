#include<iostream>
#include<cmath>
#include<fstream>
using namespace std;

const double pi = 4.0 * atan(1.0); //define constant pi=3.142
double CDF(double);

int main()
{
    double Call_Option, d1, d2;
    double S = 100, E = 100, r = 0.05, vol = 0.2, tau = 0.05; 
    
    ofstream out; // create object “out” to printout to excel file 
    out.open("BSE.xls"); 

    for (S = 0; S <= 200; S += 5) 
    {
        d1 = (log(S / E) + (r + 0.5 * vol * vol) * tau) / (vol * sqrt(tau));
        d2 = d1 - vol * sqrt(tau);
        Call_Option = S * CDF(d1) - E * exp(-r * tau) * CDF(d2);
    
        out << S << '\t' << Call_Option << endl;
    }
    
    out.close();

    return 0; 
}


double CDF(double X)
{
    const double a1 = 0.319381530, a2 = -0.356563782, a3 = 1.781477937, a4 = -1.821255978, a5 = 1.330274429;
    double x = 0, k = 0; //always good to initialise variables
    double N, CDF, n;
    
    x = fabs(X);
    k = 1 / (1 + 0.2316419 * x);
    n = (1 / sqrt(2 * M_PI)) * exp(-0.5 * x * x);
    N = 1 - n * (a1 * k + a2 * k * k + a3 * pow(k, 3) + a4 * pow(k, 4) + a5 * pow(k, 5));
    
    if (X >= 0)
        CDF = N;
    else 
        CDF = 1 - N;

    return CDF; 
}
