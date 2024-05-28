#include <iostream>
using namespace std;
#include "FDM.h"

int main()
{   
    float T = 1;
    float K = 100;
    float sigma = 0.2;
    float r = 0.05;
    float D = 0;
    float S_max = 350;
    char opttype = 'C';

    const int N = 200;
    const int M = 8000;

    FDM_Explicit FDM_Instance(N, M);

    FDM_Instance.compute(T, K, sigma, r, D, S_max, opttype);
    cout << FDM_Instance.predict(100) << endl;

    return 0;	
}


