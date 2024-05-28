#include <iostream>
#include <cmath>
using namespace std;
#include "FDM.h"


float interp(float x, const float* xp, const float* yp, const unsigned int n)
{
    int idx; 
    
    for (int i=0; i<n; i++)
    {
       if (xp[i] > x)
       {
           idx = i-1;
           break;
       }
    }

    // Calculate grad of line between the two nearest points.
    float grad = (yp[idx+1] - yp[idx]) / (xp[idx+1] - xp[idx]);
    
    // Interpolate the y-value using line eq.
    float y = yp[idx] + grad*(x - xp[idx]);
    
    return y;

}


float* arange(int start, int stop)
{
    float* arr;

    int length = abs(stop - start);

    arr = new float[length];

    for (int i=0; i<length; i++)
    {
        if (start <= stop)
        {
            arr[i] = start + i;
        }
        else
        {
            arr[i] = start - i;
        }
    }
    return arr;
}


void FDM_Explicit::compute(float T, float K, float sigma, float r, float D, float S_max, char opttype)
{
    dt = T/M;
    dS = S_max/N;
    const int I = (opttype == 'C') ? 1 : -1;  
    float constraint = dt*(pow(sigma,2)*pow(N,2));
    cout << "Constraint (< 1): " << constraint << endl;


    if (constraint < 1)
    {            
        float alpha[N];
        float beta[N];
        float gamma[N];

        for (int i=0; i<N; i++)
        {
            // Compute boundary conditions.
            alpha[i] = 0.5*(pow(i,2)*pow(sigma,2) - i*(r - D))*dt;
            beta[i] = 1 - (r + pow(i,2)*pow(sigma,2))*dt;
            gamma[i] = 0.5*(pow(i,2)*pow(sigma,2) + i*(r - D))*dt;
            
            // Compute final payoff.
            grid[i][M-1] = max(I*(i*dS - K), 0.0f);
        }

        // Walk back through the grid.
        for (int m=M-1; m>-1; m--)
        {
            // S = 0.
            grid[0][m-1] = beta[0]*grid[0][m];
            
            // S = S*.
            grid[N-1][m-1] = (alpha[N-1] - gamma[N-1])*grid[N-2][m] + (beta[N-1] + 2*gamma[N-1]*grid[N-1][m]);

            for (int i=1; i<N-1; i++)
            {
                grid[i][m-1] = alpha[i]*grid[i-1][m] + beta[i]*grid[i][m] + gamma[i]*grid[i+1][m];
            }
        }
    }
}

float FDM_Explicit::predict(float S)
{
    float* x = arange(0, N);
    float* y = new float[N];

    for (int i=0; i<N; i++)
    {
        x[i] *= dS;
        y[i] = grid[i][0];
        //cout << x[i] << ", " << y[i] << endl;
    } 
    
    float y_star = interp(S, x, y, N);
    
    delete[] y;
    
    return y_star;
}



