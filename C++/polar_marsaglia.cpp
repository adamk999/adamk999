#include<iostream>
#include<cmath>
using namespace std;

double normal(double std, double mean);

/*int main()
{
    cout << normal(0.3, 1) << endl;
}
*/

double normal(double std, double mean)
{
    static int iset = 0;
    static double gset;
    double fac, r, v1, v2;
    
    // create two normally-distributed numbers
    if (iset == 0)
    {
        r = 0;
        do
        {
            //compute two possibles
            v1 = 2.0 * rand() / RAND_MAX - 1.0;
            v2 = 2.0 * rand() / RAND_MAX - 1.0;
            // they define radius
            r = v1 * v1 + v2 * v2;
        } while (r >= 1.0 || r == 0.0);
        // in unit circle? if not try again

        fac = sqrt((-2 * log(r)) / r); // Box-Muller transform
        gset = (v1 * fac);
        iset = 1;  // save one
        v2 = v2 * fac * std + mean; // scale and return one
        return v2;
    }
    else
    {
        iset = 0;
        return (gset * std) + mean;
        //scale and return the saved one
    } 
}
