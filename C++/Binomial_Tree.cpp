#include <iostream>
#include <algorithm>
using namespace std;
#include "Binomial_Tree.h"

int main()
{
    float S_0 = 100;
    float u = 1.2;
    float d = 1/u;
    float K = 100;
    float r = 0;
    int n = 20;
    float T = 1;

    cout << Binomial_Tree(K, T, S_0, r, n, u, d, 1) << endl;

	return 0;
}
