#include <cmath>

//S_0 = 100
//u = 1.2
//d = 1/u
//K = 100
//r = 0
//n = 4
//T = 1
//
//def binomial_tree(K, T, S_0, r, n, u, d, opttype='C'):
//    # Compute constants.
//    direc = 1 if opttype == "C" else -1
//  dt = T/n
//    q = (np.exp(r*dt) - d) / (u-d)
//    gamma = np.exp(-r*dt)
//    
//    # Initialise asset prices at maturity.
//    V = S_0 * d ** (np.arange(n,-1,-1)) * u ** (np.arange(0,n+1,1))
//    
//    # Initialise option values at maturity.
//    V = np.maximum(direc*(V - K), 0)
//        
//    # Step backwards through tree.
//    for i in np.arange(n,0,-1):
//        V = gamma*(q*V[1:i+1] + (1-q)*V[0:i])
//
//    return V[0]
//
//binomial_tree(K,T,S_0,r,n,u,d)


struct Array 
{
    float* data;
    int length; 

    // Constructor to initialize the array with given length.
    Array(int n) 
    {
        data = new float[n]; // Dynamically allocate memory for the array.
        length = n;
    }

    // Destructor to release the dynamically allocated memory.
    ~Array() 
    {
        //cout << "Destroying Array" << endl;
        delete[] data;
    }


    void print()
    {
        for (int i=0; i<length; i++)
        {
            cout << data[i] << endl;
        }
    }
};


Array* arange(int start, int stop)
{

    int length = abs(stop - start);

    Array* arr = new Array(length);

    for (int i=0; i<length; i++)
    {
        if (start <= stop)
        {
            arr->data[i] = start + i;
        }
        else
        {
            arr->data[i] = start - i;
        }
    }
    return arr;
}


float Binomial_Tree(float K, float T, float S_0, float r, int n, float u, float d, int direc)
{
    float dt = T/n;
    float q = (exp(r*dt) - d)/(u - d);
    float gamma = exp(-r*dt);

    // Initialise asset prices at maturity.
    Array V = Array(n);

    for (int i=0; i<n; i++)
    {
        V.data[i] = S_0*pow(d,n-i)*pow(u,i);
    }


    // Initialise option values at maturity.
    for (int i=0; i<n; i++)
    {
        V.data[i] = max(direc*(V.data[i] - K), 0.0f);
    }
    
    // Step backwards through tree.
    // for i in np.arange(n,0,-1):
    //      V = gamma*(q*V[1:i] + (1-q)*V[0:i-1])
    
    for (int i=n; i>0; i--)
    {
        for (int j=0; j<i; j++)
        {
            V.data[j] = gamma*(q*V.data[j+1] + (1-q)*V.data[j]);
        }
    }

    return V.data[0];

}
