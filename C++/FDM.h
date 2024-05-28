#pragma once

void Print();

float* arange(int start, int stop);

struct OptionParams
{
    float T;
    float K;
    float sigma;
    float r;
    float D;
    float S_max;
    char opttype;
};

class FDM_Explicit
{
private:
    float dS, dt;
    float** grid;
    

public:
    const int N, M;

    FDM_Explicit(const int N_rows, const int M_cols) : N(N_rows), M(M_cols) 
    {
        // Allocate memory for the rows
        grid = new float*[N_rows];
        for (int i = 0; i < N_rows; i++) 
        {
            grid[i] = new float[M_cols];
        }
    } 

    ~FDM_Explicit() 
    {
        // Deallocate memory for the rows.
        for (int i = 0; i < N; i++) 
        {
            delete[] grid[i];
        }
        // Deallocate memory for the array of pointers.
        delete[] grid;
    }

    void compute(float T, float K, float sigma, float r, float D, float S_max, char opttype);
    float predict(float S);
};

