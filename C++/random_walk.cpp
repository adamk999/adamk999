#include<iostream>
#include<cmath>
#include<ctime>
#include<fstream>
using namespace std;

double normal(double, double); // function prototype

int main()
{
       srand((unsigned)time(NULL));
       ofstream print;
       print.open("results.xls");
       
       long N = 1000;
       double asset = 100, IR = 0.05, vol = 0.2; // variables & parameters
       double dt = 1.0 / N;  // step-size for time
       
       print << 0 << '\t' << asset << endl;
       
       for (unsigned short int i = 1; i <= N; i++) {
              double time = i * dt;
              double dX = normal(1.0, 0.0) * sqrt(dt);
              double dS = asset * (IR * dt + vol * dX);
              asset += dS;
              print << time << '\t' << asset << endl;
       }
       
       print.close();
       
       return 0; 
}
