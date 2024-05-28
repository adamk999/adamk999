void fill_up(double a[], int& size)
{
    cout << "Enter size" << endl;
    cin >> size;
    cout << "Now enter numbers" << endl;

    for (int i=0; i <size; i++)
    {
        cin >> a[i];
    }
}

void display(double a[], int& size)
{
    cout << "You have entered the following values: " <<endl;
    for (int i=0; i<size; i++)
    {
        cout << a[i] << endl;
    }
}

double average(double a[], int size)
{
    double av = 0.0;
    for (int i=0; i<size; i++)
    {
        av += a[i];
    }
    double mean = av/size;
    cout << "The average is " << mean << endl;
    return mean;
}

double stan_dev(double a[], int size, double mean)
{
    double sum = 0.0;
    for (int i=1; i<=size; i++)
    {
        sum += ((a[i] - mean)*(a[i] - mean));
    }
    double std = sum/size;
    cout << "The standard deviation is " << sqrt(std) << endl;
    return 0;
}
