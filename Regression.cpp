#include <iostream>
#include <math.h>
#include <vector>
#include <functional> // std::minus, std::divides 
#include <numeric> // std::inner_product 
  

using namespace std;


class DataSet
{
    private:  
        vector<vector <double> > data;
    
    public:     
        void push(double x, double y) 
        {
            vector<double> entry(x,y);
            data.push_back(entry);
        }

        vector<vector <double> > getD()
        {
            return data;
        }
};

class Regression
{
    private:   
        vector<vector <double> > data;
        double sumX;
        double sumY;
        double inpXX;
        double inpXY;
        double inpYY;

        double accumulate(int index)
        {
            double sum = 0;

            for (int v = 0; v < data.size(); v ++)
            {
                sum += data[v][index];
            }

            return sum;

        }

        double innerProduct(int indexX, int indexY)
        {
            double inprod = 0;

            if  (indexX != indexY) 
            {   
                // innerproduct : sum += xi * yi
                for (int v = 0; v < data.size(); v ++)
                {
                    
                    inprod += (data[v][indexX] * data[v][indexY]);
                    
                }
            }

            else 
            {
                //inner product: sum += xi * x(i+1)
                for (int v = 0; v < data.size()-1; v ++)
                {
                    
                    inprod += (data[v][indexX] * data[v+1][indexX]);
                    
                }
            }
            return inprod;
        }

        double return_cost(double a, double b, double da, double db)
        {
            int size = data.size();
            inpYY = innerProduct(1, 1);

            double cost = inpYY - 2 * a * inpXY - 2 * b * sumY + pow(a,2) * inpXX + 2 * a * b * sumX + size * pow(b,2);
            cost /= size;

            da = 2 * (-inpXY + a * inpXX + b * sumX) / size;
            db = 2 * (-sumY + a * sumX + size * b) / size;

            return cost;
        }


    public:   
        Regression (vector<vector <double> > dataset) 
        {
            data = dataset;
        }

        double slope()
        {
            sumX = accumulate(0);
            sumY = accumulate(1);

            inpXX = innerProduct(0, 0);
            inpXY = innerProduct(0, 1);

            int size = data.size();

            double denor = (size  * inpXX) -  (sumX * sumX);
            double nor = (size * inpXY) -  (sumX * sumY);

            if (denor != 0)
            {
                return nor / denor;
            }

            else
            {
                
                return numeric_limits<double>::max();
            }
        }

        double intercept(double slope)
        {
            int size = data.size();
            return (sumY - slope * sumX) /  size; 
        }


        void LinearRegression(double slope = 1, double intercept = 0)
        {
            double lrate = 0.0002;
            double threshold = 0.0001;
            int counter = 0;

            while (true)
            {
                double da = 0; double db = 0;
                double cost = return_cost(slope, intercept, da, db);

                
                cout << "pass: " << counter << " cost = " << cost << " da = " << da << " db = " << db << endl;

                counter ++;

                if (abs(da) < threshold && abs(db) < threshold)
                {
                    cout << "\n\ny = " << slope << "x" << " + "<< intercept << endl;
                    break;
                }

                //decrement
                slope -= lrate* da;
                intercept -= lrate * db;

            }
        }    
};


int main()
{
    // y = bias + coefficient * x

    DataSet data = DataSet();
    data.push(71, 160);
    data.push(73, 183);
    data.push(64, 154);
    data.push(65, 168);
    data.push(61, 159);
    data.push(70, 180);
    data.push(65, 145);
    data.push(72, 210);
    data.push(63, 132);
    data.push(67, 168);
    data.push(64, 141);

    Regression linear = Regression(data.getD());
    double slope = linear.slope();
    double intercept = linear.intercept(slope);

    cout << "slope " << slope << " intercept: " << intercept << endl;


    linear.LinearRegression(slope, intercept);
    
}