#include <iostream>
#include <math.h>
#include <vector>
#include <functional> // std::minus, std::divides 
#include <numeric> // std::inner_product 
  

using namespace std;

class Regression
{
    private:   
        vector<vector <double> > train_data;
        vector<vector <double> > test_data;

        double sumX;
        double sumY;
        double inpXX;
        double inpXY;
        double inpYY;


        struct Coefficient
        {
            double theta0;
            double theta1;

        };

        Coefficient* coefficient0;
        Coefficient* coefficient1;

        double accumulate(int index)
        {
            double sum = 0;

            for (int v = 0; v < train_data.size(); v ++)
            {
                sum += train_data[v][index];
            }

            return sum;

        }

        double mean(int index)
        {
            int size = train_data.size();

            double sum = 0;
            
            for (int v = 0; v < train_data.size(); v ++)
            {
            
                sum += train_data[v][index];

            }
            
            return sum/size;
        }

        double innerProduct(int indexX, int indexY)
        {
            double inprod = 0;

            if  (indexX != indexY) 
            {   
                // innerproduct : sum += xi * yi
                for (int v = 0; v < train_data.size(); v ++)
                {
                    
                    inprod += (train_data[v][indexX] * train_data[v][indexY]);
                    
                }
            }

            else 
            {
                //inner product: sum += xi * x(i+1)
                for (int v = 0; v < train_data.size()-1; v ++)
                {
                    
                    inprod += (train_data[v][indexX] * train_data[v+1][indexX]);
                    
                }
            }
            return inprod;
        }

    public:   
        Regression ()
        {
            coefficient0 = new Coefficient();
            coefficient1 = new Coefficient();
        }

        void push(double x, double y) 
        {
            vector<double> entry;
            entry.push_back(x);
            entry.push_back(y);

            train_data.push_back(entry);
        }

        void push_test(double x, double y)
        {
            vector<double> entry;
            entry.push_back(x);
            entry.push_back(y);

            test_data.push_back(entry); 
        }

        void estimate_coefficient()
        {
            int size = train_data.size();
            double meanX = mean(0);
            double meanY = mean(1);

            double sumX = accumulate(0);
            double sumY = accumulate(1);

            double inpXX = innerProduct(0,0);
            double inpXY = innerProduct(0,1);

            double SS_xy = inpXY - (size * meanY * meanX);
            double SS_xx = inpXX - (size * meanX * meanX);

            double SS_xy_sum = inpXY - (size * sumY * sumX);
            double SS_xx_sum = inpXY - (size * sumX * sumX);
             
            double theta_1 = SS_xy / SS_xx;
            double theta_0 = meanY - theta_1 * meanX;

            double theta_1_sum = SS_xy_sum / SS_xx_sum;
            double theta_0_sum = sumY - theta_1_sum * sumX;

            coefficient0->theta0 = theta_0;
            coefficient0->theta1 = theta_1;

            coefficient1->theta0 = theta_0_sum;
            coefficient1->theta1 = theta_1_sum;

            //H0 is based on using mean to find thetas
            //H1 is based on using sum to find thetas
            cout << "H0: y = " << coefficient0->theta0 << " + " << coefficient0->theta1 << "x " << endl;
            cout << "H1: y = " << coefficient1->theta0 << " + " << coefficient1->theta1 << "x " << endl;

            cout << "Pick the lowest cost" << endl;
            cout << "H0's cost: " << cost(coefficient0) << endl;
            cout << "H1's cost: " << cost(coefficient1) << endl; 

        }

        double cost(Coefficient* coefficient)
        {
            double sum_of_squared = 0;
            int m = train_data.size();

            double theta0 = coefficient->theta0;
            double theta1 = coefficient->theta1;

            for (int v = 0; v < test_data.size(); v ++ )
            {
                // grab X from test_data
                double y = theta0 + theta1 * test_data[v][0];
        
                //compare Y and squared it
                double y_difference = pow((test_data[v][1] - y), 2);

                sum_of_squared += y_difference;

            }
            
            return sum_of_squared / (2 * m);

        }

};


int main()
{
    // y = bias + coefficient * x
    Regression data = Regression();
    data.push(84, 21);
    data.push(70, 14);
    data.push(37, 8);
    data.push(152, 33);
    data.push(39, 9);
    data.push(53, 13);
    data.push(77, 22);
    data.push(72, 17);
    data.push(157, 36);
    data.push(97, 30);
    data.push(117, 31);
    data.push(87, 18);
    data.push(145, 39);
    data.push(113, 26);
    data.push(123, 37);
    data.push(18, 4);
    data.push(104, 32);
    data.push(149, 47);
    data.push(55, 15);
    data.push(76, 21);
    data.push(126, 36);
    data.push(71, 17);
    data.push(16, 3);
    data.push(116, 37);
    data.push(94, 22);
    data.push(58, 14);
    data.push(56, 16);
    data.push(106, 22);
   
    data.push_test(85, 17);
    data.push_test(93, 29);
    data.push_test(29, 27);
    data.push_test(99, 21);
    data.push_test(156, 34);
    data.push_test(46, 11);
    data.push_test(76, 19);
    data.push_test(83, 19);
    data.push_test(50, 12);

    data.estimate_coefficient();

}