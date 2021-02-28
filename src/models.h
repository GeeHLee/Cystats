#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <math.h>

using namespace std;

class OLS{
    public:
        // copy of inputs:
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> X;
        Eigen::Array<double, Eigen::Dynamic, 1> Y;
        
        // set attribute of OLS class:
        // params is the coefficient of each variable of X.
        Eigen::VectorXd params;
        Eigen::VectorXd fittedvalues;
        Eigen::VectorXd resid;
        double _rss = 0;
        double llf;
        double aic = 0;
        double degree = 0;
        Eigen::VectorXd se;
        Eigen::VectorXd tvalues;

        // public functions:
        void fit(Eigen::MatrixXd input_X, Eigen::VectorXd input_y, bool has_const);
        Eigen::VectorXd predict(Eigen::MatrixXd input_X);
        double log_like(int n, double rss);
    
    private:
        bool _fit = false;
        double k_constant = 0.0;
};

class ADF{
    public:
        double stat;
        // get maxlag function
        int get_maxlag(Eigen::Array<double, Eigen::Dynamic, 1> X);
        // lagmat function
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lagmat(Eigen::Array<double, Eigen::Dynamic, 1> X, int maxlag);
        // make diff for 1 order
        Eigen::ArrayXd make_diff(Eigen::Array<double, Eigen::Dynamic, 1> X);
        // add constant to matrix
        Eigen::MatrixXd add_const(Eigen::MatrixXd M, bool prepend);
        // run test function
        void run(Eigen::Array<double, Eigen::Dynamic, 1> X, OLS* mod);
    private:
        int ntrend = 1;
        Eigen::Vector2d out = Eigen::Vector2d::Zero();
};
