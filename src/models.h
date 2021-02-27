#include <iostream>
#include <Eigen/Dense>
#include <algorithm>
#include <math.h>

using namespace std;

class OLS{
    public:
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> X;
        Eigen::Array<float, Eigen::Dynamic, 1> Y;
        Eigen::VectorXf params;
        Eigen::VectorXf resid;
        float aic;
        Eigen::VectorXf se;

        void fit(Eigen::MatrixXf input_X, Eigen::VectorXf input_y);
};

class ADF{
    public:
        float stat;
        // get maxlag function
        int get_maxlag(Eigen::Array<float, Eigen::Dynamic, 1> X);
        // lagmat function
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lagmat(Eigen::Array<float, Eigen::Dynamic, 1> X, int maxlag);
        // make diff for 1 order
        Eigen::ArrayXf make_diff(Eigen::Array<float, Eigen::Dynamic, 1> X);
        // add constant to matrix
        Eigen::MatrixXf add_const(Eigen::MatrixXf M, bool prepend);
        // run test function
        void run(Eigen::Array<float, Eigen::Dynamic, 1> X, OLS* mod);
    private:
        int ntrend = 1;
        Eigen::Vector2f out = Eigen::Vector2f::Zero();
};
