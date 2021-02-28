#include "models.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
# define M_PI 3.141592653589793  /* pi */

void OLS::fit(Eigen::MatrixXd input_X, Eigen::VectorXd input_y, bool has_const=true){
    //set fit is true:
    _fit = true;
    int n = input_X.rows();
    int m = input_X.cols();

    X.resize(n, m);
    Y.resize(n, 1);
    params.resize(m, 1);
    X << input_X;
    Y << input_y;
    params = (X.transpose() * X).ldlt().solve(X.transpose() * input_y);

    // make a prediction:
    fittedvalues = OLS::predict(input_X);
    // calculate the residual:
    resid.resize(n, 1);
    for(int i=0; i<n; i++){
        resid(i, 1) = input_y(i, 1) - fittedvalues(i, 1);
    }
    // resid = input_y - fittedvalues;
    // calculate the rss:
    _rss = resid.array().pow(2).sum();
    // calculate loglikehood:
    llf = OLS::log_like(n ,_rss);
    // calculate AIC:
    double rank = input_X.fullPivLu().rank();
    if (has_const == true){
        k_constant = 1;
    }
    degree = rank - k_constant;
    aic = -2 * llf + 2 * (degree + k_constant);

    // cov_mat:
    Eigen::MatrixXd con_mat = _rss/(n - degree - 1) * (X.transpose() * X).inverse();
    se = con_mat.diagonal().array().pow(0.5);
    tvalues = params.array()/se.array();
}

Eigen::VectorXd OLS::predict(Eigen::MatrixXd input_X){
    if (_fit == true){
        return input_X*params.matrix();
    }else(
        throw "fit does not be called yet."
    );
}

double OLS::log_like(int n, double rss){
    int n2 = n/2.0;
    return -n2 * log(2*M_PI) - n2 * log(rss/n) - n2;
}



int ADF::get_maxlag(Eigen::Array<double, Eigen::Dynamic, 1> X){
    int nobs = X.rows();
    int maxlag = int(ceil(12 * pow(nobs/100.0, 1/4.0)));
    return min(int(floor(nobs / 2) - ntrend - 1), maxlag);
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ADF::lagmat(Eigen::Array<double, Eigen::Dynamic, 1> X, int maxlag){
    int nobs = X.rows();
    int nvar = X.cols();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> lm;
    lm.resize(nobs - maxlag, maxlag + 1);
    lm *= 0;
    for (int i=0; i < maxlag + 1; i++){
        for (int j=0; j < nobs - maxlag; j++){
            lm(j, i) = X(maxlag+j-i, 0);
        }
    }
    return lm;
}

Eigen::ArrayXd ADF::make_diff(Eigen::Array<double, Eigen::Dynamic, 1> X){
    int n_size = X.rows();
    Eigen::ArrayXd diff_mat;
    diff_mat.resize(n_size - 1, 1);
    for (int i=0; i < n_size - 1; i++){
        diff_mat(i, 0) = (X(i+1, 0) - X(i, 0));
    }
    return diff_mat;
}

Eigen::MatrixXd ADF::add_const(Eigen::MatrixXd M, bool prepend){
    Eigen::MatrixXd M_c(M.rows(), M.cols() + 1);
    Eigen::ArrayXd constant = Eigen::ArrayXd::Ones(M.rows(), 1);
    if(prepend == true){
        M_c << constant, M;
    }else{
        M_c << M, constant;
    }
    return M_c;
}

void ADF::run(Eigen::Array<double, Eigen::Dynamic, 1> X, OLS* mod){
    //get maxlag for input:
    int maxlag = ADF::get_maxlag(X);
    //get first diff:
    Eigen::ArrayXd Xdiff = ADF::make_diff(X);
    //get lag matrix:
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Xdall = ADF::lagmat(Xdiff, maxlag);
    int nobs = Xdall.rows();
    Xdall.col(0) = X.tail(nobs+1).head(nobs);
    Eigen::VectorXd xdshort = Xdiff.tail(nobs);
    Eigen::MatrixXd fullRHS = ADF::add_const(Xdall, true);
    
    int startlag = fullRHS.cols() - Xdall.cols() + 1;
    //do old for all lags:
    double aic_opt = 10e15;
    double lag_opt = 0;
    for(int lag=startlag; lag < startlag + maxlag + 1; lag++){
        mod->fit(fullRHS.leftCols(lag), xdshort);
        if(mod->aic <= aic_opt){
            aic_opt = mod->aic;
            lag_opt = lag;
        }
    }
    lag_opt = lag_opt - startlag;

    // re do ols for best lag:
    Xdall = ADF::lagmat(Xdiff, maxlag=lag_opt);
    nobs = Xdall.rows();
    Xdall.col(0) = X.tail(nobs+1).head(nobs);
    xdshort = Xdiff.tail(nobs);
    int usedlag = lag_opt;
    mod->fit(ADF::add_const(Xdall, false), xdshort);
    stat = (mod->params.array()/mod->se.array())(0, 0);
}

namespace py = pybind11;
PYBIND11_MODULE(Cystats, m){
    py::class_<OLS>(m, "OLS")
        .def(py::init())
        .def("fit", &OLS::fit)
        .def("predict", &OLS::predict)
        .def_readwrite("params", &OLS::params)
        .def_readwrite("fittedvalues", &OLS::fittedvalues)
        .def_readwrite("resid", &OLS::resid)
        .def_readwrite("rss", &OLS::_rss)
        .def_readwrite("llf", &OLS::llf)
        .def_readwrite("aic", &OLS::aic)
        .def_readwrite("degree", &OLS::degree)
        .def_readwrite("se", &OLS::se)
        .def_readwrite("tvalues", &OLS::tvalues);
        
    py::class_<ADF>(m, "ADF")
        .def(py::init())
        .def("get_maxlag", &ADF::get_maxlag)
        .def("lagmat", &ADF::lagmat)
        .def("make_diff", &ADF::make_diff)
        .def("add_const", &ADF::add_const, py::return_value_policy::reference_internal)
        .def("run", &ADF::run)
        .def_readwrite("stat", &ADF::stat);
}


// int main(){
//     OLS* mod = new OLS();
//     ADF* test = new ADF();
//     Eigen::VectorXf input_X(10);
    
//     input_X << 0.23298801, -0.78057248, -1.23946256 , 0.12780824, -0.85410113, 0.77147467, 2.11312377, 0.75754486, 0.11385792, 1.86346807;
//     int maxlag = test->get_maxlag(input_X);
//     Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lm = test->lagmat(input_X, maxlag);
//     test->run(input_X, mod);
//     cout << test->stat << endl;
//     delete test;
// }





// PYBIND11_MODULE(LR, m){
//     pybind11::class_<OLS>(m, "OLS")
//         .def(pybind11::init())
//         .def("fit", &OLS::fit);
// }

// int main(){
//     OLS* model = new OLS();
//     Eigen::MatrixXf input_X(6, 3);
//     Eigen::VectorXf input_Y(6);
//     input_X <<  1, 0.127808, 1.36727, 
//         1, -0.854101, -0.981909, 
//         1, 0.771475, 1.62558, 
//         1, 2.11312, 1.34165, 
//         1, 0.757545, -1.35558, 
//         1, 0.113858, -0.643687;
//     input_Y << -0.981909, 1.62558, 1.34165, -1.35558, -0.643687, 1.74961;
//     model->fit(input_X, input_Y);
//     delete model;
// }

