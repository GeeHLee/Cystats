#include "models.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

void OLS::fit(Eigen::MatrixXf input_X, Eigen::VectorXf input_y){
    X.resize(input_X.rows(), input_X.cols());
    Y.resize(input_y.rows(), 1);
    params.resize(input_X.cols(), 1);
    X << input_X;
    Y << input_y;
    params = (X.transpose() * X).ldlt().solve(X.transpose() * input_y);

    //aic calculator:
    resid = input_y - X * params;
    float ssr = resid.array().pow(2).sum();
    float nobs = input_X.rows();
    float nobs2 = input_X.rows()/2;
    float llf = -nobs2 * log(2*M_PI) - nobs2 * log(ssr/nobs) - nobs2;
    int rank = input_X.fullPivLu().rank();
    aic = -2 * llf + 2 * (rank - 1 + 1);

    //bse:
    float sigma_squared_hat = ssr/(input_X.rows() - input_X.cols() + 1 - 1);
    Eigen::MatrixXf var_beta_hat = (X.transpose() * X).inverse() * sigma_squared_hat; 
    se = var_beta_hat.diagonal().array().pow(0.5);
}


int ADF::get_maxlag(Eigen::Array<float, Eigen::Dynamic, 1> X){
    int nobs = X.rows();
    int maxlag = int(ceil(12 * pow(nobs/100.0, 1/4.0)));
    return min(int(floor(nobs / 2) - ntrend - 1), maxlag);
}

Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> ADF::lagmat(Eigen::Array<float, Eigen::Dynamic, 1> X, int maxlag){
    int nobs = X.rows();
    int nvar = X.cols();
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> lm;
    lm.resize(nobs - maxlag, maxlag + 1);
    lm *= 0;
    for (int i=0; i < maxlag + 1; i++){
        for (int j=0; j < nobs - maxlag; j++){
            lm(j, i) = X(maxlag+j-i, 0);
        }
    }
    return lm;
}

Eigen::ArrayXf ADF::make_diff(Eigen::Array<float, Eigen::Dynamic, 1> X){
    int n_size = X.rows();
    Eigen::ArrayXf diff_mat;
    diff_mat.resize(n_size - 1, 1);
    for (int i=0; i < n_size - 1; i++){
        diff_mat(i, 0) = (X(i+1, 0) - X(i, 0));
    }
    return diff_mat;
}

Eigen::MatrixXf ADF::add_const(Eigen::MatrixXf M, bool prepend){
    Eigen::MatrixXf M_c(M.rows(), M.cols() + 1);
    Eigen::ArrayXf constant = Eigen::ArrayXf::Ones(M.rows(), 1);
    if(prepend == true){
        M_c << constant, M;
    }else{
        M_c << M, constant;
    }
    return M_c;
}

void ADF::run(Eigen::Array<float, Eigen::Dynamic, 1> X, OLS* mod){
    //get maxlag for input:
    int maxlag = ADF::get_maxlag(X);
    //get first diff:
    Eigen::ArrayXf Xdiff = ADF::make_diff(X);
    //get lag matrix:
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> Xdall = ADF::lagmat(Xdiff, maxlag);
    int nobs = Xdall.rows();
    Xdall.col(0) = X.tail(nobs+1).head(nobs);
    Eigen::VectorXf xdshort = Xdiff.tail(nobs);
    Eigen::MatrixXf fullRHS = ADF::add_const(Xdall, true);
    
    int startlag = fullRHS.cols() - Xdall.cols() + 1;
    //do old for all lags:
    float aic_opt = 10e15;
    float lag_opt = 0;
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


Eigen::MatrixXf add_test(Eigen::MatrixXf M, bool prepend){
    Eigen::MatrixXf M_c(M.rows(), M.cols() + 1);
    Eigen::ArrayXf constant = Eigen::ArrayXf::Ones(M.rows(), 1);
    if(prepend == true){
        M_c << constant, M;
    }else{
        M_c << M, constant;
    }
    return M_c;
}


namespace py = pybind11;
PYBIND11_MODULE(Cystats, m){
    py::class_<OLS>(m, "OLS")
        .def(py::init())
        .def("fit", &OLS::fit)
        .def_readwrite("params", &OLS::params)
        .def_readwrite("resid", &OLS::resid)
        .def_readwrite("aic", &OLS::aic)
        .def_readwrite("se", &OLS::se);
        
    py::class_<ADF>(m, "ADF")
        .def(py::init())
        .def("get_maxlag", &ADF::get_maxlag)
        .def("lagmat", &ADF::lagmat)
        .def("make_diff", &ADF::make_diff)
        .def("add_const", &ADF::add_const, py::return_value_policy::reference_internal)
        .def("run", &ADF::run);
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

