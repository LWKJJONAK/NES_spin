#include "minresqlp.h"
#include <iostream>
#include <Eigen/Dense>

int main()
{
    Eigen::VectorXd v(3);
    Eigen::MatrixXd m(3, 3);
    v << 1, 2, 3;
    m << 1, 0, 1, 1, 2, 1, 1, 4, 1;
    Eigen::VectorXd r(3);
    minresqlp([&](auto &&v){ return m * v; }, v, r, 1e-5, 1000);
    std::cout << r << std::endl;
    return 0;
}
