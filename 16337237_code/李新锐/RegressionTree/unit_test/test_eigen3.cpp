//
// Created by 李新锐 on 01/10/2018.
//

#include <iostream>
#include <Eigen/Dense>
#include "gtest/gtest.h"
#ifdef USE_MLK
    #include "mkl.h"
#endif
#include "range/v3/view.hpp"
#include "functional"
#include "AI_utility.h"
using namespace ranges;


using namespace std;
using namespace Eigen;
using Eigen::MatrixXd;

class Foo
{
public:
    Eigen::Vector2d v;
};

TEST(E, 1)
{
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}

TEST(E, 2)
{
    MatrixXd m(2,2);
    m(0,0) = 1;
    m(1,0) = 2;
    m(0,1) = 3;
    m(1,1) = 4;
    m = (m + MatrixXd::Constant(2, 2, 1)) * 50;
    cout << "m is " << endl << m << endl;
    VectorXd v(2);
    v << 1, 2;
    cout << v << endl;
    cout << "m * v = " << endl << m * v << endl;

}

TEST(E, 3)
{
    MatrixXd m(2, 5);
    m.resize(4, 3);
    m.transposeInPlace();
}

TEST(E, 4)
{
    Foo *foo = new Foo;
    foo-> v << 1, 2;
    cout << foo->v << endl;
}

TEST(E, 5)
{
    std::vector<std::vector<int>> trainX = {
            {1, 2, 3, 4},
            {2, 3, 4, 5},
            {3, 4, 5, 6}
    };
    std::set<int> D = {1,2};
    auto v = std::vector<const std::vector<int>*>{};
    for (auto i : D)
    {
        v.push_back(&trainX[i]);
    }
    for(auto x : v)
    {
        for(const auto& i : *x)
            cout << i << ',';
        cout << endl;
    }
}


TEST(E, 6)
{
    Eigen::MatrixXd trainX(3,4);
    trainX<<1, 2, 3, 4,
            2, 3, 4, 5,
            3, 4, 5, 6;
    Eigen::MatrixXd mp = trainX.block(0,1, 3,2);
//    Eigen::Map<MatrixXd> mp(trainX.data() + 6 * sizeof(double), trainX.rows(), trainX.cols() / 2);
    cout << mp;
}

int main(int argc, char *argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

