---
author: "KnightSnape"
title: "ThirdParty Test and Summary"
date: "2024-1-22"
description: "Test and evaluate cpp ThirdParty and Summary It"
tags: ["ThirdParty"]
aliases: ["migrate-from-jekyl"]
ShowToc: true
TocOpen: true
weight: 2
---

# C++第三方库

## 前言

有关这个部分将是在我目前对我遇到的所有C++第三方库的认知，测试以及相关的总结，由于自己实力很给，所以如果总结有误，请联系KnightSnape

* [1.框架类](#1-框架类)

    * [1.1.Qt](#11-Qt)

    * [1.2.Boost](#12-Boost)

* [2.数学类](#2-数学类)

    * [2.1.Eigen](#21-Eigen)

    * [2.2.ceres-solver](#22-ceres-solver)

    * [2.3.g2o](#23-g2o)

* [3.机器人类](#3-机器人类)

    * [3.1.PCL](#31-PCL)

    * [3.2.ROS/ROS2](#32-ROS/ROS2)

* [4.音频类](#4-音频类)

    * [4.1.groove](#41-groove)

    * [4.2.OpenAL](#42-OpenAL)

* [5.AI类](#5-AI类)

    * [5.1.BehaviorTree-CPP](#51-BehaviorTree-CPP)

    * [5.2.shark](#52-shark)

    * [5.3.mlpack](#53-mlpack)

    * [5.4.libtorch](#54-libtorch)

    * [5.5.OpenCV](#55-OpenCV)

    * [5.6.OpenVINO](#56-OpenVINO)

* [6.参数和日志](#6-参数和日志)

    * [6.1.spdlog](61-spdlog)

    * [6.2.yaml-cpp](62-yaml-cpp)

    * [6.3.nlohmann-json](63-nlohmann-json)

# 框架类

## Qt

## Boost



# 数学类

## Eigen

经典的线性代数库了，这个在有关机器人的所有方向上使用范围都是非常广的，除了图像级矩阵直接使用OpenCV之外，矩阵运算全用的是它。

* **矩阵定义**

```cpp
#include <Eigen/Dense>

Matrix<double, 3, 3> A;               // Fixed rows and cols. Same as Matrix3d.
Matrix<double, 3, Dynamic> B;         // Fixed rows, dynamic cols.
Matrix<double, Dynamic, Dynamic> C;   // Full dynamic. Same as MatrixXd.
Matrix<double, 3, 3, RowMajor> E;     // Row major; default is column-major.
Matrix3f P, Q, R;                     // 3x3 float matrix.
Vector3f x, y, z;                     // 3x1 float matrix.
RowVector3f a, b, c;                  // 1x3 float matrix.
VectorXd v;                           // Dynamic column vector of doubles
// Eigen          // Matlab           // comments
x.size()          // length(x)        // vector size
C.rows()          // size(C,1)        // number of rows
C.cols()          // size(C,2)        // number of columns
x(i)              // x(i+1)           // Matlab is 1-based
C(i,j)            // C(i+1,j+1)       //
```

* ** Eigen 基础使用**

```cpp
// Basic usage
// Eigen        // Matlab           // comments
x.size()        // length(x)        // vector size
C.rows()        // size(C,1)        // number of rows
C.cols()        // size(C,2)        // number of columns
x(i)            // x(i+1)           // Matlab is 1-based
C(i, j)         // C(i+1,j+1)       //

A.resize(4, 4);   // Runtime error if assertions are on.
B.resize(4, 9);   // Runtime error if assertions are on.
A.resize(3, 3);   // Ok; size didn't change.
B.resize(3, 9);   // Ok; only dynamic cols changed.
                  
A << 1, 2, 3,     // Initialize A. The elements can also be
     4, 5, 6,     // matrices, which are stacked along cols
     7, 8, 9;     // and then the rows are stacked.
B << A, A, A;     // B is three horizontally stacked A's.
A.fill(10);       // Fill A with all 10's.
```

* **Eigen 特殊矩阵生成**

```cpp
// Eigen                            // Matlab
MatrixXd::Identity(rows,cols)       // eye(rows,cols)
C.setIdentity(rows,cols)            // C = eye(rows,cols)
MatrixXd::Zero(rows,cols)           // zeros(rows,cols)
C.setZero(rows,cols)                // C = ones(rows,cols)
MatrixXd::Ones(rows,cols)           // ones(rows,cols)
C.setOnes(rows,cols)                // C = ones(rows,cols)
MatrixXd::Random(rows,cols)         // rand(rows,cols)*2-1        // MatrixXd::Random returns uniform random numbers in (-1, 1).
C.setRandom(rows,cols)              // C = rand(rows,cols)*2-1
VectorXd::LinSpaced(size,low,high)  // linspace(low,high,size)'
v.setLinSpaced(size,low,high)       // v = linspace(low,high,size)'
```

* **Eigen 矩阵分块**

```cpp
// Matrix slicing and blocks. All expressions listed here are read/write.
// Templated size versions are faster. Note that Matlab is 1-based (a size N
// vector is x(1)...x(N)).
// Eigen                           // Matlab
x.head(n)                          // x(1:n)
x.head<n>()                        // x(1:n)
x.tail(n)                          // x(end - n + 1: end)
x.tail<n>()                        // x(end - n + 1: end)
x.segment(i, n)                    // x(i+1 : i+n)
x.segment<n>(i)                    // x(i+1 : i+n)
P.block(i, j, rows, cols)          // P(i+1 : i+rows, j+1 : j+cols)
P.block<rows, cols>(i, j)          // P(i+1 : i+rows, j+1 : j+cols)
P.row(i)                           // P(i+1, :)
P.col(j)                           // P(:, j+1)
P.leftCols<cols>()                 // P(:, 1:cols)
P.leftCols(cols)                   // P(:, 1:cols)
P.middleCols<cols>(j)              // P(:, j+1:j+cols)
P.middleCols(j, cols)              // P(:, j+1:j+cols)
P.rightCols<cols>()                // P(:, end-cols+1:end)
P.rightCols(cols)                  // P(:, end-cols+1:end)
P.topRows<rows>()                  // P(1:rows, :)
P.topRows(rows)                    // P(1:rows, :)
P.middleRows<rows>(i)              // P(i+1:i+rows, :)
P.middleRows(i, rows)              // P(i+1:i+rows, :)
P.bottomRows<rows>()               // P(end-rows+1:end, :)
P.bottomRows(rows)                 // P(end-rows+1:end, :)
P.topLeftCorner(rows, cols)        // P(1:rows, 1:cols)
P.topRightCorner(rows, cols)       // P(1:rows, end-cols+1:end)
P.bottomLeftCorner(rows, cols)     // P(end-rows+1:end, 1:cols)
P.bottomRightCorner(rows, cols)    // P(end-rows+1:end, end-cols+1:end)
P.topLeftCorner<rows,cols>()       // P(1:rows, 1:cols)
P.topRightCorner<rows,cols>()      // P(1:rows, end-cols+1:end)
P.bottomLeftCorner<rows,cols>()    // P(end-rows+1:end, 1:cols)
P.bottomRightCorner<rows,cols>()   // P(end-rows+1:end, end-cols+1:end)
```

* **Eigen 矩阵元素交换**

```cpp
// Of particular note is Eigen's swap function which is highly optimized.
// Eigen                           // Matlab
R.row(i) = P.col(j);               // R(i, :) = P(:, i)
R.col(j1).swap(mat1.col(j2));      // R(:, [j1 j2]) = R(:, [j2, j1])
```

* **Eigen 矩阵转置**

```cpp
// Views, transpose, etc; all read-write except for .adjoint().
// Eigen                           // Matlab
R.adjoint()                        // R'
R.transpose()                      // R.' or conj(R')
R.diagonal()                       // diag(R)
x.asDiagonal()                     // diag(x)
R.transpose().colwise().reverse(); // rot90(R)
R.conjugate()                      // conj(R)
```

* **Eigen 矩阵乘积**

```cpp
// All the same as Matlab, but matlab doesn't have *= style operators.
// Matrix-vector.  Matrix-matrix.   Matrix-scalar.
y  = M*x;          R  = P*Q;        R  = P*s;
a  = b*M;          R  = P - Q;      R  = s*P;
a *= M;            R  = P + Q;      R  = P/s;
                   R *= Q;          R  = s*P;
                   R += Q;          R *= s;
                   R -= Q;          R /= s;
```

* **Eigen 矩阵单个元素操作**

```cpp
// Vectorized operations on each element independently
// Eigen                  // Matlab
R = P.cwiseProduct(Q);    // R = P .* Q
R = P.array() * s.array();// R = P .* s
R = P.cwiseQuotient(Q);   // R = P ./ Q
R = P.array() / Q.array();// R = P ./ Q
R = P.array() + s.array();// R = P + s
R = P.array() - s.array();// R = P - s
R.array() += s;           // R = R + s
R.array() -= s;           // R = R - s
R.array() < Q.array();    // R < Q
R.array() <= Q.array();   // R <= Q
R.cwiseInverse();         // 1 ./ P
R.array().inverse();      // 1 ./ P
R.array().sin()           // sin(P)
R.array().cos()           // cos(P)
R.array().pow(s)          // P .^ s
R.array().square()        // P .^ 2
R.array().cube()          // P .^ 3
R.cwiseSqrt()             // sqrt(P)
R.array().sqrt()          // sqrt(P)
R.array().exp()           // exp(P)
R.array().log()           // log(P)
R.cwiseMax(P)             // max(R, P)
R.array().max(P.array())  // max(R, P)
R.cwiseMin(P)             // min(R, P)
R.array().min(P.array())  // min(R, P)
R.cwiseAbs()              // abs(P)
R.array().abs()           // abs(P)
R.cwiseAbs2()             // abs(P.^2)
R.array().abs2()          // abs(P.^2)
(R.array() < s).select(P,Q);  // (R < s ? P : Q)
```

* **Eigen 矩阵化简**

```cpp
// Reductions.
int r, c;
// Eigen                  // Matlab
R.minCoeff()              // min(R(:))
R.maxCoeff()              // max(R(:))
s = R.minCoeff(&r, &c)    // [s, i] = min(R(:)); [r, c] = ind2sub(size(R), i);
s = R.maxCoeff(&r, &c)    // [s, i] = max(R(:)); [r, c] = ind2sub(size(R), i);
R.sum()                   // sum(R(:))
R.colwise().sum()         // sum(R)
R.rowwise().sum()         // sum(R, 2) or sum(R')'
R.prod()                  // prod(R(:))
R.colwise().prod()        // prod(R)
R.rowwise().prod()        // prod(R, 2) or prod(R')'
R.trace()                 // trace(R)
R.all()                   // all(R(:))
R.colwise().all()         // all(R)
R.rowwise().all()         // all(R, 2)
R.any()                   // any(R(:))
R.colwise().any()         // any(R)
R.rowwise().any()         // any(R, 2)
```

* **Eigen 矩阵点乘**

```cpp
// Dot products, norms, etc.
// Eigen                  // Matlab
x.norm()                  // norm(x).    Note that norm(R) doesn't work in Eigen.
x.squaredNorm()           // dot(x, x)   Note the equivalence is not true for complex
x.dot(y)                  // dot(x, y)
x.cross(y)                // cross(x, y) Requires #include <Eigen/Geometry>
```

* **Eigen 矩阵类型转换**

```cpp
//// Type conversion
// Eigen                           // Matlab
A.cast<double>();                  // double(A)
A.cast<float>();                   // single(A)
A.cast<int>();                     // int32(A)
A.real();                          // real(A)
A.imag();                          // imag(A)
// if the original type equals destination type, no work is done
```

* **Eigen 求解线性方程组 Ax = b**

```cpp
// Solve Ax = b. Result stored in x. Matlab: x = A \ b.
x = A.ldlt().solve(b));  // A sym. p.s.d.    #include <Eigen/Cholesky>
x = A.llt() .solve(b));  // A sym. p.d.      #include <Eigen/Cholesky>
x = A.lu()  .solve(b));  // Stable and fast. #include <Eigen/LU>
x = A.qr()  .solve(b));  // No pivoting.     #include <Eigen/QR>
x = A.svd() .solve(b));  // Stable, slowest. #include <Eigen/SVD>
// .ldlt() -> .matrixL() and .matrixD()
// .llt()  -> .matrixL()
// .lu()   -> .matrixL() and .matrixU()
// .qr()   -> .matrixQ() and .matrixR()
// .svd()  -> .matrixU(), .singularValues(), and .matrixV()
```

* **Eigen 矩阵特征值**

```cpp
// Eigenvalue problems
// Eigen                          // Matlab
A.eigenvalues();                  // eig(A);
EigenSolver<Matrix3d> eig(A);     // [vec val] = eig(A)
eig.eigenvalues();                // diag(val)
eig.eigenvectors();               // vec
// For self-adjoint matrices use SelfAdjointEigenSolver<>
```

## ceres-solver

机器人的最经典的非线性优化库，可以直接移步到：http://ceres-solver.org/

最经典的场景，如：$f(x) = Asin(\omega x + \phi)$已知$x_i和y_i$，其中$y_i$为观测值，拟合其余参数。

## g2o

目前未使用过(甚至Demo)，同样的都是优化库，但是它本身基于图优化。

# 机器人类

## PCL

## ROS/ROS2

ROS的链接：https://www.ros.org/

ROS2的链接：https://docs.ros.org/en/rolling/index.html

集成度不错的机器人系统库

# 音频类

## groove

## OpenAL

# AI类

## BehaviorTree-CPP

官方使用的行为树框架了属于是，参考：https://www.behaviortree.dev/

它似乎比较喜欢使用xml来预架构行为树，是个比较规范但是非常麻烦的方式

类型和实现来说相对比较全面，对于偏大型的行为层，可以使用这组第三方库，如果行为树规模较小，不推荐使用

## shark

## mlpack

## libtorch

## OpenCV

## OpenVINO

# 参数和日志

## spdlog

## yaml-cpp

## nlohmann-json