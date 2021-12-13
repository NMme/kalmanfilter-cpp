#include <iostream>
#include "kalman.hpp"

KalmanFilter::KalmanFilter(
    double dt, 
    const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& B,
    const Eigen::MatrixXd& Q,
    const Eigen::MatrixXd& H,
    const Eigen::MatrixXd& R,
    const Eigen::MatrixXd& P)
    : F(F), B(B), Q(Q), H(H), R(R), P0(P),
      m(H.rows()), n(F.rows()), dt(dt), initialized(false),
      x_hat(n), x_hat_new(n), I(n, n)
      {
        I.setIdentity();
      }

KalmanFilter::KalmanFilter() {}

void KalmanFilter::init() {
  // initialize to zero
  x_hat.setZero();
  P = P0;
  t0 = 0;
  t = t0;
  initialized = true;
}

void KalmanFilter::init(double t0, const Eigen::VectorXd& x0) {
  x_hat = x0;
  this->t0 = t0;
  t = t0;
  initialized = true;
}

void KalmanFilter::predict(const Eigen::VectorXd& u) {
  // prediction step based on system model and control input: x_hat_new = F*x_hat + B*u
  x_hat_new = F * x_hat + B * u;
  P = F * P * F.transpose() + Q;
  // do not increase time yet!
}

void KalmanFilter::predict() {
  // prediction step based on system dynamic x_hat_new = F*x_hat
  x_hat_new = F * x_hat;
  P = F * P * F.transpose() + Q;
  // do not increase time yet!
}

void KalmanFilter::update(const Eigen::VectorXd& y) {
  // update step of kalman filter -> measurement residual y is defined as y = z - H*x_hat_new
  // 1. determine optimal Kalman gain -> optimally weighthing innovation and prediction covariance
  K = P * H.transpose() * (H * P * H.transpose() + R).inverse();
  // 2. update a posteriori state estimate
  x_hat_new = x_hat_new + K * y;
  // 3. update a posteriori estimate covariance matrix
  P = (I - K * H) * P;
  // update variables 
  x_hat = x_hat_new;
  t += dt;
}

void KalmanFilter::filter_step(const Eigen::VectorXd& y){
  predict();
  update(y);
}

void KalmanFilter::filter_step(const Eigen::VectorXd& y, const Eigen::VectorXd& u){
  predict(u);
  update(y);
}