#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  // x' = Fx + u; P' = FP(Ft) + Q
  x_ = F_*x_; // 0 process noise assumed
  MatrixXd Ft = F_.transpose();
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::UpdateLidar(const VectorXd &z) {
  // measurement provided in Cartesian coordinates
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  UpdateHelper(y, H_);
}

void KalmanFilter::UpdateRadar(const VectorXd &z) {
  // measurement provided in Polar coordinates, we use the Jacobian of H instead as H is non-linear
  MatrixXd Hj = Tools::CalculateJacobian(x_);
  double rho = sqrt(x_[0]*x_[0] + x_[1]*x_[1]);
  double phi, rho_dot;

  if (fabs(rho) > 0.001) {
    phi = atan(x_[1]/x_[0]);
    rho_dot = (x_[0]*x_[2]+x_[1]*x_[3])/rho;
  } else {
    phi = 0;
    rho_dot = 0;
  }

  VectorXd z_pred = VectorXd(3);
  z_pred << rho, phi, rho_dot;
  VectorXd y = z - z_pred;
  UpdateHelper(y, Hj);
}

void KalmanFilter::UpdateHelper(const VectorXd &y, const MatrixXd &H) {
  // x = x + (K * y); P = (I - K * H) * P;
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  MatrixXd Ht = H.transpose();
  MatrixXd S = H * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  /* K = Kalman filter gain. Combines prediction uncertainty (P') and measurement uncertainty (R)
   * Note: Kalman filter puts more weight to the value that is more certain (prediction vs measurement (P' vs R))
   */
  MatrixXd K = P_ * Ht * Si; 
  x_ = x_ + (K * y);
  P_ = (I - K * H) * P_;
}
