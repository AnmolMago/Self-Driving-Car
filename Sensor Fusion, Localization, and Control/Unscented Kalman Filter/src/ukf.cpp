#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

UKF::UKF() {
  // ensures initial state will be set before use
  is_initialized_ = false;
  previous_timestamp_ = 0.0;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = n_x_ + 2;

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // initial augmented sigma point matrix
  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // initial predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // weights_ of sigma points
  weights_ = VectorXd(2 * n_aug_ + 1);
  weights_(0) = lambda_/(lambda_ + n_aug_);
  for (int i = 1; i < 2*n_aug_+1; i++) {
    weights_(i) = 0.5/(lambda_ + n_aug_);
  }

  // the current NIS for radar
  NIS_radar_ = 0;

  // the current NIS for laser
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} measurement_pack The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
  if (!is_initialized_) {

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      double rho = measurement_pack.raw_measurements_[0];
      double phi = measurement_pack.raw_measurements_[1];
      double rate = measurement_pack.raw_measurements_[2];

      double px = rho * cos(phi);
      double py = rho * sin(phi);       
      double vx = rate * cos(phi);
      double vy = rate * sin(phi);
      double v = fabs(rate);
      double yaw = atan(phi);
      double yawd = 0.0; 

      if(vx != 0) yaw = vy/vx;
      if (px == 0.0) px = 0.0001;
      if (py == 0.0) py = 0.0001;
      x_ << px, py, v, yaw, yawd;
    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      double px = measurement_pack.raw_measurements_[0];
      double py = measurement_pack.raw_measurements_[1];

      if (px == 0.0) px = 0.0001;
      if (py == 0.0) py = 0.0001;

      x_ << px, py, 0.0, 0.0, 0.0;
    }
    
    previous_timestamp_ = measurement_pack.timestamp_;
    is_initialized_ = true;
    return;
  }
  double microseconds_per_second = 1000000.0;
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / microseconds_per_second;

  GenerateAugmentedSigmaPoints();
  Prediction(dt);
  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    Update(measurement_pack.raw_measurements_, measurement_pack.sensor_type_, 
           NIS_radar_);
  } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    Update(measurement_pack.raw_measurements_, measurement_pack.sensor_type_, 
           NIS_laser_);
  }
  previous_timestamp_ = measurement_pack.timestamp_;
}

/**
 * Generated sigma points with augmented state and covariance 
 * matrix which include (νa, νΨ••)
 */
void UKF::GenerateAugmentedSigmaPoints() {
  //create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_ * std_a_;
  P_aug(6,6) = std_yawdd_ * std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++) {
    Xsig_aug_.col(i + 1) = x_aug + sqrt(lambda_ + n_aug_) * L.col(i);
    Xsig_aug_.col(i + 1 + n_aug_) = x_aug - sqrt(lambda_ + n_aug_) * L.col(i);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //predict sigma points
  Xsig_pred_.fill(0.0);
  for (int i = 0; i< 2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double delta_yaw = Xsig_aug_(4,i);
    double a_noise = Xsig_aug_(5,i);
    double delta_yaw_noise = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(delta_yaw) > 0.001) {
        px_p = p_x + v/delta_yaw * (sin(yaw + delta_yaw * delta_t) - sin(yaw));
        py_p = p_y + v/delta_yaw * (cos(yaw) - cos(yaw + delta_yaw * delta_t));
    } else {
        px_p = p_x + v * delta_t * cos(yaw);
        py_p = p_y + v * delta_t * sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + delta_yaw * delta_t;
    double yawd_p = delta_yaw;

    //add noise
    px_p = px_p + 0.5 * a_noise * delta_t * delta_t * cos(yaw);
    py_p = py_p + 0.5 * a_noise * delta_t * delta_t * sin(yaw);
    v_p = v_p + a_noise * delta_t;

    yaw_p = yaw_p + 0.5 * delta_yaw_noise * delta_t * delta_t;
    yawd_p = yawd_p + delta_yaw_noise * delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  // predict mean
  VectorXd x = VectorXd(n_x_);
  x.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    x = x + weights_(i) * Xsig_pred_.col(i);
  }

  // predicted covariance
  MatrixXd P = MatrixXd(n_x_, n_x_);
  P.fill(0.0);
  for (int i = 0; i < 2*n_aug_+1; i++) {
    VectorXd x_diff = Xsig_pred_.col(i) - x;
    //angle normalization
    while (x_diff(3) > M_PI) x_diff(3) -= 2.0*M_PI;
    while (x_diff(3) < -M_PI) x_diff(3) += 2.0*M_PI;
    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }
  x_ = x;
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser or radar 
 * measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::Update(const VectorXd &z, 
                 const MeasurementPackage::SensorType &measurement_type, 
                 double &NIS_) {
  int n_m = (measurement_type == MeasurementPackage::LASER) ? 2 : 3;
  // initial matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_m, 2 * n_aug_ + 1);
  //transform sigma points into measurement space
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    if (measurement_type == MeasurementPackage::LASER) {
      Zsig(0,i) = Xsig_pred_(0,i);
      Zsig(1,i) = Xsig_pred_(1,i);
    } else {
      // extract values for better readibility
      double p_x = Xsig_pred_(0,i);
      double p_y = Xsig_pred_(1,i);
      double v  = Xsig_pred_(2,i);
      double yaw = Xsig_pred_(3,i);
      double v1 = cos(yaw)*v;
      double v2 = sin(yaw)*v;

      // measurement model
      Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y); //r
      Zsig(1,i) = atan2(p_y,p_x);          //phi
      if (Zsig(0,i) > 0.001) {                               
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / Zsig(0,i);   //r_dot
      } else {
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / 0.001;
      } 
    }
  } 

  // initial mean predicted measurement
  VectorXd z_pred = VectorXd(n_m);
  //mean predicted measurement
  z_pred.fill(0.0);
  for (int i=0; i < 2 * n_aug_ + 1; i++) {
      z_pred += weights_(i) * Zsig.col(i);
  }

  // initial measurement covariance matrix S
  MatrixXd S = MatrixXd(n_m, n_m);  
  //measurement covariance matrix S
  S.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    S += weights_(i) * z_diff * z_diff.transpose();
  }

  //add measurement noise covariance matrix
  MatrixXd R = MatrixXd(n_m, n_m);
  if (measurement_type == MeasurementPackage::LASER) {
    R <<  std_laspx_*std_laspx_, 0,
          0,                     std_laspy_*std_laspy_;
  } else {
    R << std_radr_*std_radr_, 0,                       0,  
         0,                   std_radphi_*std_radphi_, 0,
         0,                   0,                       std_radrd_*std_radrd_;
  }
  S = S + R;
 
  //calculate cross correlation matrix
  MatrixXd Tc = MatrixXd(n_x_, n_m);
  Tc.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //angle normalization
    while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
    while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();

  //residual
  VectorXd z_diff = z - z_pred;

  //angle normalization
  while (z_diff(1)> M_PI) z_diff(1)-=2.*M_PI;
  while (z_diff(1)<-M_PI) z_diff(1)+=2.*M_PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S*K.transpose();

  // Calculate NIS
  NIS_ = z_diff.transpose() * S.inverse() * z_diff;
}
