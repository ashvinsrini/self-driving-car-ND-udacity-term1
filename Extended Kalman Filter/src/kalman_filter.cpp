#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/* 
 * Please note that the Eigen library does not initialize 
 *   VectorXd or MatrixXd objects with zeros upon creation.
 */

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
  /**
   * TODO: predict the state
   */
    x_ = F_ * x_;
  	MatrixXd Ft = F_.transpose();
  	P_ = F_ * P_ * Ft + Q_;
  
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
   * TODO: update the state by using Kalman Filter equations
   */
  VectorXd z_hat = H_ * x_;
  VectorXd y = z - z_hat;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd P_Ht = P_ * Ht;
  MatrixXd K = P_Ht * Si;
  
    //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
  
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
   * TODO: update the state by using Extended Kalman Filter equations
   */
  
  
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
    
  // Equations for h_func below
  float equation = sqrt(px * px + py * py);
  //check division by zero
  if(equation < .00001) {
    px += .001;
    py += .001;
    equation = sqrt(px * px + py * py);
  }
  float equationTwo = atan2(py,px);  // finding the arctan angle 
  float equationThree = (px*vx+py*vy)/equation;
    
  //Declaring equations here
  VectorXd H_func(3);
  H_func << equation, equationTwo, equationThree;
    
  VectorXd y = z - H_func;
  // Normalizing the angle as given by the project resources guidelines
  // using the hint provided: HINT: when working in radians, you can add 2π2\pi2π or subtract 2π2\pi2π until the angle is within the desired range.
  while (y(1)>M_PI) {
    y(1) -= 2 * M_PI;
  }
  while (y(1)<-M_PI) {
    y(1) += 2 * M_PI;
  }
  
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
    
  //new estimate
  x_ = x_ + (K * y);
  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
