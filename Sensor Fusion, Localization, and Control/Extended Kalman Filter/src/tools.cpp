#include <iostream>
#include <cmath>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0.0, 0.0, 0.0, 0.0;

	if (estimations.size() != ground_truth.size() || ground_truth.size() == 0) {
		std::cout << "Invalid data provided" << std::endl;
		return rmse;
	}
	for (int i = 0; i < ground_truth.size(); i++) {
		VectorXd error = ground_truth[i] - estimations[i];
		// converting to Eigen::Array ensures element-wise multiplication
		rmse += static_cast<VectorXd>(error.array() * error.array());
	}
	rmse /= ground_truth.size();
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
	//x_state = [px, py, vx, vy]
	double px = x_state(0), py = x_state(1);
	double vx = x_state(2), vy = x_state(3);
	MatrixXd jacobian(3,4);

	// Pre-compute constant terms which appear often in the partial derivatives
	const float c1 = px*px + py*py;
	const float c2 = sqrt(c1);
	const float c3 = c1*c2;

	if (fabs(c1) < 0.0001) {
		std::cout << "Calculating Jacobian would result in division by 0." << std::endl;
		jacobian << 0,    0,    0, 0,
		        	1e+9, 1e+9, 0, 0,
		        	0,    0,    0, 0;
		return jacobian;
	}

	jacobian << px/c2,               py/c2,               0,     0,
				-py/c1,              px/c1,               0,     0,
				py*(vx*py-vy*px)/c3, px*(px*vy-py*vx)/c3, px/c2, py/c2;

	return jacobian;
}
