#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;
	if (estimations.size() != ground_truth.size() || ground_truth.size() == 0) {
		std::cout << "Invalid data provided, aborting rmse calculation." << std::endl;
		return rmse;
	}
	for (int i = 0; i < ground_truth.size(); i++) {
		VectorXd error = ground_truth[i] - estimations[i];
		rmse += static_cast<VectorXd>(error.array().square());
	}
	rmse /= ground_truth.size();
	rmse = rmse.array().sqrt();
	return rmse;
}
