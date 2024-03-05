#ifndef PNPSOLVER_PNP_SOLVER_H
#define PNPSOLVER_PNP_SOLVER_H

#include <vector>
#include <Eigen/Core>

namespace colpnp {

// Sampling method
enum Sampler {
    RANDOM_SAMPLE = 1,
    WEIGHT_SAMPLE = 2,
};

// RANSAC method
enum Robustor { RANSAC = 1, LORANSAC = 2 };


// PnP Option
struct RansacPnPOption {
    // RANSAC option
    double max_error = 12.0;
    double min_inlier_ratio = 0.1;
    double confidence = 0.99999;
    size_t min_num_trials = 1000;

    // Whether to estimate the focal length.
    bool estimate_focal_length = false;
    size_t num_focal_length_samples = 30;
    double min_focal_length_ratio = 0.2;
    double max_focal_length_ratio = 5;

    // Optimization option
    int max_num_iterations = 100;
    bool refine_focal_length = true;
    bool refine_extra_params = false;
};


// Estimate the camera pose from 2D-3D correspondences and refine pose with all
// inliers
//
// @param points2D       2D image points
// @param points3D       3D world points
// @param camera_model   camera model name, for example: SIMPLE_PINHOLE,
//                       SIMPLE_RADIAL, etc.
// @param params         The focal length, principal point, and extra
// parameters.
// @param options        RANSAC PnP option
// @param qvec           Quaternion (qw, qx, qy, qz) from world to camera
// @param tvec           Translation (x, y, z) from world to camera
// @param mask           Inlier mask
// @param robustor       RANSAC(with p3p) or LORANSAC(p3p and epnp)
// @param sampler        RANSAC sampling method: RANDOM, WEIGHT
//
// @param priors         When using weighted sampler
//
// @return               Whether the solution is usable.
bool sovle_pnp_ransac(const std::vector<Eigen::Vector2d> &points2D,
                      const std::vector<Eigen::Vector3d> &points3D,
                      const std::string &camera_model,
                      const RansacPnPOption& options, 
                      std::vector<double> &params,
                      Eigen::Vector4d &qvec,
                      Eigen::Vector3d &tvec, 
                      size_t &num_inlier,
                      std::vector<char> *mask = nullptr,
                      Robustor robustor = RANSAC,
                      Sampler sampler = RANDOM_SAMPLE,
                      std::vector<double> *priors = nullptr);

}  // namespace colpnp

#endif  // PNPSOLVER_PNP_SOLVER_H
