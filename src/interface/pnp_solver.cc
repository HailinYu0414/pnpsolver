#include "pnp_solver.h"

#include "base/camera.h"
#include "estimators/pose.h"

namespace colpnp {

bool sovle_pnp_ransac(const std::vector<Eigen::Vector2d> &points2D,
                      const std::vector<Eigen::Vector3d> &points3D,
                      const std::string &camera_model,
                      const RansacPnPOption& options, 
                      std::vector<double> &params, 
                      Eigen::Vector4d &qvec, Eigen::Vector3d &tvec, 
                      size_t &num_inlier,
                      std::vector<char> *mask, Robustor robustor,
                      Sampler sampler, std::vector<double> *priors) {
    // CHECK(points2D.size() == points3D.size());
    if (points2D.size() < 4) {
        return false;
    }

    colmap::AbsolutePoseEstimationOptions pnp_options;
    pnp_options.estimate_focal_length = options.estimate_focal_length;
    pnp_options.num_focal_length_samples = options.num_focal_length_samples;
    pnp_options.min_focal_length_ratio = options.min_focal_length_ratio;
    pnp_options.max_focal_length_ratio = options.max_focal_length_ratio;
    pnp_options.ransac_options.max_error = options.max_error;
    pnp_options.ransac_options.min_inlier_ratio = options.min_inlier_ratio;
    pnp_options.ransac_options.confidence = options.confidence;
    pnp_options.ransac_options.min_num_trials = options.min_num_trials;

    colmap::Camera camera;
    camera.SetModelIdFromName(camera_model);
    camera.SetParams(params);

    num_inlier = 0;

    colmap::RansacSampler abs_pose_sampler;
    if (sampler == RANDOM_SAMPLE) {
        abs_pose_sampler = colmap::RANDOM_SAMPLE;
        // CHECK(priors == nullptr);
    } else if (sampler == WEIGHT_SAMPLE) {
        abs_pose_sampler = colmap::WEIGHT_SAMPLE;
        // CHECK(priors->size() == points3D.size());
    }

    colmap::RansacRobustor abs_pose_robustor;
    if (robustor == RANSAC) {
        abs_pose_robustor = colmap::ROBUSTRER_RANSAC;
    } else if (robustor == LORANSAC) {
        abs_pose_robustor = colmap::ROBUSTER_LORANSAC;
    }

    auto success = colmap::EstimateAbsolutePose(
        pnp_options, points2D, points3D, &qvec, &tvec, &camera, &num_inlier, mask,
        abs_pose_robustor, abs_pose_sampler, *priors);
    if (!success) {
        return false;
    }
    params = camera.Params();

    colmap::AbsolutePoseRefinementOptions refine_options;
    refine_options.refine_focal_length = options.refine_focal_length;
    refine_options.refine_extra_params = options.refine_extra_params;
    refine_options.max_num_iterations = options.max_num_iterations;
    refine_options.print_summary = false;

    success = colmap::RefineAbsolutePose(refine_options, *mask, points2D, points3D,
                                         &qvec, &tvec, &camera);
    if (!success) {
        return false;
    }
    params = camera.Params();
    return true;
}

}  // namespace colpnp