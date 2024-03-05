#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl_bind.h>

#include "interface/pnp_solver.h"


namespace py = pybind11;

py::dict solve_pnp_ransac_py(
    const Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>>
        points2D,
    const Eigen::Ref<Eigen::Matrix<float, 3, Eigen::Dynamic, Eigen::RowMajor>>
        points3D,
    const Eigen::Ref<Eigen::Matrix<float, 1, Eigen::Dynamic, Eigen::RowMajor>>
        priors,
    const py::dict camera, const py::dict options) {

    assert(points2D.cols() == points3D.cols());
    assert(points3D.cols() == priors.cols());

    std::string camera_model_name = camera["model_name"].cast<std::string>();
    std::vector<double> params = camera["params"].cast<std::vector<double>>();

    std::vector<Eigen::Vector2d> point2D_vec(points2D.cols());
    std::vector<Eigen::Vector3d> point3D_vec(points3D.cols());
    std::vector<double> priors_vec(priors.cols());
    for (size_t i = 0; i != point2D_vec.size(); ++i) {
        point2D_vec[i][0] = static_cast<double>(points2D(0, i));
        point2D_vec[i][1] = static_cast<double>(points2D(1, i));
        point3D_vec[i][0] = static_cast<double>(points3D(0, i));
        point3D_vec[i][1] = static_cast<double>(points3D(1, i));
        point3D_vec[i][2] = static_cast<double>(points3D(2, i));
        priors_vec[i] = static_cast<double>(priors(0, i));
    }

    colpnp::RansacPnPOption opt;
    opt.max_error = options["max_error"].cast<double>();
    opt.min_inlier_ratio = options["min_inlier_ratio"].cast<double>();
    opt.confidence = options["confidence"].cast<double>();
    opt.min_num_trials = options["min_num_trials"].cast<size_t>();
    opt.estimate_focal_length = options["estimate_focal_length"].cast<bool>();
    opt.num_focal_length_samples = options["num_focal_length_samples"].cast<size_t>();
    opt.min_focal_length_ratio = options["min_focal_length_ratio"].cast<double>();
    opt.max_focal_length_ratio = options["max_focal_length_ratio"].cast<double>();
    opt.max_num_iterations = options["max_num_iterations"].cast<int>();
    opt.refine_focal_length = options["refine_focal_length"].cast<bool>();
    opt.refine_extra_params = options["refine_extra_params"].cast<bool>();


    Eigen::Vector4d qvec;
    Eigen::Vector3d tvec;
    std::vector<char> mask;

    colpnp::Robustor robustor = colpnp::RANSAC;
    bool lo = options["local_optimal"].cast<bool>();
    if (lo) {
        robustor = colpnp::LORANSAC;
    }

    py::dict result;
    result["ninlier"] = 0;
    result["mask"] = mask;
    result["qvec"] = qvec;
    result["tvec"] = tvec;
    result["params"] = params;

    size_t num_inliers = 0;
    bool success = colpnp::sovle_pnp_ransac(
        point2D_vec, point3D_vec, camera_model_name, opt, params, qvec, tvec,
        num_inliers, &mask, robustor, colpnp::WEIGHT_SAMPLE, &priors_vec);
    
    if (success) {
        result["ninlier"] = num_inliers;
        result["mask"] = mask;
        result["qvec"] = qvec;
        result["tvec"] = tvec;
        result["params"] = params;
    }
    return result;
}



PYBIND11_MODULE(pypose, m) {
    m.def("solve_pnp_ransac", &solve_pnp_ransac_py);
}