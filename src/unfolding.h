#pragma once
/**
 * @file unfolding.h
 * @brief UV unfolding via LSCM (Least Squares Conformal Maps) built from scratch.
 */

#include "mesh_loader.h"
#include "segmentation.h"
#include "config.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

/**
 * @brief UV unfolding result for a single patch.
 */
struct UnfoldResult {
    int             patch_id  = 0;
    Eigen::MatrixXd UV;          ///< UV coordinates (nV × 2)
    Eigen::MatrixXd V;           ///< 3D vertex positions (nV × 3)
    Eigen::MatrixXi F;           ///< Face indices (nF × 3, 0-based into V/UV)
    double          distortion = 0.0; ///< ARAP distortion proxy
};

/**
 * @brief LSCM unfolding implemented from scratch using Eigen sparse solvers.
 *
 * Sets up the LSCM conformality constraints (Lévy et al. 2002), pins two
 * boundary vertices and solves the resulting sparse linear system with
 * SimplicialLDLT.
 *
 * @param V  Vertex positions (nV × 3).
 * @param F  Face index array (nF × 3).
 * @return   UV coordinates (nV × 2).
 */
Eigen::MatrixXd lscm_eigen(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);

/**
 * @brief PCA-based planar projection fallback unfolding.
 * @param V  Vertex positions (nV × 3).
 * @param F  Face index array (nF × 3, unused but kept for API symmetry).
 * @return   UV coordinates in [0,1]² (nV × 2).
 */
Eigen::MatrixXd planar_projection(const Eigen::MatrixXd& V,
                                   const Eigen::MatrixXi& F);

/**
 * @brief Compute an ARAP-inspired distortion proxy (area-ratio log deviation).
 * @param V   3D vertex positions.
 * @param F   Face indices.
 * @param UV  UV coordinates.
 * @return    Mean absolute log area-ratio distortion over all faces.
 */
double compute_arap_proxy(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const Eigen::MatrixXd& UV);

/**
 * @brief Unfold all patches using LSCM (parallelised with OpenMP if available).
 * @param patches  Segmented patches.
 * @param cfg      Pipeline configuration.
 * @return         Vector of UnfoldResult objects, one per patch.
 */
std::vector<UnfoldResult>
unfold_patches(const std::vector<Patch>& patches, const Config& cfg);

/**
 * @brief Render all UV layouts packed into one 1024×1024 PNG.
 * @param results   Unfolding results.
 * @param out_path  Output PNG path.
 */
void render_uv_layout_png(const std::vector<UnfoldResult>& results,
                           const std::string& out_path);
