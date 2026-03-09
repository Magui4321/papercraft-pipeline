#pragma once
/**
 * @file unfolding.h
 * @brief UV unfolding via LSCM (Least Squares Conformal Maps) built from scratch,
 *        with Iterative Hierarchical Splitting for high-distortion patches.
 */

#include "mesh_loader.h"
#include "segmentation.h"
#include "config.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <string>

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
 * @brief Aggregated result from the UV unfolding stage.
 */
struct UnfoldingResult {
    std::vector<UnfoldResult> patches;
    int    patches_split        = 0; ///< Patches dynamically bisected due to distortion
    int    patches_using_fallback = 0; ///< Patches that used planar_projection fallback
    double elapsed_ms           = 0.0;

    /** @brief Pretty-print a summary to stdout. */
    void print() const;

    /** @brief Serialise to a JSON file. */
    void save_json(const std::string& path) const;
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
 * @brief Normalise UV coordinates so the total 2D UV area matches the total
 *        3D surface area of the patch.
 *
 * LSCM is conformal but not isometric — it produces an arbitrarily scaled
 * 2D layout.  Calling this function before compute_arap_proxy removes the
 * global scale mismatch so that the distortion proxy compares areas on the
 * same scale.
 *
 * @param V3  3D vertex positions (nV × 3).
 * @param F   Face index array (nF × 3).
 * @param UV  UV coordinates (nV × 2), modified in-place.
 */
void normalize_uv_area(const Eigen::MatrixXd& V3,
                        const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& UV);

/**
 * @brief Count the number of distinct boundary loops in a mesh.
 *
 * Traverses half-edges to identify and count separate, continuous boundary
 * loops. A topological disk has exactly one boundary loop (B=1).
 *
 * @param mesh  Input mesh (must have status attributes requested).
 * @return Number of boundary loops.
 */
int count_boundaries(PaperMesh& mesh);

/**
 * @brief Enforce disk topology on a mesh by unzipping interior seams.
 *
 * If the mesh has more than one boundary loop, this function repeatedly
 * finds the shortest path (by 3D edge length) between two distinct boundary
 * loops and "unzips" the mesh along that path, duplicating interior vertices
 * to merge the two loops into one. Iterates until a single boundary loop
 * remains or a safety iteration limit is hit.
 *
 * @param mesh  Mesh to fix in-place. Must have status attributes requested.
 */
void enforce_disk_topology(PaperMesh& mesh);

/**
 * @brief Main unfolding driver with Iterative Hierarchical Splitting.
 *
 * Implements a queue-based algorithm:
 * 1. Initialise queue from seg.patches.
 * 2. Pop a patch, attempt LSCM, compute distortion.
 * 3. If distortion > cfg.max_distortion_warn AND faces > 4: bisect via
 *    Fiedler vector, push two sub-patches back into queue.
 * 4. Otherwise accept the patch (using planar fallback if LSCM fails or
 *    the patch is tiny).
 *
 * Updates seg.face_labels to reflect any new bisected patches so the 3D
 * preview colours match the final patch layout.
 *
 * @param mesh  Simplified mesh (used for Fiedler adjacency computation).
 * @param seg   Segmentation result (modified in-place for label updates).
 * @param cfg   Pipeline configuration.
 * @return UnfoldingResult with all unfolded patches and split statistics.
 */
UnfoldingResult unfold_patches(const PaperMesh& mesh,
                                SegmentationResult& seg,
                                const Config& cfg);

/**
 * @brief Render all UV layouts packed into one 1024×1024 PNG.
 * @param results   Unfolding results (vector of UnfoldResult).
 * @param out_path  Output PNG path.
 */
void render_uv_layout_png(const std::vector<UnfoldResult>& results,
                           const std::string& out_path);
