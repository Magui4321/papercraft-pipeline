#pragma once
/**
 * @file simplification.h
 * @brief Fold-aware mesh simplification using OpenMesh Decimater + QEM.
 */

#include "mesh_loader.h"
#include "config.h"
#include <vector>

/**
 * @brief Detect fold edges — interior edges whose dihedral angle exceeds a threshold.
 * @param mesh              Input mesh (normals will be recomputed internally).
 * @param angle_thresh_deg  Dihedral angle threshold in degrees.
 * @return Vector of edge handles classified as fold edges.
 */
std::vector<OpenMesh::EdgeHandle>
detect_fold_edges(const PaperMesh& mesh, double angle_thresh_deg);

/**
 * @brief Simplify a mesh with the QEM decimater while trying to preserve fold edges.
 *
 * Uses a retry loop: if the fold-preservation ratio drops below
 * cfg.fold_preserve_ratio the target face count is relaxed and decimation is
 * retried (up to 3 attempts).
 *
 * @param mesh  Mesh modified in-place.
 * @param cfg   Pipeline configuration.
 * @return Fraction of original fold edges still present after decimation.
 */
double fold_aware_simplify(PaperMesh& mesh, const Config& cfg);

/**
 * @brief Render mesh preview with fold edges highlighted in red.
 * @param mesh        Mesh to render.
 * @param fold_edges  Fold edge handles (drawn in red).
 * @param out_path    Output PNG path.
 */
void render_foldlines_png(const PaperMesh& mesh,
                          const std::vector<OpenMesh::EdgeHandle>& fold_edges,
                          const std::string& out_path);
