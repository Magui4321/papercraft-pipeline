#pragma once
/**
 * @file simplification.h
 * @brief Fold-aware mesh simplification using Constraint-Plane QEM.
 */

#include "mesh_loader.h"
#include "config.h"
#include <vector>
#include <unordered_set>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
// FoldEdgeSet — typed container for fold-edge bookkeeping
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Set of fold-edge indices together with aggregate statistics.
 *
 * Replaces the plain `std::vector<OpenMesh::EdgeHandle>` used previously so
 * that downstream code can carry provenance (total detected, mean angle) and
 * call convenience methods without passing the mesh again.
 */
struct FoldEdgeSet {
    std::unordered_set<int> edge_indices;  ///< Indices of fold edges in the mesh
    size_t total_detected   = 0;           ///< Number of fold edges found before simplification
    double mean_dihedral_angle = 0.0;      ///< Mean dihedral angle of detected fold edges (radians)

    /**
     * @brief Fraction of original fold edges still present in @p simplified.
     *
     * Counts how many indices in this set are still valid in @p simplified.
     */
    double preservation_ratio(const PaperMesh& simplified,
                               const PaperMesh& original) const;

    /**
     * @brief Serialise fold-edge statistics to a JSON file.
     * @param path Destination file path.
     */
    void save_json(const std::string& path) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// SimplificationResult
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief All outputs from the fold-aware simplification stage.
 */
struct SimplificationResult {
    PaperMesh    mesh;
    FoldEdgeSet  fold_edges;
    size_t       faces_before          = 0;
    size_t       faces_after           = 0;
    double       fold_preservation_ratio = 1.0;
    int          retries_needed        = 0;   ///< Always 0 with constraint-plane approach
    double       elapsed_ms            = 0.0;

    /** @brief Pretty-print a summary to stdout. */
    void print() const;

    /** @brief Serialise stage metrics to a JSON file. */
    void save_metrics_json(const std::string& path) const;
};

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Detect fold edges — interior edges whose dihedral angle exceeds a threshold.
 * @param mesh              Input mesh (normals will be recomputed internally).
 * @param angle_thresh_deg  Dihedral angle threshold in degrees.
 * @return FoldEdgeSet with indices, count, and mean dihedral angle.
 */
FoldEdgeSet detect_fold_edges(const PaperMesh& mesh, double angle_thresh_deg);

/**
 * @brief Fold-aware simplification using Constraint-Plane QEM.
 *
 * Implements the "Fictitious Constraint Planes" algorithm: rather than a
 * retry loop, we inject extra error quadrics for fold edges so that the
 * standard decimater naturally avoids collapsing them.  This executes in a
 * single pass and needs zero retries.
 *
 * @param mesh        Input mesh (read-only; a working copy is made internally).
 * @param fold_edges  Pre-computed fold-edge set.
 * @param cfg         Pipeline configuration (uses dihedral_weight & target_face_count).
 * @return SimplificationResult containing the new mesh and metrics.
 */
SimplificationResult fold_aware_simplify(const PaperMesh& mesh,
                                         const FoldEdgeSet& fold_edges,
                                         const Config& cfg);

/**
 * @brief Render mesh preview with fold edges highlighted in red.
 * @param mesh        Mesh to render.
 * @param fold_edges  Fold-edge set (drawn in red).
 * @param out_path    Output PNG path.
 */
void render_foldlines_png(const PaperMesh& mesh,
                          const FoldEdgeSet& fold_edges,
                          const std::string& out_path);
