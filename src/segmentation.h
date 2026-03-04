#pragma once
/**
 * @file segmentation.h
 * @brief Spectral mesh segmentation into disk-topology patches.
 */

#include "mesh_loader.h"
#include "config.h"

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Sparse>

/**
 * @brief A single mesh patch — a connected subset of the mesh faces.
 */
struct Patch {
    int              id = 0;
    std::vector<int> face_indices;   ///< Indices into the original mesh's face array
    std::vector<int> vertex_indices; ///< Indices into the original mesh's vertex array
    Eigen::MatrixXd  V;              ///< Vertex positions (nV × 3, local copy)
    Eigen::MatrixXi  F;              ///< Face index array (nF × 3, 0-based into V)
};

/**
 * @brief Build a weighted face-adjacency matrix.
 *
 * Interior edge weight is @p dihedral_weight for fold edges and 1.0 otherwise.
 *
 * @param mesh             Input mesh.
 * @param fold_edges       Edges classified as folds.
 * @param dihedral_weight  Weight assigned to fold edges.
 * @return Symmetric sparse adjacency matrix of size nF × nF.
 */
Eigen::SparseMatrix<double>
build_face_adjacency_matrix(const PaperMesh& mesh,
                             const std::vector<OpenMesh::EdgeHandle>& fold_edges,
                             double dihedral_weight);

/**
 * @brief Compute spectral embedding of faces via the normalised graph Laplacian.
 * @param adj  Face adjacency matrix (nF × nF).
 * @param k    Number of eigenvectors to keep.
 * @return Matrix of shape (nF, k) with spectral coordinates.
 */
Eigen::MatrixXd
compute_spectral_embedding(const Eigen::SparseMatrix<double>& adj, int k);

/**
 * @brief K-means clustering with k-means++ initialisation (Lloyd's algorithm).
 * @param data      Data matrix (n × d).
 * @param k         Number of clusters.
 * @param max_iter  Maximum Lloyd iterations.
 * @return Per-row cluster labels in [0, k).
 */
std::vector<int>
kmeans_cluster(const Eigen::MatrixXd& data, int k, int max_iter = 100);

/**
 * @brief Normal-variation distortion proxy per cluster (lower = better).
 * @param mesh    Mesh with face normals.
 * @param labels  Per-face cluster labels.
 * @param k       Number of clusters.
 * @return Mean distortion value over all clusters.
 */
double
segmentation_distortion_proxy(const PaperMesh& mesh,
                               const std::vector<int>& labels, int k);

/**
 * @brief Find the elbow in a distortion curve via maximum second derivative.
 * @param distortions  Distortion value for k = 2, 3, …
 * @return Index into @p distortions of the elbow (corresponds to k = index+2).
 */
int find_elbow(const std::vector<double>& distortions);

/**
 * @brief Full segmentation pipeline: spectral embedding → k-means → patches.
 * @param mesh        Input (simplified) mesh.
 * @param cfg         Pipeline configuration.
 * @param fold_edges  Fold edges for adjacency weighting.
 * @return Vector of Patch objects.
 */
std::vector<Patch>
segment_mesh(const PaperMesh& mesh, const Config& cfg,
             const std::vector<OpenMesh::EdgeHandle>& fold_edges);

/**
 * @brief Ensure a patch has disk topology (exactly one boundary loop).
 *
 * If the patch has more than one boundary loop the function removes faces
 * that bridge disconnected boundary components until only one loop remains.
 *
 * @param patch      Patch to fix (modified in-place).
 * @param orig_mesh  Original mesh (used for position lookup).
 */
void ensure_disk_topology(Patch& patch, const PaperMesh& orig_mesh);

/**
 * @brief Render each patch in a distinct colour and write a PNG.
 * @param mesh      Original mesh.
 * @param patches   Segmentation result.
 * @param out_path  Output PNG path.
 */
void render_patches_png(const PaperMesh& mesh,
                         const std::vector<Patch>& patches,
                         const std::string& out_path);

/**
 * @brief Render the k-vs-distortion elbow curve as a PNG bar chart.
 * @param distortions  Distortion values (k = 2, 3, …).
 * @param elbow        Elbow index (highlighted).
 * @param out_path     Output PNG path.
 */
void render_elbow_png(const std::vector<double>& distortions,
                      int elbow,
                      const std::string& out_path);
