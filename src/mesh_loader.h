#pragma once
/**
 * @file mesh_loader.h
 * @brief Mesh loading, repair, analysis and preview rendering.
 */

#include <string>
#include <vector>

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/IO/MeshIO.hh>

/// Primary mesh type used throughout the pipeline.
using PaperMesh = OpenMesh::TriMesh_ArrayKernelT<>;

/**
 * @brief Aggregate statistics about a mesh.
 */
struct MeshStats {
    int    n_vertices       = 0;
    int    n_edges          = 0;
    int    n_faces          = 0;
    double bbox_diag        = 0.0;
    double avg_edge_length  = 0.0;
    bool   is_manifold      = false;
    bool   has_boundary     = false;
    int    n_boundary_loops = 0;
    int    n_components     = 0;
};

/**
 * @brief Load a mesh from file, stripping materials and removing degenerate faces.
 * @param path Path to the input mesh file (OBJ, STL, PLY, …).
 * @return Loaded PaperMesh.
 * @throws std::runtime_error if the file cannot be read.
 */
PaperMesh load_mesh(const std::string& path);

/**
 * @brief Repair a mesh in-place: remove isolated vertices and recompute normals.
 * @param mesh Mesh to repair.
 */
void repair_mesh(PaperMesh& mesh);

/**
 * @brief Extract only the largest connected component of a mesh.
 * @param mesh Input mesh.
 * @return New mesh containing only the largest component.
 */
PaperMesh largest_component(const PaperMesh& mesh);

/**
 * @brief Compute mesh statistics (vertex/face/edge counts, bbox, etc.).
 * @param mesh Input mesh.
 * @return Populated MeshStats struct.
 */
MeshStats compute_stats(const PaperMesh& mesh);

/**
 * @brief Save a mesh to an OBJ file.
 * @param mesh  Mesh to save.
 * @param path  Output file path.
 * @throws std::runtime_error if writing fails.
 */
void save_mesh(const PaperMesh& mesh, const std::string& path);

/**
 * @brief Render a shaded wireframe preview using a software orthographic rasteriser.
 *
 * The projection direction is along (1,1,1). The image is written as an 800×800 RGBA PNG.
 *
 * @param mesh     Mesh to render.
 * @param out_path Destination PNG path.
 */
void render_mesh_png(const PaperMesh& mesh, const std::string& out_path);
