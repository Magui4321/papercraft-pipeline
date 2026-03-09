/**
 * @file test_math.cpp
 * @brief Catch2 v3 unit tests for math pinch-points.
 *
 * Tests cover:
 *   1. Constraint-Plane QEM validates parallel sliding along fold lines.
 *   2. Fiedler vector bisects dumbbell topology.
 *   3. LSCM and ARAP handle coplanar geometry: flat mesh has consistent area ratios.
 *
 * All geometry is constructed in memory — no .obj files are loaded.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "simplification.h"
#include "segmentation.h"
#include "unfolding.h"

#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>
#include <OpenMesh/Core/Geometry/QuadricT.hh>
#include <OpenMesh/Core/Geometry/VectorT.hh>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <cmath>
#include <vector>
#include <unordered_set>

// Convenience alias for the quadric type used in tests
using Quadricd = OpenMesh::Geometry::Quadricd;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a PaperMesh from flat vertex/face arrays
// ─────────────────────────────────────────────────────────────────────────────

static PaperMesh make_mesh(
    const std::vector<std::array<double,3>>& verts,
    const std::vector<std::array<int,3>>&   faces)
{
    PaperMesh m;
    std::vector<PaperMesh::VertexHandle> vhs;
    for (auto& v : verts)
        vhs.push_back(m.add_vertex(PaperMesh::Point(
            static_cast<float>(v[0]),
            static_cast<float>(v[1]),
            static_cast<float>(v[2]))));
    for (auto& f : faces)
        m.add_face(vhs[f[0]], vhs[f[1]], vhs[f[2]]);
    m.request_face_normals();
    m.update_face_normals();
    return m;
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 1 — Constraint-Plane QEM validates parallel sliding
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Construct a V-fold mesh (two triangular faces sharing an edge, forming a
 * 90-degree dihedral angle), verify that the per-vertex constraint-plane
 * quadric penalises lateral displacement while permitting sliding along the
 * fold axis.
 *
 * Geometry:
 *   v0 = (0,0,0),  v1 = (1,0,0)   <- fold edge v0–v1
 *   v2 = (0.5,1,0)                 <- vertex in XY plane
 *   v3 = (0.5,0,1)                 <- vertex in XZ plane
 *   Face 0: v0, v1, v2  (XY plane)
 *   Face 1: v1, v0, v3  (XZ plane) — reversed winding to avoid complex edge
 *   Shared fold edge: v0-v1
 */
TEST_CASE("Constraint-Plane QEM validates parallel sliding")
{
    // --- Build V-fold mesh ------------------------------------------------
    // NOTE: face 1 uses {1,0,3} (reversed winding) so that the shared edge
    // v0-v1 is traversed in opposite directions by the two faces, giving a
    // valid manifold mesh with no complex edges.
    PaperMesh mesh = make_mesh(
        { {0.0, 0.0, 0.0},   // v0
          {1.0, 0.0, 0.0},   // v1
          {0.5, 1.0, 0.0},   // v2
          {0.5, 0.0, 1.0} }, // v3
        { {0, 1, 2},          // face 0 — XY plane (normal ≈ (0,0,1))
          {1, 0, 3} }         // face 1 — XZ plane (normal ≈ (0,-1,0)), reversed winding
    );

    // --- Detect fold edges -----------------------------------------------
    // The dihedral angle between XY (normal ≈ (0,0,1)) and XZ (normal ≈ (0,1,0))
    // planes is 90 degrees. Use a threshold of 45 degrees so the edge is detected.
    FoldEdgeSet fold_edges = detect_fold_edges(mesh, 45.0);
    REQUIRE(fold_edges.total_detected >= 1);

    // --- Build the constraint-plane quadric for v0 -----------------------
    // We replicate the quadric-construction logic from ModFoldAwareQuadricT
    // for unit-testing purposes.
    //
    // Standard face quadrics for v0:
    //   Face 0: v0=(0,0,0), v1=(1,0,0), v2=(0.5,1,0)
    //     e0=(1,0,0), e1=(0.5,1,0), n = e0×e1 = (0,0,1), d=0
    //   Face 1 (with reversed winding): v1=(1,0,0), v0=(0,0,0), v3=(0.5,0,1)
    //     e0=v0-v1=(-1,0,0), e1=v3-v1=(-0.5,0,1), n = e0×e1 = (0,-1,0), d=0
    const double dihedral_weight = 1000.0;
    Quadricd Q_face;
    Q_face.clear();

    // Face 0 plane: z=0, normal=(0,0,1), d=0
    Q_face += Quadricd(0.0, 0.0, 1.0, 0.0);
    // Face 1 plane: y=0, normal=(0,-1,0) (reversed winding of XZ face), d=0
    // Actually with face {1,0,3}: e0=v0-v1=(-1,0,0), e1=v3-v1=(-0.5,0,1)
    //   n = (-1,0,0)×(-0.5,0,1) = (0·1-0·0, 0·(-0.5)-(-1)·1, (-1)·0-0·(-0.5))
    //     = (0, 1, 0)
    Q_face += Quadricd(0.0, 1.0, 0.0, 0.0);

    // Constraint-plane quadric for the fold edge v0–v1:
    //   edge_dir = (1,0,0) (pointing from v0 to v1)
    //   n_avg = avg of (0,0,1) and (0,1,0) = (0,0.5,0.5), unit = (0,1/√2,1/√2)
    //   c_n = edge_dir × n_avg = (1,0,0)×(0,1/√2,1/√2)
    //       = (0·1/√2-0·1/√2, 0·0-1·1/√2, 1·1/√2-0·0) = (0, -1/√2, 1/√2)
    //   midpoint = (0.5, 0, 0)
    //   d_c = -(c_n · mid) = -(0·0.5 + (-1/√2)·0 + 1/√2·0) = 0
    {
        double inv_sqrt2 = 1.0 / std::sqrt(2.0);
        Eigen::Vector3d n_avg(0, inv_sqrt2, inv_sqrt2);
        Eigen::Vector3d edge_dir(1.0, 0.0, 0.0);
        Eigen::Vector3d c_n = edge_dir.cross(n_avg);
        double clen = c_n.norm();
        REQUIRE(clen > 1e-10); // must be non-degenerate

        c_n /= clen;
        double d_c = -c_n.dot(Eigen::Vector3d(0.5, 0.0, 0.0));
        Quadricd Kc(c_n[0], c_n[1], c_n[2], d_c);
        Kc *= dihedral_weight;
        Q_face += Kc;
    }

    // --- Evaluate error for sliding ALONG the fold line ------------------
    // Moving v0 from (0,0,0) to (-0.5,0,0) is a slide along the fold edge.
    // The constraint plane normal is perpendicular to the edge direction,
    // so this displacement incurs only the standard face-plane quadric cost.
    {
        OpenMesh::Vec3d p(-0.5, 0.0, 0.0);
        double e = Q_face(p);
        // The standard face quadrics evaluate to 0 for points on the z=0 and y=0
        // planes at x-axis positions; slide along the edge should be cheap.
        REQUIRE(e < 10.0);
    }

    // --- Evaluate error for displacement PERPENDICULAR to fold planes -----
    // Moving v0 to (0,1,0) moves into the y>0 half-space while staying at z=0.
    // This violates the constraint plane (which has normal (0,-1/√2,1/√2),
    // i.e., y=z), incurring a massive cost proportional to dihedral_weight.
    {
        OpenMesh::Vec3d p(0.0, 1.0, 0.0);
        double e = Q_face(p);
        REQUIRE(e > 100.0); // Moving away from fold should incur large cost
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 2 — Fiedler Vector bisects dumbbell topology
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Build a dumbbell adjacency matrix directly (bypassing mesh geometry) and
 * verify that compute_fiedler_vector correctly separates the two dense clusters.
 *
 * Dumbbell topology:
 *   Left cluster  — nodes 0,1,2 strongly connected (weight 10 each internal edge)
 *   Bridge        — node 3 weakly connecting left and right (weight 1)
 *   Right cluster — nodes 4,5,6 strongly connected (weight 10 each internal edge)
 *
 * The Fiedler vector of the weighted normalised Laplacian should assign
 * the same sign to {0,1,2} and the opposite sign to {4,5,6}.
 */
TEST_CASE("Fiedler Vector bisects dumbbell topology")
{
    // Build the dumbbell adjacency matrix directly
    const int n = 7;
    using T = Eigen::Triplet<double>;
    std::vector<T> trips;

    auto add_edge = [&](int i, int j, double w) {
        trips.push_back(T(i, j, w));
        trips.push_back(T(j, i, w));
    };

    // Left cluster K3: all three pairs strongly connected
    add_edge(0, 1, 10.0);
    add_edge(1, 2, 10.0);
    add_edge(0, 2, 10.0);

    // Bridge: weakly connect left cluster (node 2) to right cluster (node 4)
    // via bridge node 3
    add_edge(2, 3, 1.0);
    add_edge(3, 4, 1.0);

    // Right cluster K3: all three pairs strongly connected
    add_edge(4, 5, 10.0);
    add_edge(5, 6, 10.0);
    add_edge(4, 6, 10.0);

    Eigen::SparseMatrix<double> adj(n, n);
    adj.setFromTriplets(trips.begin(), trips.end());

    // Compute Fiedler vector
    Eigen::VectorXd fiedler = compute_fiedler_vector(adj);
    REQUIRE(fiedler.size() == n);

    // Left cluster nodes {0,1,2} should have the same sign
    double sign01 = fiedler[0] * fiedler[1];
    double sign02 = fiedler[0] * fiedler[2];
    REQUIRE(sign01 > 0.0);
    REQUIRE(sign02 > 0.0);

    // Right cluster nodes {4,5,6} should have the same sign
    double sign45 = fiedler[4] * fiedler[5];
    double sign46 = fiedler[4] * fiedler[6];
    REQUIRE(sign45 > 0.0);
    REQUIRE(sign46 > 0.0);

    // Left and right clusters should have opposite signs
    double sign_across = fiedler[0] * fiedler[4];
    REQUIRE(sign_across < 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 3 — LSCM and ARAP handle coplanar geometry consistently
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A perfectly flat 2×2 grid of triangles in the XY plane.  LSCM should
 * produce a conformal (angle-preserving) mapping; for a flat surface, this
 * is equivalent to a scaled rotation of the 3D coordinates.  The key
 * property is that ALL triangles get the same scale factor, so the per-face
 * area ratios should be consistent (standard deviation near zero).
 *
 * Vertices:
 *   (0,0,0), (1,0,0), (2,0,0)
 *   (0,1,0), (1,1,0), (2,1,0)
 *
 * Faces (4 triangles):
 *   (0,1,3), (1,4,3), (1,2,4), (2,5,4)
 */
TEST_CASE("LSCM and ARAP handle coplanar geometry consistently")
{
    // --- Construct flat grid -----------------------------------------------
    Eigen::MatrixXd V(6, 3);
    V << 0,0,0,
         1,0,0,
         2,0,0,
         0,1,0,
         1,1,0,
         2,1,0;

    Eigen::MatrixXi F(4, 3);
    F << 0,1,3,
         1,4,3,
         1,2,4,
         2,5,4;

    // --- LSCM unfolding ---------------------------------------------------
    Eigen::MatrixXd UV = lscm_eigen(V, F);

    // LSCM must return a UV matrix of the correct size
    REQUIRE(UV.rows() == V.rows());
    REQUIRE(UV.cols() == 2);

    // --- ARAP consistency check -------------------------------------------
    // For a flat surface, LSCM gives a conformal (uniform-scale) map.
    // All triangles get the same area ratio → the per-face |log(ratio)|
    // values should be equal (standard deviation near 0).
    std::vector<double> per_face_ratios;
    for (int f = 0; f < F.rows(); f++) {
        int a = F(f,0), b = F(f,1), c = F(f,2);
        Eigen::Vector3d e0 = V.row(b) - V.row(a);
        Eigen::Vector3d e1 = V.row(c) - V.row(a);
        double a3d = 0.5 * e0.cross(e1).norm();

        Eigen::Vector2d u0 = UV.row(b) - UV.row(a);
        Eigen::Vector2d u1 = UV.row(c) - UV.row(a);
        double a2d = 0.5 * std::abs(u0(0)*u1(1) - u0(1)*u1(0));

        if (a3d > 1e-14)
            per_face_ratios.push_back(a2d / a3d);
    }

    REQUIRE(static_cast<int>(per_face_ratios.size()) == F.rows());

    // Compute standard deviation of area ratios
    double mean_r = 0.0;
    for (double r : per_face_ratios) mean_r += r;
    mean_r /= per_face_ratios.size();

    double var = 0.0;
    for (double r : per_face_ratios) var += (r - mean_r) * (r - mean_r);
    var /= per_face_ratios.size();
    double std_dev = std::sqrt(var);

    // All faces should have the same area ratio (std dev near 0)
    REQUIRE(std_dev == Catch::Approx(0.0).margin(1e-4));

    // Also verify that the overall distortion is finite and non-negative
    double distortion = compute_arap_proxy(V, F, UV);
    REQUIRE(distortion >= 0.0);
    REQUIRE(std::isfinite(distortion));
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 4 — normalize_uv_area scales UV to match 3D surface area
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Verify that after calling normalize_uv_area, the total 2D UV area equals
 * the total 3D surface area, and that a subsequent compute_arap_proxy call
 * on a flat mesh returns a distortion close to zero.
 *
 * We use a single unit right-angle triangle:
 *   V3: (0,0,0), (1,0,0), (0,1,0)  → area3D = 0.5
 *   UV:  (0,0),  (0.1,0), (0,0.1) → area2D = 0.005  (deliberately wrong scale)
 *
 * After normalize_uv_area, area2D should equal area3D (= 0.5).
 */
TEST_CASE("normalize_uv_area scales UV to match 3D surface area")
{
    Eigen::MatrixXd V3(3, 3);
    V3 << 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    // Deliberately mis-scaled UV (10× smaller than it should be)
    Eigen::MatrixXd UV(3, 2);
    UV << 0.0, 0.0,
          0.1, 0.0,
          0.0, 0.1;

    // 3D area = 0.5
    double area3d_expected = 0.5;

    // 2D area before normalisation = 0.5 * |0.1*0.1 - 0*0| = 0.005
    {
        Eigen::Vector2d u0 = UV.row(1) - UV.row(0);
        Eigen::Vector2d u1 = UV.row(2) - UV.row(0);
        double area2d_before = 0.5 * std::abs(u0(0)*u1(1) - u0(1)*u1(0));
        REQUIRE(area2d_before == Catch::Approx(0.005).margin(1e-10));
    }

    normalize_uv_area(V3, F, UV);

    // After normalisation, 2D area must equal 3D area
    {
        Eigen::Vector2d u0 = UV.row(1) - UV.row(0);
        Eigen::Vector2d u1 = UV.row(2) - UV.row(0);
        double area2d_after = 0.5 * std::abs(u0(0)*u1(1) - u0(1)*u1(0));
        REQUIRE(area2d_after == Catch::Approx(area3d_expected).margin(1e-8));
    }

    // For a flat mesh with matching areas the ARAP distortion must be ~0
    double distortion = compute_arap_proxy(V3, F, UV);
    REQUIRE(distortion == Catch::Approx(0.0).margin(1e-6));
}

/**
 * Edge-case: normalize_uv_area must not modify UV when 2D area is zero.
 */
TEST_CASE("normalize_uv_area does not modify UV when 2D area is zero")
{
    Eigen::MatrixXd V3(3, 3);
    V3 << 0.0, 0.0, 0.0,
          1.0, 0.0, 0.0,
          0.0, 1.0, 0.0;

    Eigen::MatrixXi F(1, 3);
    F << 0, 1, 2;

    // Degenerate UV: all points collapsed to the origin → area2D = 0
    Eigen::MatrixXd UV = Eigen::MatrixXd::Zero(3, 2);
    Eigen::MatrixXd UV_original = UV;

    normalize_uv_area(V3, F, UV);

    // UV must remain unchanged
    REQUIRE((UV - UV_original).norm() == Catch::Approx(0.0).margin(1e-15));
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 5 — count_boundaries correctly identifies boundary loops
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A flat open strip reports exactly one boundary loop.
 * A cylindrical band (two open ends, with an intermediate interior ring so
 * that the two boundaries are not directly edge-adjacent) reports two.
 *
 * Two-ring cylinder layout (12 vertices, 16 triangles):
 *   Bottom ring (boundary): v0-v3
 *   Middle ring (interior): v4-v7
 *   Top    ring (boundary): v8-v11
 */
TEST_CASE("count_boundaries correctly identifies boundary loops")
{
    // ── Case 1: single flat open patch → 1 boundary loop ─────────────────
    PaperMesh disk = make_mesh(
        { {0,0,0}, {1,0,0}, {2,0,0}, {0,1,0}, {1,1,0} },
        { {0,1,3}, {1,4,3}, {1,2,4} });

    REQUIRE(count_boundaries(disk) == 1);

    // ── Case 2: two-ring cylinder → 2 boundary loops ─────────────────────
    PaperMesh cyl = make_mesh(
        { {1,0,0},{0,1,0},{-1,0,0},{0,-1,0},      // bottom: v0-v3
          {1,0,.5},{0,1,.5},{-1,0,.5},{0,-1,.5},   // middle: v4-v7
          {1,0,1},{0,1,1},{-1,0,1},{0,-1,1} },     // top:    v8-v11
        { // lower half (bottom -> middle)
          {0,1,5},{0,5,4}, {1,2,6},{1,6,5}, {2,3,7},{2,7,6}, {3,0,4},{3,4,7},
          // upper half (middle -> top)
          {4,5,9},{4,9,8}, {5,6,10},{5,10,9}, {6,7,11},{6,11,10}, {7,4,8},{7,8,11} });

    REQUIRE(count_boundaries(cyl) == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// TEST 6 — enforce_disk_topology merges two boundary loops into one
// ─────────────────────────────────────────────────────────────────────────────

/**
 * After enforce_disk_topology the two-ring cylinder must have exactly one
 * boundary loop.  Every edge in the result must be shared by at most two
 * faces (manifold condition), and all faces must have three distinct vertices.
 */
TEST_CASE("enforce_disk_topology merges two boundary loops into one")
{
    // Two-ring cylinder: bottom loop (v0-v3), interior ring (v4-v7),
    // top loop (v8-v11).  Dijkstra finds a path through the interior ring.
    PaperMesh cyl = make_mesh(
        { {1,0,0},{0,1,0},{-1,0,0},{0,-1,0},
          {1,0,.5},{0,1,.5},{-1,0,.5},{0,-1,.5},
          {1,0,1},{0,1,1},{-1,0,1},{0,-1,1} },
        { {0,1,5},{0,5,4}, {1,2,6},{1,6,5}, {2,3,7},{2,7,6}, {3,0,4},{3,4,7},
          {4,5,9},{4,9,8}, {5,6,10},{5,10,9}, {6,7,11},{6,11,10}, {7,4,8},{7,8,11} });

    REQUIRE(count_boundaries(cyl) == 2);

    enforce_disk_topology(cyl);

    REQUIRE(count_boundaries(cyl) == 1);

    // Manifold check: no edge may have both half-edges on the boundary
    for (auto eh : cyl.edges()) {
        auto heh0 = cyl.halfedge_handle(eh, 0);
        auto heh1 = cyl.halfedge_handle(eh, 1);
        REQUIRE(!(cyl.is_boundary(heh0) && cyl.is_boundary(heh1)));
    }

    // All faces must have three distinct vertices
    for (auto fh : cyl.faces()) {
        std::vector<int> fverts;
        for (auto fv : cyl.fv_range(fh))
            fverts.push_back(fv.idx());
        REQUIRE(fverts.size() == 3);
        REQUIRE(fverts[0] != fverts[1]);
        REQUIRE(fverts[1] != fverts[2]);
        REQUIRE(fverts[0] != fverts[2]);
    }
}
