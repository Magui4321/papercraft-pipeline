/**
 * @file simplification.cpp
 * @brief Fold-aware QEM mesh simplification — Constraint-Plane approach.
 */

#include "simplification.h"
#include "utils.h"

// OpenMesh decimater infrastructure
#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModBaseT.hh>

// OpenMesh quadric type
#include <OpenMesh/Core/Geometry/QuadricT.hh>

#include <Eigen/Dense>
#include <stb_image_write.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>
#include <chrono>

// ─────────────────────────────────────────────────────────────────────────────
// detect_fold_edges
// ─────────────────────────────────────────────────────────────────────────────

FoldEdgeSet detect_fold_edges(const PaperMesh& mesh, double angle_thresh_deg)
{
    FoldEdgeSet result;
    const double thresh_rad = deg2rad(angle_thresh_deg);

    // Need mutable mesh for normal computation
    PaperMesh& m = const_cast<PaperMesh&>(mesh);
    m.request_face_normals();
    m.update_face_normals();

    double angle_sum = 0.0;
    int    fold_count = 0;

    for (auto eh : mesh.edges()) {
        if (mesh.is_boundary(eh)) continue;
        auto heh0 = mesh.halfedge_handle(eh, 0);
        auto heh1 = mesh.halfedge_handle(eh, 1);
        auto fh0  = mesh.face_handle(heh0);
        auto fh1  = mesh.face_handle(heh1);
        if (!fh0.is_valid() || !fh1.is_valid()) continue;

        auto n0 = m.normal(fh0);
        auto n1 = m.normal(fh1);
        double dot   = clamp(static_cast<double>(n0.dot(n1)), -1.0, 1.0);
        double angle = std::acos(dot);

        if (angle > thresh_rad) {
            result.edge_indices.insert(eh.idx());
            angle_sum  += angle;
            fold_count++;
        }
    }

    result.total_detected      = result.edge_indices.size();
    result.mean_dihedral_angle = fold_count > 0 ? angle_sum / fold_count : 0.0;
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// FoldEdgeSet::preservation_ratio
// ─────────────────────────────────────────────────────────────────────────────

double FoldEdgeSet::preservation_ratio(const PaperMesh& simplified,
                                        const PaperMesh& /*original*/) const
{
    if (total_detected == 0) return 1.0;

    // Re-detect fold edges on the simplified mesh using half of the stored
    // mean dihedral angle as the detection threshold.  This avoids relying on
    // stale edge indices (which are reassigned after garbage_collection) while
    // still capturing the same class of high-dihedral edges.
    //
    // mean_dihedral_angle is in radians; convert to degrees for detect_fold_edges.
    // We use half the mean angle as the threshold so that edges near the original
    // mean are still counted.
    const double thresh_deg = (mean_dihedral_angle > 0.0)
                             ? rad2deg(mean_dihedral_angle * 0.5)
                             : 30.0; // fallback
    FoldEdgeSet post = detect_fold_edges(simplified, thresh_deg);
    return static_cast<double>(post.total_detected) /
           static_cast<double>(total_detected);
}

// ─────────────────────────────────────────────────────────────────────────────
// FoldEdgeSet::save_json
// ─────────────────────────────────────────────────────────────────────────────

void FoldEdgeSet::save_json(const std::string& path) const
{
    nlohmann::json j;
    j["total_detected"]      = total_detected;
    j["mean_dihedral_angle"] = mean_dihedral_angle;
    std::vector<int> idx(edge_indices.begin(), edge_indices.end());
    std::sort(idx.begin(), idx.end());
    j["edge_indices"] = idx;
    std::ofstream f(path);
    if (f.is_open()) f << j.dump(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// SimplificationResult helpers
// ─────────────────────────────────────────────────────────────────────────────

void SimplificationResult::print() const
{
    std::cout << "[Stage 2] Fold-aware simplification (Constraint-Plane QEM)\n"
              << "  faces before  : " << faces_before  << "\n"
              << "  faces after   : " << faces_after   << "\n"
              << "  fold preserve : " << fold_preservation_ratio << "\n"
              << "  retries needed: " << retries_needed << "\n"
              << "  elapsed ms    : " << elapsed_ms << "\n";
}

void SimplificationResult::save_metrics_json(const std::string& path) const
{
    nlohmann::json j;
    j["faces_before"]           = faces_before;
    j["faces_after"]            = faces_after;
    j["fold_preservation_ratio"] = fold_preservation_ratio;
    j["retries_needed"]         = retries_needed;
    j["elapsed_ms"]             = elapsed_ms;
    std::ofstream f(path);
    if (f.is_open()) f << j.dump(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// === CONTRIBUTION: Fold-Aware Simplification (Constraint-Plane QEM) ===
//
// Standard QEM (Garland & Heckbert 1997) minimises distance to adjacent
// planes, treating all edges equally.  This destroys high-dihedral fold lines
// that are critical for papercraft assembly.
//
// Instead of a naive retry loop we implement a custom OpenMesh Decimater
// module that modifies the fundamental Quadric Error Metric mathematics.
//
// Algorithm (Fictitious Constraint Planes):
//
//  1. For every vertex, accumulate the standard plane-distance quadric Q_v.
//
//  2. For every fold edge (e in fold_edges):
//     a. Compute the edge direction vector  d  (unit).
//     b. Average the normals of the two adjacent faces to obtain  n_avg.
//     c. Define the constraint-plane normal
//            c_n = d × n_avg   (perpendicular to both edge and face normal).
//        This plane passes through the edge and has its normal pointing away
//        from the fold fan — collapses that pull vertices off the fold incur
//        maximal cost, while sliding along the edge costs nothing.
//     d. Form the fundamental plane quadric   K_fp = p * p^T
//        where  p = (c_n.x, c_n.y, c_n.z, -c_n·edge_midpoint).
//     e. Add  cfg.dihedral_weight * K_fp  to Q_v for both endpoint vertices.
//
//  3. Run standard edge-collapse decimation (single pass, no retries) using
//     these modified quadrics.
//
// Mathematical justification:
//   Edge collapses that move vertices ALONG the fold line are unpenalised
//   because the constraint plane normal is perpendicular to the edge.
//   Collapses that pull vertices AWAY from the fold line violate the
//   constraint plane and incur a massive error cost proportional to
//   cfg.dihedral_weight.  The decimater will naturally preserve the papercraft
//   skeleton without arbitrary thresholding or post-hoc retries.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Convenience: OpenMesh quadric type used throughout.
using Quadricd = OpenMesh::Geometry::Quadricd;

/**
 * @brief Custom OpenMesh Decimater module implementing Constraint-Plane QEM.
 *
 * Inherits ModBaseT so that it integrates seamlessly into the standard
 * OpenMesh Decimater pipeline.  On initialisation it computes modified
 * per-vertex quadrics; the collapse_priority() method evaluates the standard
 * QEM cost using these augmented quadrics.
 */
template <class MeshT>
class ModFoldAwareQuadricT : public OpenMesh::Decimater::ModBaseT<MeshT>
{
public:
    DECIMATING_MODULE(ModFoldAwareQuadricT, MeshT, FoldAwareQuadric);

    // ── Construction ─────────────────────────────────────────────────────

    explicit ModFoldAwareQuadricT(MeshT& mesh)
        : OpenMesh::Decimater::ModBaseT<MeshT>(mesh, false),
          max_err_(std::numeric_limits<double>::max())
    {}

    // ── Configuration setters ────────────────────────────────────────────

    void set_fold_edges(const FoldEdgeSet& fe) { fold_indices_ = fe.edge_indices; }
    void set_dihedral_weight(double w)          { dihedral_weight_ = w; }
    void unset_max_err()                        { max_err_ = std::numeric_limits<double>::max(); }

    // ── OpenMesh Decimater interface ─────────────────────────────────────

    /**
     * @brief Called once before decimation begins; builds per-vertex quadrics.
     *
     * For each vertex we accumulate:
     *   Q_v  = sum of plane quadrics of incident faces   (standard QEM)
     *        + sum of constraint quadrics for incident fold edges
     */
    void initialize() override
    {
        MeshT& m = OpenMesh::Decimater::ModBaseT<MeshT>::mesh();

        // Request a vertex property to store the augmented quadrics
        if (!m.get_property_handle(vquadric_, "FoldAwareQuadric_q"))
            m.add_property(vquadric_, "FoldAwareQuadric_q");

        m.request_face_normals();
        m.update_face_normals();

        // --- Step 1: standard face-plane quadrics -------------------------
        for (auto vh : m.vertices())
            m.property(vquadric_, vh).clear();

        for (auto fh : m.faces()) {
            // Plane equation:  ax + by + cz + d = 0
            auto heh = m.halfedge_handle(fh);
            auto p0  = m.point(m.from_vertex_handle(heh));
            auto p1  = m.point(m.to_vertex_handle(heh));
            auto p2  = m.point(m.to_vertex_handle(m.next_halfedge_handle(heh)));

            // Face normal (not normalised by OpenMesh normal()) needs explicit calc
            OpenMesh::Vec3d e0(p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]);
            OpenMesh::Vec3d e1(p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]);
            OpenMesh::Vec3d n = e0 % e1;   // cross product
            double len = n.norm();
            if (len < 1e-14) continue;
            n /= len;

            double d = -(n | OpenMesh::Vec3d(p0[0], p0[1], p0[2]));  // dot product
            Quadricd Kf(n[0], n[1], n[2], d);

            // Distribute plane quadric to all three vertices
            for (auto vit = m.cfv_begin(fh); vit != m.cfv_end(fh); ++vit)
                m.property(vquadric_, *vit) += Kf;
        }

        // --- Step 2: fictitious constraint-plane quadrics for fold edges --
        //
        // For each fold edge we synthesise an extra constraint plane whose
        // normal lies in the plane perpendicular to the edge direction.
        // Vertices that lie on the fold line incur zero penalty; vertices
        // displaced laterally incur a cost proportional to dihedral_weight_.
        for (auto eh : m.edges()) {
            if (!fold_indices_.count(eh.idx())) continue;
            if (m.is_boundary(eh))              continue;

            auto heh0 = m.halfedge_handle(eh, 0);
            auto heh1 = m.halfedge_handle(eh, 1);
            auto fh0  = m.face_handle(heh0);
            auto fh1  = m.face_handle(heh1);
            if (!fh0.is_valid() || !fh1.is_valid()) continue;

            // Edge direction
            auto vfrom = m.from_vertex_handle(heh0);
            auto vto   = m.to_vertex_handle(heh0);
            auto pf    = m.point(vfrom);
            auto pt    = m.point(vto);
            OpenMesh::Vec3d edge_vec(pt[0]-pf[0], pt[1]-pf[1], pt[2]-pf[2]);
            double elen = edge_vec.norm();
            if (elen < 1e-14) continue;
            OpenMesh::Vec3d edge_dir = edge_vec / elen;

            // Average face normal
            auto fn0 = m.normal(fh0);
            auto fn1 = m.normal(fh1);
            OpenMesh::Vec3d n_avg(fn0[0]+fn1[0], fn0[1]+fn1[1], fn0[2]+fn1[2]);
            double nlen = n_avg.norm();
            if (nlen < 1e-14) continue;
            n_avg /= nlen;

            // Constraint-plane normal: perpendicular to both edge and avg normal
            //   c_n = edge_dir × n_avg
            OpenMesh::Vec3d c_n = edge_dir % n_avg;
            double clen = c_n.norm();
            if (clen < 1e-14) continue;
            c_n /= clen;

            // Plane passes through midpoint of edge
            OpenMesh::Vec3d mid((pf[0]+pt[0])*0.5, (pf[1]+pt[1])*0.5, (pf[2]+pt[2])*0.5);
            double d_c = -(c_n | mid);

            // Fundamental quadric of constraint plane, scaled by dihedral_weight_
            Quadricd Kc(c_n[0], c_n[1], c_n[2], d_c);
            Kc *= dihedral_weight_;

            // Add to both endpoints — these vertices now resist lateral displacement
            m.property(vquadric_, vfrom) += Kc;
            m.property(vquadric_, vto)   += Kc;
        }
    }

    /**
     * @brief Evaluate collapse priority (lower = better = executed first).
     *
     * We compute the optimal collapse position (minimiser of the summed
     * quadric) and return its error as the priority.  Collapses whose
     * minimum error exceeds max_err_ are rejected.
     */
    float collapse_priority(const OpenMesh::Decimater::CollapseInfoT<MeshT>& ci) override
    {
        MeshT& m = OpenMesh::Decimater::ModBaseT<MeshT>::mesh();

        // Combined quadric at the resulting vertex
        Quadricd Q = m.property(vquadric_, ci.v0)
                   + m.property(vquadric_, ci.v1);

        // Evaluate error at the surviving vertex position v1.
        // QuadricT::operator() takes a vector object — use OpenMesh::Vec3d.
        auto    p1  = m.point(ci.v1);
        OpenMesh::Vec3d p(p1[0], p1[1], p1[2]);
        double err = Q(p);

        // Additional penalty if both endpoints are fold vertices and the
        // collapse would create a degenerate angle — already handled by the
        // constraint quadrics, so this is just the raw QEM cost.
        if (err > max_err_) return float(OpenMesh::Decimater::ModBaseT<MeshT>::ILLEGAL_COLLAPSE);
        return static_cast<float>(err);
    }

    /**
     * @brief Update the quadric of the surviving vertex after a collapse.
     *
     * The new quadric is the sum of the two quadrics at the collapsed pair
     * (standard QEM update rule).
     */
    void postprocess_collapse(const OpenMesh::Decimater::CollapseInfoT<MeshT>& ci) override
    {
        MeshT& m = OpenMesh::Decimater::ModBaseT<MeshT>::mesh();
        m.property(vquadric_, ci.v1) =
            m.property(vquadric_, ci.v0) + m.property(vquadric_, ci.v1);
    }

private:
    OpenMesh::VPropHandleT<Quadricd> vquadric_;
    std::unordered_set<int>           fold_indices_;
    double                            dihedral_weight_ = 1000.0;
    double                            max_err_;
};

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────
// fold_aware_simplify  — public entry point
// ─────────────────────────────────────────────────────────────────────────────

SimplificationResult fold_aware_simplify(const PaperMesh& mesh,
                                         const FoldEdgeSet& fold_edges,
                                         const Config& cfg)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    SimplificationResult result;
    result.faces_before = mesh.n_faces();
    result.retries_needed = 0; // constraint-plane approach needs no retries

    int target = cfg.target_face_count;

    // If already below target, nothing to do
    if (static_cast<int>(mesh.n_faces()) <= target) {
        result.mesh  = mesh;
        result.fold_edges = fold_edges;
        result.faces_after = mesh.n_faces();
        result.fold_preservation_ratio = 1.0;
        auto t1 = Clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        return result;
    }

    // Work on a copy so the caller's mesh is never modified
    PaperMesh work = mesh;
    work.request_face_status();
    work.request_edge_status();
    work.request_vertex_status();
    work.request_halfedge_status();
    work.request_face_normals();
    work.update_face_normals();

    // ── Set up the Constraint-Plane QEM decimater ─────────────────────────
    using Dec = OpenMesh::Decimater::DecimaterT<PaperMesh>;
    using ModHandle = typename ModFoldAwareQuadricT<PaperMesh>::Handle;

    Dec decimater(work);

    ModHandle hmod;
    decimater.add(hmod);

    // Configure the module: inject fold edge set and dihedral weight
    decimater.module(hmod).set_fold_edges(fold_edges);
    decimater.module(hmod).set_dihedral_weight(cfg.dihedral_weight);
    decimater.module(hmod).unset_max_err();

    // initialize() builds the augmented per-vertex quadrics (Step 1+2)
    if (!decimater.initialize()) {
        // Initialisation can fail if no valid collapses exist; return as-is
        result.mesh  = work;
        result.fold_edges = fold_edges;
        result.faces_after = work.n_faces();
        result.fold_preservation_ratio = 1.0;
        auto t1 = Clock::now();
        result.elapsed_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
        return result;
    }

    // Single-pass decimation to target face count (no retries needed)
    decimater.decimate_to_faces(static_cast<size_t>(target));
    work.garbage_collection();

    // ── Compute preservation ratio ────────────────────────────────────────
    // Re-detect fold edges on the simplified mesh and compare count.
    auto post_folds = detect_fold_edges(work, cfg.fold_angle_thresh);
    double ratio;
    if (fold_edges.total_detected == 0) {
        ratio = 1.0;
    } else {
        ratio = static_cast<double>(post_folds.total_detected) /
                static_cast<double>(fold_edges.total_detected);
    }

    result.mesh  = std::move(work);
    result.fold_edges = post_folds;
    result.faces_after = result.mesh.n_faces();
    result.fold_preservation_ratio = ratio;

    auto t1 = Clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// render_foldlines_png  (Bresenham wireframe, fold edges in red)
// ─────────────────────────────────────────────────────────────────────────────

void render_foldlines_png(const PaperMesh& mesh,
                          const FoldEdgeSet& fold_edges,
                          const std::string& out_path)
{
    const int W=800, H=800;
    std::vector<uint8_t> img(W*H*4);
    for (int i=0;i<W*H*4;i+=4){img[i]=240;img[i+1]=240;img[i+2]=240;img[i+3]=255;}

    if (mesh.n_vertices()==0){
        stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
        return;
    }

    Eigen::Vector3d right=Eigen::Vector3d(-1,1,0).normalized();
    Eigen::Vector3d view =Eigen::Vector3d(1,1,1).normalized();
    Eigen::Vector3d up   =view.cross(right).normalized();

    int nv=static_cast<int>(mesh.n_vertices());
    std::vector<double> sx(nv),sy(nv);
    double xmin=1e18,xmax=-1e18,ymin=1e18,ymax=-1e18;
    for (auto vh:mesh.vertices()){
        auto p=mesh.point(vh);
        Eigen::Vector3d pt(p[0],p[1],p[2]);
        sx[vh.idx()]=pt.dot(right); sy[vh.idx()]=pt.dot(up);
        xmin=std::min(xmin,sx[vh.idx()]); xmax=std::max(xmax,sx[vh.idx()]);
        ymin=std::min(ymin,sy[vh.idx()]); ymax=std::max(ymax,sy[vh.idx()]);
    }
    const double margin=0.05;
    double rx=xmax-xmin,ry=ymax-ymin;
    if(rx<1e-12)rx=1.0; if(ry<1e-12)ry=1.0;
    double scale=std::min((W*(1-2*margin))/rx,(H*(1-2*margin))/ry);
    double ox=W*margin-xmin*scale, oy=H*margin-ymin*scale;

    auto ppx=[&](int vi){return (int)(sx[vi]*scale+ox);};
    auto ppy=[&](int vi){return H-1-(int)(sy[vi]*scale+oy);};

    auto draw_line=[&](int x0,int y0,int x1,int y1,uint8_t r,uint8_t g,uint8_t b){
        int dx=std::abs(x1-x0),dy=std::abs(y1-y0);
        int ssx=(x0<x1)?1:-1,ssy=(y0<y1)?1:-1;
        int err=dx-dy;
        while(true){
            if(x0>=0&&x0<W&&y0>=0&&y0<H){int idx=(y0*W+x0)*4;img[idx]=r;img[idx+1]=g;img[idx+2]=b;img[idx+3]=255;}
            if(x0==x1&&y0==y1) break;
            int e2=2*err;
            if(e2>-dy){err-=dy;x0+=ssx;}
            if(e2< dx){err+=dx;y0+=ssy;}
        }
    };

    // All edges grey
    for (auto eh:mesh.edges()){
        auto heh=mesh.halfedge_handle(eh,0);
        int vi=mesh.from_vertex_handle(heh).idx();
        int vj=mesh.to_vertex_handle(heh).idx();
        draw_line(ppx(vi),ppy(vi),ppx(vj),ppy(vj),180,180,180);
    }
    // Fold edges red (draw thick by offset)
    for (auto eh:mesh.edges()){
        if (!fold_edges.edge_indices.count(eh.idx())) continue;
        auto heh=mesh.halfedge_handle(eh,0);
        int vi=mesh.from_vertex_handle(heh).idx();
        int vj=mesh.to_vertex_handle(heh).idx();
        draw_line(ppx(vi),   ppy(vi),   ppx(vj),   ppy(vj),   220,40,40);
        draw_line(ppx(vi)+1, ppy(vi),   ppx(vj)+1, ppy(vj),   220,40,40);
        draw_line(ppx(vi),   ppy(vi)+1, ppx(vj),   ppy(vj)+1, 220,40,40);
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}
