/**
 * @file unfolding.cpp
 * @brief LSCM UV unfolding implementation from scratch,
 *        with Iterative Hierarchical Splitting for high-distortion patches.
 */

#include "unfolding.h"
#include "utils.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <stb_image_write.h>
#include <nlohmann/json.hpp>

#include <igl/cut_to_disk.h>
#include <igl/cut_mesh.h>

#include <map>
#include <set>
#include <queue>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif


// ─────────────────────────────────────────────────────────────────────────────
// planar_projection  (PCA best-fit plane)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd planar_projection(const Eigen::MatrixXd& V,
                                   const Eigen::MatrixXi& /*F*/)
{
    if (V.rows() == 0) return Eigen::MatrixXd(0,2);

    Eigen::Vector3d centroid = V.colwise().mean();
    Eigen::MatrixXd C = V.rowwise() - centroid.transpose();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);
    Eigen::Vector3d u_ax = svd.matrixV().col(0);
    Eigen::Vector3d v_ax = svd.matrixV().col(1);

    Eigen::MatrixXd UV(V.rows(), 2);
    for (int i = 0; i < V.rows(); i++) {
        Eigen::Vector3d pt = V.row(i).transpose() - centroid;
        UV(i,0) = pt.dot(u_ax);
        UV(i,1) = pt.dot(v_ax);
    }

    // Normalise to [0,1]
    double umin=UV.col(0).minCoeff(), umax=UV.col(0).maxCoeff();
    double vmin=UV.col(1).minCoeff(), vmax=UV.col(1).maxCoeff();
    double ur=umax-umin, vr=vmax-vmin;
    if(ur<1e-12) ur=1.0; if(vr<1e-12) vr=1.0;
    UV.col(0)=(UV.col(0).array()-umin)/ur;
    UV.col(1)=(UV.col(1).array()-vmin)/vr;
    return UV;
}

// ─────────────────────────────────────────────────────────────────────────────
// lscm_eigen   (LSCM from scratch — Lévy et al. 2002)
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd lscm_eigen(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F)
{
    int nV = static_cast<int>(V.rows());
    int nF = static_cast<int>(F.rows());

    if (nV < 3 || nF < 1)
        return planar_projection(V, F);

    // ── Find two boundary vertices far apart ──────────────────────────────
    // Build edge-use count to identify boundary edges
    std::map<std::pair<int,int>,int> edge_cnt;
    for (int f = 0; f < nF; f++)
        for (int e = 0; e < 3; e++) {
            int a=F(f,e), b=F(f,(e+1)%3);
            edge_cnt[{std::min(a,b),std::max(a,b)}]++;
        }

    std::set<int> bverts_set;
    for (auto& [key,cnt] : edge_cnt)
        if (cnt==1){ bverts_set.insert(key.first); bverts_set.insert(key.second); }

    int p0=0, p1=1;
    if (!bverts_set.empty()) {
        std::vector<int> bv(bverts_set.begin(), bverts_set.end());
        double best=-1;
        int nb=std::min((int)bv.size(), 30);
        for (int i=0;i<nb;i++)
            for (int j=i+1;j<nb;j++){
                double d=(V.row(bv[i])-V.row(bv[j])).squaredNorm();
                if(d>best){best=d;p0=bv[i];p1=bv[j];}
            }
    }
    if (p0 == p1) { p1 = (p0+1) % nV; }

    // ── Build free-variable index maps ──────────────────────────────────
    // Unknowns: [u_{free0..}, v_{free0..}]  (pinned p0 at (0,0), p1 at (1,0))
    std::vector<int> u_idx(nV,-1), v_idx(nV,-1);
    int cnt_u=0;
    for (int i=0;i<nV;i++) if(i!=p0&&i!=p1) u_idx[i]=cnt_u++;
    int nfu=cnt_u, cnt_v=0;
    for (int i=0;i<nV;i++) if(i!=p0&&i!=p1) v_idx[i]=nfu+cnt_v++;
    int nfree=nfu+cnt_v; // = 2*(nV-2)

    if (nfree==0) {
        Eigen::MatrixXd UV=Eigen::MatrixXd::Zero(nV,2);
        UV(p1,0)=1.0; return UV;
    }

    // ── Assemble system A x = rhs  (2*nF rows, nfree cols) ──────────────
    using SpMat   = Eigen::SparseMatrix<double>;
    using Triplet = Eigen::Triplet<double>;

    std::vector<Triplet> trips;
    trips.reserve(nF*18);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(2*nF);

    const double u_p0=0.0, v_p0=0.0, u_p1=1.0, v_p1=0.0;

    for (int f=0;f<nF;f++){
        int vi=F(f,0), vj=F(f,1), vk=F(f,2);
        Eigen::Vector3d Pi=V.row(vi), Pj=V.row(vj), Pk=V.row(vk);

        Eigen::Vector3d e0=Pj-Pi, e1=Pk-Pi;
        Eigen::Vector3d n=e0.cross(e1);
        double area2=n.norm();
        if(area2<1e-14) continue;

        // Local 2D frame: x along e0, y = n_hat × x
        Eigen::Vector3d x_ax=e0.normalized();
        Eigen::Vector3d n_hat=n/area2;
        Eigen::Vector3d y_ax=n_hat.cross(x_ax); // already unit

        // Local 2D positions: pi=(0,0), pj=(qj,0), pk=(qkx,qky)
        double qj  = e0.norm();
        double qkx = e1.dot(x_ax);
        double qky = e1.dot(y_ax);

        double A2 = area2;   // = 2*area

        // Complex weights W_i=(pk-pj)/(2A), W_j=(pi-pk)/(2A), W_k=(pj-pi)/(2A)
        double wr_i=(qkx-qj)/A2,  wi_i=qky/A2;
        double wr_j=(-qkx)/A2,    wi_j=(-qky)/A2;
        double wr_k=qj/A2,         wi_k=0.0;

        int row_re=2*f, row_im=2*f+1;

        // For each vertex, add its contribution (or move to RHS if pinned)
        auto add=[&](int vert, double wr, double wi){
            if (vert==p0){
                rhs[row_re] -= wr*u_p0 - wi*v_p0;
                rhs[row_im] -= wi*u_p0 + wr*v_p0;
            } else if (vert==p1){
                rhs[row_re] -= wr*u_p1 - wi*v_p1;
                rhs[row_im] -= wi*u_p1 + wr*v_p1;
            } else {
                trips.push_back({row_re, u_idx[vert],  wr});
                trips.push_back({row_re, v_idx[vert], -wi});
                trips.push_back({row_im, u_idx[vert],  wi});
                trips.push_back({row_im, v_idx[vert],  wr});
            }
        };

        add(vi,wr_i,wi_i);
        add(vj,wr_j,wi_j);
        add(vk,wr_k,wi_k);
    }

    SpMat A_mat(2*nF, nfree);
    A_mat.setFromTriplets(trips.begin(),trips.end());

    // ── Normal equations:  (A^T A) x = A^T b ─────────────────────────────
    SpMat ATA = A_mat.transpose() * A_mat;
    Eigen::VectorXd ATb = A_mat.transpose() * rhs;

    Eigen::SimplicialLDLT<SpMat> solver;
    solver.compute(ATA);
    if (solver.info() != Eigen::Success)
        return planar_projection(V,F);

    Eigen::VectorXd x = solver.solve(ATb);
    if (solver.info() != Eigen::Success)
        return planar_projection(V,F);

    // ── Reconstruct UV ────────────────────────────────────────────────────
    Eigen::MatrixXd UV(nV,2);
    for (int i=0;i<nV;i++){
        if      (i==p0){UV(i,0)=u_p0;UV(i,1)=v_p0;}
        else if (i==p1){UV(i,0)=u_p1;UV(i,1)=v_p1;}
        else           {UV(i,0)=x[u_idx[i]];UV(i,1)=x[v_idx[i]];}
    }
    return UV;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_arap_proxy  (mean |log area_ratio| over faces)
// ─────────────────────────────────────────────────────────────────────────────

double compute_arap_proxy(const Eigen::MatrixXd& V,
                           const Eigen::MatrixXi& F,
                           const Eigen::MatrixXd& UV)
{
    if (F.rows()==0) return 0.0;
    double total=0.0; int cnt=0;
    for (int f=0;f<F.rows();f++){
        int a=F(f,0),b=F(f,1),c=F(f,2);
        Eigen::Vector3d e0=V.row(b)-V.row(a), e1=V.row(c)-V.row(a);
        double a3d=0.5*e0.cross(e1).norm();
        Eigen::Vector2d u0=UV.row(b)-UV.row(a), u1=UV.row(c)-UV.row(a);
        double a2d=0.5*std::abs(u0(0)*u1(1)-u0(1)*u1(0));
        if(a3d<1e-14) continue;
        double ratio=a2d/(a3d+1e-14);
        if(ratio<1e-12) ratio=1e-12;
        total+=std::abs(std::log(ratio));
        cnt++;
    }
    return cnt>0?total/cnt:0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// normalize_uv_area  (scale UV so total 2D area == total 3D area)
// ─────────────────────────────────────────────────────────────────────────────

void normalize_uv_area(const Eigen::MatrixXd& V3,
                        const Eigen::MatrixXi& F,
                        Eigen::MatrixXd& UV)
{
    if (F.rows() == 0) return;

    double area3d = 0.0;
    for (int f = 0; f < F.rows(); f++) {
        int a = F(f,0), b = F(f,1), c = F(f,2);
        Eigen::Vector3d e0 = V3.row(b) - V3.row(a);
        Eigen::Vector3d e1 = V3.row(c) - V3.row(a);
        area3d += 0.5 * e0.cross(e1).norm();
    }

    double area2d = 0.0;
    for (int f = 0; f < F.rows(); f++) {
        int a = F(f,0), b = F(f,1), c = F(f,2);
        Eigen::Vector2d u0 = UV.row(b) - UV.row(a);
        Eigen::Vector2d u1 = UV.row(c) - UV.row(a);
        area2d += 0.5 * std::abs(u0(0)*u1(1) - u0(1)*u1(0));
    }

    if (area2d < 1e-8) return;

    double s = std::sqrt(area3d / area2d);
    UV *= s;
}

// ─────────────────────────────────────────────────────────────────────────────
// UnfoldingResult helpers
// ─────────────────────────────────────────────────────────────────────────────

void UnfoldingResult::print() const
{
    std::cout << "[Stage 4] LSCM UV unfolding (Iterative Hierarchical Splitting)\n"
              << "  total patches      : " << patches.size() << "\n"
              << "  patches split      : " << patches_split << "\n"
              << "  patches fallback   : " << patches_using_fallback << "\n"
              << "  elapsed ms         : " << elapsed_ms << "\n";
}

void UnfoldingResult::save_json(const std::string& path) const
{
    nlohmann::json j;
    j["total_patches"]         = patches.size();
    j["patches_split"]         = patches_split;
    j["patches_using_fallback"] = patches_using_fallback;
    j["elapsed_ms"]            = elapsed_ms;
    std::ofstream f(path);
    if (f.is_open()) f << j.dump(2);
}

// ─────────────────────────────────────────────────────────────────────────────
// unfold_patches  — Queue-based Iterative Hierarchical Splitting
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Helper: bisect a patch using the Fiedler vector of its local
 *        face-adjacency graph.
 *
 * @param mesh      Original mesh (for adjacency lookup).
 * @param patch     Patch to bisect.
 * @param fold_edges  Global fold-edge set for weighted adjacency.
 * @param cfg       Pipeline configuration.
 * @param next_id   Next available patch ID (incremented on return).
 * @param[out] left   Faces with Fiedler value <= median.
 * @param[out] right  Faces with Fiedler value > median.
 */
static void bisect_patch_fiedler(
    const PaperMesh& mesh,
    const Patch& patch,
    const FoldEdgeSet& fold_edges,
    const Config& cfg,
    int& next_id,
    Patch& left,
    Patch& right)
{
    // Build local face adjacency for this patch only
    auto local_adj = build_local_face_adjacency(mesh, patch, fold_edges,
                                                 cfg.dihedral_weight);

    // Compute the Fiedler vector
    Eigen::VectorXd fiedler = compute_fiedler_vector(local_adj);

    // Threshold at median value (proper median for even-sized vectors)
    Eigen::VectorXd sorted = fiedler;
    std::sort(sorted.data(), sorted.data() + sorted.size());
    int sz = static_cast<int>(sorted.size());
    double median = (sz % 2 == 1)
                  ? sorted[sz / 2]
                  : (sorted[sz/2 - 1] + sorted[sz/2]) * 0.5;

    // Split face_indices into two halves
    // Global-to-local face index map was implicitly built when constructing
    // local_adj — local index i corresponds to patch.face_indices[i].
    int nLocal = static_cast<int>(patch.face_indices.size());
    std::vector<int> left_fi, right_fi;
    for (int i = 0; i < nLocal; i++) {
        if (fiedler[i] <= median)
            left_fi.push_back(patch.face_indices[i]);
        else
            right_fi.push_back(patch.face_indices[i]);
    }

    // Ensure both halves are non-empty (degenerate: send all to left, clear right)
    if (left_fi.empty() || right_fi.empty()) {
        left  = patch;
        right.face_indices.clear();
        right.id = -1; // sentinel: degenerate bisection, do not enqueue
        return;
    }

    // Helper to rebuild a Patch from a face index list
    auto make_patch = [&](std::vector<int>& fids, int id) -> Patch {
        Patch p;
        p.id = id;
        p.face_indices = fids;

        std::unordered_map<int,int> vmap;
        for (int fi : fids) {
            auto fh = mesh.face_handle(fi);
            for (auto fv = mesh.cfv_begin(fh); fv != mesh.cfv_end(fh); ++fv) {
                int vi = fv->idx();
                if (!vmap.count(vi)) {
                    int li = static_cast<int>(vmap.size());
                    vmap[vi] = li;
                    p.vertex_indices.push_back(vi);
                }
            }
        }

        int nv = static_cast<int>(p.vertex_indices.size());
        p.V.resize(nv, 3);
        for (int i = 0; i < nv; i++) {
            auto pt = mesh.point(mesh.vertex_handle(p.vertex_indices[i]));
            p.V.row(i) = Eigen::Vector3d(pt[0], pt[1], pt[2]);
        }

        int nf = static_cast<int>(fids.size());
        p.F.resize(nf, 3);
        for (int fi = 0; fi < nf; fi++) {
            auto fh = mesh.face_handle(fids[fi]);
            int col = 0;
            for (auto fv = mesh.cfv_begin(fh); fv != mesh.cfv_end(fh); ++fv)
                p.F(fi, col++) = vmap.at(fv->idx());
        }
        return p;
    };

    left  = make_patch(left_fi,  next_id++);
    right = make_patch(right_fi, next_id++);
}

UnfoldingResult unfold_patches(const PaperMesh& mesh,
                                SegmentationResult& seg,
                                const Config& cfg)
{
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    UnfoldingResult result;

    // Re-detect fold edges from the mesh for local adjacency weighting in bisection.
    // The fold_edges in the simplification result are stored on the simplified mesh;
    // we need them here for Fiedler-vector adjacency weighting.
    FoldEdgeSet mesh_fold_edges = detect_fold_edges(mesh, cfg.fold_angle_thresh);

    // Queue-based processing loop
    std::queue<Patch> q;
    for (auto& p : seg.patches)
        q.push(p);

    int next_id = static_cast<int>(seg.patches.size());

    while (!q.empty()) {
        Patch patch = q.front();
        q.pop();

        UnfoldResult r;
        r.patch_id = patch.id;
        r.V = patch.V;
        r.F = patch.F;

        // Step 1: attempt LSCM unfolding
        bool lscm_ok = false;
        if (patch.V.rows() >= 3 && patch.F.rows() >= 1) {
            // ── Topological cut-to-disk: guarantee the patch is a disk before LSCM ──
            {
                Eigen::MatrixXi cuts;
                igl::cut_to_disk(patch.F, cuts);
                if (cuts.rows() > 0) {
                    Eigen::MatrixXd V_cut;
                    Eigen::MatrixXi F_cut;
                    igl::cut_mesh(patch.V, patch.F, cuts, V_cut, F_cut);
                    patch.V = V_cut;
                    patch.F = F_cut;
                    // Also update the UnfoldResult's V and F to match the cut mesh
                    r.V = patch.V;
                    r.F = patch.F;
                }
            }
            r.UV = lscm_eigen(patch.V, patch.F);
            lscm_ok = (r.UV.rows() == patch.V.rows());
        }

        if (!lscm_ok || patch.V.rows() < 3 || patch.F.rows() < 1) {
            // LSCM failed or patch too small — use planar projection fallback
            if (patch.V.rows() > 0)
                r.UV = planar_projection(patch.V, patch.F);
            else
                r.UV = Eigen::MatrixXd(0, 2);
            normalize_uv_area(r.V, r.F, r.UV);
            r.distortion = compute_arap_proxy(r.V, r.F, r.UV);
            result.patches_using_fallback++;
            result.patches.push_back(std::move(r));
            continue;
        }

        // Step 2: measure ARAP distortion
        normalize_uv_area(r.V, r.F, r.UV);
        r.distortion = compute_arap_proxy(r.V, r.F, r.UV);

        // Step 3: if distortion too high AND patch is large enough, bisect
        if (r.distortion > cfg.max_distortion_warn &&
            static_cast<int>(patch.face_indices.size()) > 4)
        {
            std::cout << "[Stage 4] Distortion " << r.distortion
                      << " exceeds threshold. Bisecting patch "
                      << patch.id << ".\n";

            Patch left_patch, right_patch;
            bisect_patch_fiedler(mesh, patch, mesh_fold_edges,
                                  cfg, next_id, left_patch, right_patch);

            if (!right_patch.face_indices.empty()) {
                // Update face labels in SegmentationResult so 3D preview matches
                for (int fi : left_patch.face_indices)
                    if (fi < static_cast<int>(seg.face_labels.size()))
                        seg.face_labels[fi] = left_patch.id;
                for (int fi : right_patch.face_indices)
                    if (fi < static_cast<int>(seg.face_labels.size()))
                        seg.face_labels[fi] = right_patch.id;

                q.push(left_patch);
                q.push(right_patch);
                result.patches_split++;
                continue;
            }
            // Degenerate bisection: fall through and accept as-is
        }

        // Step 4: accept the patch
        result.patches.push_back(std::move(r));
    }

    auto t1 = Clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1-t0).count();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// render_uv_layout_png
// ─────────────────────────────────────────────────────────────────────────────

void render_uv_layout_png(const std::vector<UnfoldResult>& results,
                           const std::string& out_path)
{
    const int W=1024, H=1024;
    std::vector<uint8_t> img(W*H*4);
    for(int i=0;i<W*H*4;i+=4){img[i]=30;img[i+1]=30;img[i+2]=30;img[i+3]=255;}

    static const uint8_t PALETTE[20][3]={
        {230,25,75},{60,180,75},{255,225,25},{0,130,200},{245,130,48},
        {145,30,180},{70,240,240},{240,50,230},{210,245,60},{250,190,212},
        {0,128,128},{220,190,255},{154,99,36},{255,250,200},{128,0,0},
        {170,255,195},{128,128,0},{255,215,180},{0,0,128},{128,128,128}
    };

    int n=static_cast<int>(results.size());
    if(n==0){stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);return;}

    // Grid layout: ceil(sqrt(n)) x ceil(sqrt(n))
    int cols=static_cast<int>(std::ceil(std::sqrt((double)n)));
    int rows=(n+cols-1)/cols;
    double cell_w=static_cast<double>(W)/cols;
    double cell_h=static_cast<double>(H)/rows;

    for(int pi=0;pi<n;pi++){
        const UnfoldResult& r=results[pi];
        if(r.UV.rows()==0) continue;
        int col=pi%cols, row=pi/cols;
        double ox=col*cell_w, oy_base=row*cell_h;
        const double pad=0.05;

        // Find UV bbox
        double umin=r.UV.col(0).minCoeff(),umax=r.UV.col(0).maxCoeff();
        double vmin=r.UV.col(1).minCoeff(),vmax=r.UV.col(1).maxCoeff();
        double ur=umax-umin,vr=vmax-vmin;
        if(ur<1e-12)ur=1.0; if(vr<1e-12)vr=1.0;
        double scale=std::min((cell_w*(1-2*pad))/ur,(cell_h*(1-2*pad))/vr);

        auto tu=[&](double u)->int{return (int)(ox+cell_w*pad+(u-umin)*scale);};
        auto tv=[&](double v)->int{return H-1-(int)(oy_base+cell_h*pad+(v-vmin)*scale);};

        const uint8_t* c=PALETTE[pi%20];

        // Draw UV triangles
        for(int f=0;f<r.F.rows();f++){
            int va=r.F(f,0),vb=r.F(f,1),vc=r.F(f,2);
            // Draw edges
            auto draw=[&](int x0,int y0,int x1,int y1){
                int dx=std::abs(x1-x0),dy=std::abs(y1-y0);
                int sx=(x0<x1)?1:-1,sy=(y0<y1)?1:-1,err=dx-dy;
                while(true){
                    if(x0>=0&&x0<W&&y0>=0&&y0<H){int idx=(y0*W+x0)*4;img[idx]=c[0];img[idx+1]=c[1];img[idx+2]=c[2];img[idx+3]=255;}
                    if(x0==x1&&y0==y1)break;
                    int e2=2*err;
                    if(e2>-dy){err-=dy;x0+=sx;}
                    if(e2< dx){err+=dx;y0+=sy;}
                }
            };
            draw(tu(r.UV(va,0)),tv(r.UV(va,1)),tu(r.UV(vb,0)),tv(r.UV(vb,1)));
            draw(tu(r.UV(vb,0)),tv(r.UV(vb,1)),tu(r.UV(vc,0)),tv(r.UV(vc,1)));
            draw(tu(r.UV(vc,0)),tv(r.UV(vc,1)),tu(r.UV(va,0)),tv(r.UV(va,1)));
        }
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}
