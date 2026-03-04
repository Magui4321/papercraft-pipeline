/**
 * @file segmentation.cpp
 * @brief Spectral segmentation implementation.
 */

#include "segmentation.h"
#include "utils.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <stb_image_write.h>

#include <queue>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// Palette for patch colouring (20 distinct colours)
// ─────────────────────────────────────────────────────────────────────────────
static const uint8_t PATCH_PALETTE[20][3] = {
    {230, 25,  75 }, {60,  180, 75 }, {255, 225, 25 }, {0,   130, 200},
    {245, 130, 48 }, {145, 30,  180}, {70,  240, 240}, {240, 50,  230},
    {210, 245, 60 }, {250, 190, 212}, {0,   128, 128}, {220, 190, 255},
    {154, 99,  36 }, {255, 250, 200}, {128, 0,   0  }, {170, 255, 195},
    {128, 128, 0  }, {255, 215, 180}, {0,   0,   128}, {128, 128, 128}
};

// ─────────────────────────────────────────────────────────────────────────────
// build_face_adjacency_matrix
// ─────────────────────────────────────────────────────────────────────────────

Eigen::SparseMatrix<double>
build_face_adjacency_matrix(const PaperMesh& mesh,
                             const std::vector<OpenMesh::EdgeHandle>& fold_edges,
                             double dihedral_weight)
{
    int nF = static_cast<int>(mesh.n_faces());
    std::unordered_set<int> fold_set;
    for (auto eh : fold_edges) fold_set.insert(eh.idx());

    using T = Eigen::Triplet<double>;
    std::vector<T> trips;

    for (auto eh : mesh.edges()) {
        if (mesh.is_boundary(eh)) continue;
        auto heh0 = mesh.halfedge_handle(eh,0);
        auto heh1 = mesh.halfedge_handle(eh,1);
        auto fh0  = mesh.face_handle(heh0);
        auto fh1  = mesh.face_handle(heh1);
        if (!fh0.is_valid() || !fh1.is_valid()) continue;

        int fi = fh0.idx(), fj = fh1.idx();
        double w = fold_set.count(eh.idx()) ? dihedral_weight : 1.0;
        trips.push_back({fi, fj, w});
        trips.push_back({fj, fi, w});
    }

    Eigen::SparseMatrix<double> adj(nF, nF);
    adj.setFromTriplets(trips.begin(), trips.end());
    return adj;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_spectral_embedding
// ─────────────────────────────────────────────────────────────────────────────

Eigen::MatrixXd
compute_spectral_embedding(const Eigen::SparseMatrix<double>& adj, int k)
{
    int n = adj.rows();
    if (n == 0) return Eigen::MatrixXd(0, k);

    // Degree vector
    Eigen::VectorXd deg = Eigen::VectorXd::Zero(n);
    for (int i = 0; i < n; i++)
        for (Eigen::SparseMatrix<double>::InnerIterator it(adj,i); it; ++it)
            deg[i] += it.value();

    // d^{-1/2}
    Eigen::VectorXd d_inv_sqrt(n);
    for (int i = 0; i < n; i++)
        d_inv_sqrt[i] = (deg[i] > 1e-12) ? 1.0/std::sqrt(deg[i]) : 0.0;

    // Normalised Laplacian L = I - D^{-1/2} A D^{-1/2}
    Eigen::MatrixXd L = Eigen::MatrixXd::Identity(n,n);
    for (int col = 0; col < adj.outerSize(); col++)
        for (Eigen::SparseMatrix<double>::InnerIterator it(adj,col); it; ++it)
            L(it.row(), col) -= d_inv_sqrt[it.row()] * it.value() * d_inv_sqrt[col];

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(L);
    if (solver.info() != Eigen::Success)
        return Eigen::MatrixXd::Random(n, k);

    int k_actual = std::min(k, n);
    // Return k smallest eigenvectors
    return solver.eigenvectors().leftCols(k_actual);
}

// ─────────────────────────────────────────────────────────────────────────────
// kmeans_cluster  (k-means++ init + Lloyd)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<int>
kmeans_cluster(const Eigen::MatrixXd& data, int k, int max_iter)
{
    int n = static_cast<int>(data.rows());
    if (n == 0 || k <= 0) return {};
    if (k >= n) {
        std::vector<int> lbl(n);
        std::iota(lbl.begin(),lbl.end(),0);
        return lbl;
    }

    std::mt19937 rng(42);

    // k-means++ initialisation
    std::vector<int> centers;
    {
        std::uniform_int_distribution<int> uid(0, n-1);
        centers.push_back(uid(rng));

        for (int c = 1; c < k; c++) {
            Eigen::VectorXd dist2(n);
            for (int i = 0; i < n; i++) {
                double d = std::numeric_limits<double>::max();
                for (int ci : centers)
                    d = std::min(d, (data.row(i)-data.row(ci)).squaredNorm());
                dist2[i] = d;
            }
            double total = dist2.sum();
            if (total < 1e-12) { centers.push_back(uid(rng)); continue; }
            std::uniform_real_distribution<double> ud(0.0, total);
            double target = ud(rng), cum = 0.0;
            int chosen = n-1;
            for (int i = 0; i < n; i++) {
                cum += dist2[i];
                if (cum >= target) { chosen = i; break; }
            }
            centers.push_back(chosen);
        }
    }

    // Initialise centroids
    Eigen::MatrixXd centroids(k, data.cols());
    for (int c = 0; c < k; c++) centroids.row(c) = data.row(centers[c]);

    std::vector<int> labels(n, 0);

    for (int iter = 0; iter < max_iter; iter++) {
        // Assignment
        bool changed = false;
        for (int i = 0; i < n; i++) {
            int best = 0;
            double best_d = (data.row(i)-centroids.row(0)).squaredNorm();
            for (int c = 1; c < k; c++) {
                double d = (data.row(i)-centroids.row(c)).squaredNorm();
                if (d < best_d) { best_d=d; best=c; }
            }
            if (labels[i] != best) { labels[i]=best; changed=true; }
        }
        if (!changed) break;

        // Update centroids
        centroids.setZero();
        std::vector<int> cnt(k,0);
        for (int i = 0; i < n; i++) { centroids.row(labels[i]) += data.row(i); cnt[labels[i]]++; }
        for (int c = 0; c < k; c++) if (cnt[c]>0) centroids.row(c) /= cnt[c];
    }
    return labels;
}

// ─────────────────────────────────────────────────────────────────────────────
// segmentation_distortion_proxy
// ─────────────────────────────────────────────────────────────────────────────

double segmentation_distortion_proxy(const PaperMesh& mesh,
                                      const std::vector<int>& labels, int k)
{
    PaperMesh& m = const_cast<PaperMesh&>(mesh);
    m.request_face_normals();
    m.update_face_normals();

    std::vector<Eigen::Vector3d> nsum(k, Eigen::Vector3d::Zero());
    std::vector<int> cnt(k, 0);

    for (auto fh : mesh.faces()) {
        int fi = fh.idx();
        if (fi >= static_cast<int>(labels.size())) continue;
        int lbl = labels[fi];
        auto n = m.normal(fh);
        nsum[lbl] += Eigen::Vector3d(n[0],n[1],n[2]);
        cnt[lbl]++;
    }

    double distortion = 0.0;
    for (int c = 0; c < k; c++) {
        if (cnt[c] == 0) continue;
        double mean_norm = (nsum[c] / cnt[c]).norm();
        distortion += (1.0 - mean_norm);
    }
    return (k > 0) ? distortion / k : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// find_elbow  (maximum second derivative)
// ─────────────────────────────────────────────────────────────────────────────

int find_elbow(const std::vector<double>& distortions)
{
    int n = static_cast<int>(distortions.size());
    if (n < 3) return 0;

    int best = 0;
    double best_curv = -1e18;
    for (int i = 1; i < n-1; i++) {
        double curv = distortions[i-1] - 2*distortions[i] + distortions[i+1];
        if (curv > best_curv) { best_curv=curv; best=i; }
    }
    return best;
}

// ─────────────────────────────────────────────────────────────────────────────
// segment_mesh
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Patch>
segment_mesh(const PaperMesh& mesh, const Config& cfg,
             const std::vector<OpenMesh::EdgeHandle>& fold_edges)
{
    int nF = static_cast<int>(mesh.n_faces());
    if (nF == 0) return {};

    // Build adjacency
    auto adj = build_face_adjacency_matrix(mesh, fold_edges, cfg.dihedral_weight);

    // Determine number of patches
    int k_target = cfg.n_patches;
    std::vector<int> labels;

    if (k_target <= 0) {
        // Auto: try k=2..min(12,nF/20) and pick elbow
        int k_min = 2, k_max = std::max(2, std::min(12, nF/20));
        std::vector<double> distortions;

        // Compute spectral embedding once with max k
        int embed_dim = std::min(k_max+2, nF);
        Eigen::MatrixXd embed = compute_spectral_embedding(adj, embed_dim);

        for (int k = k_min; k <= k_max; k++) {
            int dim = std::min(k+1, static_cast<int>(embed.cols()));
            Eigen::MatrixXd sub = embed.leftCols(dim);
            auto lbl = kmeans_cluster(sub, k);
            double d = segmentation_distortion_proxy(mesh, lbl, k);
            distortions.push_back(d);
        }

        int elbow_idx = find_elbow(distortions);
        k_target = k_min + elbow_idx;

        int dim = std::min(k_target+1, static_cast<int>(embed.cols()));
        labels = kmeans_cluster(embed.leftCols(dim), k_target);
    } else {
        int embed_dim = std::min(k_target+2, nF);
        Eigen::MatrixXd embed = compute_spectral_embedding(adj, embed_dim);
        int dim = std::min(k_target+1, static_cast<int>(embed.cols()));
        labels = kmeans_cluster(embed.leftCols(dim), k_target);
    }

    // Build Patch objects
    // Map original vertex index → local vertex index per patch
    std::unordered_map<int, std::vector<int>> patch_faces; // label → face indices
    for (auto fh : mesh.faces()) {
        int fi = fh.idx();
        int lbl = (fi < static_cast<int>(labels.size())) ? labels[fi] : 0;
        patch_faces[lbl].push_back(fi);
    }

    std::vector<Patch> patches;
    for (auto& [lbl, fids] : patch_faces) {
        Patch p;
        p.id = lbl;
        p.face_indices = fids;

        // Collect unique vertices
        std::unordered_map<int,int> vmap; // original → local
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

        int nv_local = static_cast<int>(p.vertex_indices.size());
        p.V.resize(nv_local, 3);
        for (int i = 0; i < nv_local; i++) {
            auto pt = mesh.point(mesh.vertex_handle(p.vertex_indices[i]));
            p.V.row(i) = Eigen::Vector3d(pt[0],pt[1],pt[2]);
        }

        int nf_local = static_cast<int>(fids.size());
        p.F.resize(nf_local, 3);
        for (int fi = 0; fi < nf_local; fi++) {
            auto fh = mesh.face_handle(fids[fi]);
            int col = 0;
            for (auto fv = mesh.cfv_begin(fh); fv != mesh.cfv_end(fh); ++fv)
                p.F(fi, col++) = vmap.at(fv->idx());
        }

        ensure_disk_topology(p, mesh);
        patches.push_back(std::move(p));
    }

    return patches;
}

// ─────────────────────────────────────────────────────────────────────────────
// ensure_disk_topology
// ─────────────────────────────────────────────────────────────────────────────

void ensure_disk_topology(Patch& patch, const PaperMesh& /*orig_mesh*/)
{
    // Count boundary loops of the local mesh (patch.F, patch.V)
    int nV = patch.V.rows();
    int nF_patch = patch.F.rows();
    if (nV == 0 || nF_patch == 0) return;

    // Build half-edge boundary: map directed edge (a→b) → count
    std::unordered_map<int,int> he_count; // encode as a*MAX+b
    auto encode = [&](int a, int b){ return a*(nV+1)+b; };

    for (int f = 0; f < nF_patch; f++)
        for (int e = 0; e < 3; e++) {
            int a = patch.F(f,e), b = patch.F(f,(e+1)%3);
            he_count[encode(a,b)]++;
        }

    // Boundary half-edges: directed (a→b) whose reverse (b→a) does not exist
    std::unordered_map<int,int> boundary_next; // boundary_next[a] = b  (a→b on boundary)
    for (auto& [key, cnt] : he_count) {
        if (cnt != 1) continue;
        int a = key / (nV+1), b = key % (nV+1);
        boundary_next[a] = b;
    }

    // Count boundary loops
    std::unordered_set<int> visited;
    int loops = 0;
    for (auto& [start, next_vertex] : boundary_next) {
        (void)next_vertex;
        if (visited.count(start)) continue;
        int cur = start;
        do { visited.insert(cur); cur = boundary_next.count(cur) ? boundary_next.at(cur) : cur; }
        while (cur != start && !visited.count(cur));
        loops++;
    }

    if (loops <= 1) return; // already disk topology

    // Multiple loops → trim faces not adjacent to the largest boundary loop
    // Simple strategy: keep only the largest connected face component
    // Re-run connected components on faces
    std::vector<bool> vis_f(nF_patch, false);
    std::vector<std::vector<int>> comps;

    // Build face-face adjacency via shared edges
    std::unordered_map<int, std::vector<int>> edge_to_faces;
    for (int f = 0; f < nF_patch; f++)
        for (int e = 0; e < 3; e++) {
            int a = patch.F(f,e), b = patch.F(f,(e+1)%3);
            int key = std::min(a,b)*(nV+1)+std::max(a,b);
            edge_to_faces[key].push_back(f);
        }

    // Build face adjacency list
    std::vector<std::vector<int>> fadj(nF_patch);
    for (auto& [key, flist] : edge_to_faces)
        for (int i = 0; i < static_cast<int>(flist.size()); i++)
            for (int j = i+1; j < static_cast<int>(flist.size()); j++) {
                fadj[flist[i]].push_back(flist[j]);
                fadj[flist[j]].push_back(flist[i]);
            }

    for (int s = 0; s < nF_patch; s++) {
        if (vis_f[s]) continue;
        std::vector<int> comp;
        std::queue<int> q; q.push(s); vis_f[s]=true;
        while (!q.empty()) {
            int fi=q.front(); q.pop(); comp.push_back(fi);
            for (int nb : fadj[fi]) if (!vis_f[nb]) { vis_f[nb]=true; q.push(nb); }
        }
        comps.push_back(std::move(comp));
    }

    if (comps.size() <= 1) return;

    // Keep only the largest component
    auto& keep = *std::max_element(comps.begin(), comps.end(),
        [](const auto& a, const auto& b){ return a.size()<b.size(); });

    std::unordered_set<int> keep_set(keep.begin(), keep.end());
    std::vector<int> new_fi_list;
    for (int fi : keep) new_fi_list.push_back(patch.face_indices[fi]);

    Eigen::MatrixXi newF(static_cast<int>(keep.size()), 3);
    for (int i = 0; i < static_cast<int>(keep.size()); i++)
        newF.row(i) = patch.F.row(keep[i]);

    patch.face_indices = new_fi_list;
    patch.F = newF;

    // Rebuild vertex list
    std::unordered_map<int,int> old_to_new;
    std::vector<int> new_vert_idx;
    for (int f = 0; f < patch.F.rows(); f++)
        for (int e = 0; e < 3; e++) {
            int v = patch.F(f,e);
            if (!old_to_new.count(v)) {
                old_to_new[v] = static_cast<int>(new_vert_idx.size());
                new_vert_idx.push_back(v);
            }
            patch.F(f,e) = old_to_new[v];
        }

    int new_nv = static_cast<int>(new_vert_idx.size());
    Eigen::MatrixXd newV(new_nv, 3);
    for (int i = 0; i < new_nv; i++) newV.row(i) = patch.V.row(new_vert_idx[i]);

    // Remap vertex_indices
    std::vector<int> new_vis(new_nv);
    for (int i = 0; i < new_nv; i++) new_vis[i] = patch.vertex_indices[new_vert_idx[i]];

    patch.V = newV;
    patch.vertex_indices = new_vis;
}

// ─────────────────────────────────────────────────────────────────────────────
// render_patches_png
// ─────────────────────────────────────────────────────────────────────────────

void render_patches_png(const PaperMesh& mesh,
                         const std::vector<Patch>& patches,
                         const std::string& out_path)
{
    const int W=800, H=800;
    std::vector<uint8_t> img(W*H*4);
    for (int i=0;i<W*H*4;i+=4){img[i]=240;img[i+1]=240;img[i+2]=240;img[i+3]=255;}

    if (mesh.n_vertices()==0){stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);return;}

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

    // Build face→patch colour map
    std::vector<int> face_patch(mesh.n_faces(), -1);
    for (int pi=0;pi<static_cast<int>(patches.size());pi++)
        for (int fi:patches[pi].face_indices)
            if (fi>=0&&fi<static_cast<int>(mesh.n_faces()))
                face_patch[fi]=pi;

    auto fill_tri=[&](int x0,int y0,int x1,int y1,int x2,int y2,uint8_t r,uint8_t g,uint8_t b){
        if(y0>y1){std::swap(y0,y1);std::swap(x0,x1);}
        if(y0>y2){std::swap(y0,y2);std::swap(x0,x2);}
        if(y1>y2){std::swap(y1,y2);std::swap(x1,x2);}
        auto interp=[](int y,int ya,int xa,int yb,int xb){return yb==ya?xa:xa+(xb-xa)*(y-ya)/(yb-ya);};
        auto span=[&](int y,int xa,int xb){
            if(y<0||y>=H)return; if(xa>xb)std::swap(xa,xb);
            xa=std::max(0,xa); xb=std::min(W-1,xb);
            for(int x=xa;x<=xb;x++){int idx=(y*W+x)*4;img[idx]=r;img[idx+1]=g;img[idx+2]=b;img[idx+3]=255;}
        };
        for(int y=y0;y<=y1;y++) span(y,interp(y,y0,x0,y2,x2),interp(y,y0,x0,y1,x1));
        for(int y=y1;y<=y2;y++) span(y,interp(y,y0,x0,y2,x2),interp(y,y1,x1,y2,x2));
    };

    for (auto fh:mesh.faces()){
        int fi=fh.idx(), pi=face_patch[fi];
        int ci=(pi>=0)?pi%20:0;
        const uint8_t* c=PATCH_PALETTE[ci];
        std::vector<int> vids;
        for(auto fv=mesh.cfv_begin(fh);fv!=mesh.cfv_end(fh);++fv) vids.push_back(fv->idx());
        if(vids.size()<3) continue;
        fill_tri(ppx(vids[0]),ppy(vids[0]),ppx(vids[1]),ppy(vids[1]),ppx(vids[2]),ppy(vids[2]),c[0],c[1],c[2]);
    }

    // Dark wireframe
    for(auto eh:mesh.edges()){
        auto heh=mesh.halfedge_handle(eh,0);
        int vi=mesh.from_vertex_handle(heh).idx();
        int vj=mesh.to_vertex_handle(heh).idx();
        int dx=std::abs(ppx(vj)-ppx(vi)),dy=std::abs(ppy(vj)-ppy(vi));
        int x0=ppx(vi),y0=ppy(vi),x1=ppx(vj),y1=ppy(vj);
        int ssx=(x0<x1)?1:-1,ssy=(y0<y1)?1:-1,err=dx-dy;
        while(true){
            if(x0>=0&&x0<W&&y0>=0&&y0<H){int idx=(y0*W+x0)*4;img[idx]=40;img[idx+1]=40;img[idx+2]=40;img[idx+3]=255;}
            if(x0==x1&&y0==y1) break;
            int e2=2*err;
            if(e2>-dy){err-=dy;x0+=ssx;}
            if(e2< dx){err+=dx;y0+=ssy;}
        }
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}

// ─────────────────────────────────────────────────────────────────────────────
// render_elbow_png  (simple bar chart)
// ─────────────────────────────────────────────────────────────────────────────

void render_elbow_png(const std::vector<double>& distortions,
                      int elbow,
                      const std::string& out_path)
{
    const int W=600, H=400;
    std::vector<uint8_t> img(W*H*4, 255);
    for(int i=0;i<W*H*4;i+=4){img[i]=250;img[i+1]=250;img[i+2]=250;img[i+3]=255;}

    int n=static_cast<int>(distortions.size());
    if(n==0){stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);return;}

    double dmax=*std::max_element(distortions.begin(),distortions.end());
    if(dmax<1e-12) dmax=1.0;

    int margin_l=50, margin_b=30, margin_r=20, margin_t=20;
    int plot_w=W-margin_l-margin_r, plot_h=H-margin_t-margin_b;

    double bar_w=static_cast<double>(plot_w)/n;
    for(int i=0;i<n;i++){
        int bh=static_cast<int>(distortions[i]/dmax*plot_h);
        int x0=margin_l+(int)(i*bar_w);
        int x1=margin_l+(int)((i+1)*bar_w)-2;
        int y0=H-margin_b-bh, y1=H-margin_b;
        uint8_t r=100,g=150,b=220;
        if(i==elbow){r=220;g=80;b=80;}
        for(int y=y0;y<y1;y++)
            for(int x=x0;x<=x1&&x<W;x++){
                int idx=(y*W+x)*4;
                img[idx]=r;img[idx+1]=g;img[idx+2]=b;img[idx+3]=255;
            }
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}
