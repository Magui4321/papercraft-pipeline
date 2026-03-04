/**
 * @file unfolding.cpp
 * @brief LSCM UV unfolding implementation from scratch.
 */

#include "unfolding.h"
#include "utils.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <stb_image_write.h>

#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

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
        double a2d=0.5*std::abs(u0.x()*u1.y()-u0.y()*u1.x());
        if(a3d<1e-14) continue;
        double ratio=a2d/(a3d+1e-14);
        if(ratio<1e-12) ratio=1e-12;
        total+=std::abs(std::log(ratio));
        cnt++;
    }
    return cnt>0?total/cnt:0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// unfold_patches
// ─────────────────────────────────────────────────────────────────────────────

std::vector<UnfoldResult>
unfold_patches(const std::vector<Patch>& patches, const Config& cfg)
{
    int n=static_cast<int>(patches.size());
    std::vector<UnfoldResult> results(n);

#ifdef _OPENMP
    int threads = cfg.threads > 0 ? cfg.threads : omp_get_max_threads();
    omp_set_num_threads(threads);
    #pragma omp parallel for schedule(dynamic)
#else
    (void)cfg;
#endif
    for (int i=0;i<n;i++){
        const Patch& p=patches[i];
        UnfoldResult r;
        r.patch_id=p.id;
        r.V=p.V;
        r.F=p.F;

        if (p.V.rows()<3||p.F.rows()<1){
            r.UV=Eigen::MatrixXd::Zero(p.V.rows(),2);
        } else {
            r.UV=lscm_eigen(p.V, p.F);
            if (r.UV.rows()!=p.V.rows())
                r.UV=planar_projection(p.V,p.F);
        }
        r.distortion=compute_arap_proxy(r.V, r.F, r.UV);
        results[i]=std::move(r);
    }
    return results;
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
