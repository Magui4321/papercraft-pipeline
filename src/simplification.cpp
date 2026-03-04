/**
 * @file simplification.cpp
 * @brief Fold-aware QEM mesh simplification.
 */

#include "simplification.h"
#include "utils.h"

#include <OpenMesh/Tools/Decimater/DecimaterT.hh>
#include <OpenMesh/Tools/Decimater/ModQuadricT.hh>

#include <Eigen/Dense>

#include <stb_image_write.h>

#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <iostream>

typedef OpenMesh::Decimater::DecimaterT<PaperMesh>       Decimater;
typedef OpenMesh::Decimater::ModQuadricT<PaperMesh>::Handle HModQuadric;

// ─────────────────────────────────────────────────────────────────────────────
// detect_fold_edges
// ─────────────────────────────────────────────────────────────────────────────

std::vector<OpenMesh::EdgeHandle>
detect_fold_edges(const PaperMesh& mesh, double angle_thresh_deg)
{
    std::vector<OpenMesh::EdgeHandle> result;
    const double thresh_rad = deg2rad(angle_thresh_deg);

    PaperMesh& m = const_cast<PaperMesh&>(mesh);
    m.request_face_normals();
    m.update_face_normals();

    for (auto eh : mesh.edges()) {
        if (mesh.is_boundary(eh)) continue;
        auto heh0 = mesh.halfedge_handle(eh, 0);
        auto heh1 = mesh.halfedge_handle(eh, 1);
        auto fh0  = mesh.face_handle(heh0);
        auto fh1  = mesh.face_handle(heh1);
        if (!fh0.is_valid() || !fh1.is_valid()) continue;

        auto n0 = m.normal(fh0);
        auto n1 = m.normal(fh1);
        double dot = clamp(static_cast<double>(n0.dot(n1)), -1.0, 1.0);
        if (std::acos(dot) > thresh_rad)
            result.push_back(eh);
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// fold_aware_simplify
// ─────────────────────────────────────────────────────────────────────────────

double fold_aware_simplify(PaperMesh& mesh, const Config& cfg)
{
    auto initial_folds = detect_fold_edges(mesh, cfg.fold_angle_thresh);
    int  init_count    = static_cast<int>(initial_folds.size());

    int target = cfg.target_face_count;
    if (static_cast<int>(mesh.n_faces()) <= target)
        return 1.0;

    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_vertex_status();
    mesh.request_halfedge_status();

    double preserve_ratio = 1.0;
    const int max_retries = 3;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        // Work on a copy so we can retry
        PaperMesh work = mesh;
        work.request_face_status();
        work.request_edge_status();
        work.request_vertex_status();
        work.request_halfedge_status();
        work.request_face_normals();
        work.update_face_normals();

        if (static_cast<int>(work.n_faces()) > target) {
            Decimater decimater(work);
            HModQuadric hmod;
            decimater.add(hmod);
            decimater.module(hmod).unset_max_err();
            decimater.initialize();
            decimater.decimate_to_faces(static_cast<size_t>(target));
            work.garbage_collection();
        }

        if (init_count == 0) { mesh = work; return 1.0; }

        auto post_folds  = detect_fold_edges(work, cfg.fold_angle_thresh);
        preserve_ratio   = static_cast<double>(post_folds.size()) / init_count;

        if (preserve_ratio >= cfg.fold_preserve_ratio || attempt == max_retries-1) {
            mesh = work;
            break;
        }
        // Relax target and retry
        target = static_cast<int>(target * 1.5);
        if (target >= static_cast<int>(mesh.n_faces())) {
            // No decimation possible at this relaxed target
            break;
        }
    }
    return preserve_ratio;
}

// ─────────────────────────────────────────────────────────────────────────────
// render_foldlines_png  (Bresenham wireframe, fold edges in red)
// ─────────────────────────────────────────────────────────────────────────────

void render_foldlines_png(const PaperMesh& mesh,
                          const std::vector<OpenMesh::EdgeHandle>& fold_edges,
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
    for (auto eh:fold_edges){
        auto heh=mesh.halfedge_handle(eh,0);
        int vi=mesh.from_vertex_handle(heh).idx();
        int vj=mesh.to_vertex_handle(heh).idx();
        draw_line(ppx(vi),   ppy(vi),   ppx(vj),   ppy(vj),   220,40,40);
        draw_line(ppx(vi)+1, ppy(vi),   ppx(vj)+1, ppy(vj),   220,40,40);
        draw_line(ppx(vi),   ppy(vi)+1, ppx(vj),   ppy(vj)+1, 220,40,40);
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}
