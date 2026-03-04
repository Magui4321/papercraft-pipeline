/**
 * @file mesh_loader.cpp
 * @brief Implementation of mesh loading, repair, statistics and preview rendering.
 */

#include "mesh_loader.h"
#include "utils.h"

#include <OpenMesh/Core/IO/MeshIO.hh>

#include <Eigen/Dense>

#include <stb_image_write.h>

#include <queue>
#include <unordered_set>
#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// Internal rasteriser helpers
// ─────────────────────────────────────────────────────────────────────────────

static void draw_line_img(std::vector<uint8_t>& img, int W, int H,
                           int x0, int y0, int x1, int y1,
                           uint8_t r, uint8_t g, uint8_t b)
{
    int dx = std::abs(x1-x0), dy = std::abs(y1-y0);
    int sx = (x0<x1)?1:-1, sy = (y0<y1)?1:-1;
    int err = dx-dy;
    while (true) {
        if (x0>=0 && x0<W && y0>=0 && y0<H) {
            int idx = (y0*W+x0)*4;
            img[idx]=r; img[idx+1]=g; img[idx+2]=b; img[idx+3]=255;
        }
        if (x0==x1 && y0==y1) break;
        int e2=2*err;
        if (e2>-dy){err-=dy; x0+=sx;}
        if (e2< dx){err+=dx; y0+=sy;}
    }
}

static void fill_tri_img(std::vector<uint8_t>& img, int W, int H,
                          int x0,int y0, int x1,int y1, int x2,int y2,
                          uint8_t r,uint8_t g,uint8_t b)
{
    if (y0>y1){std::swap(y0,y1);std::swap(x0,x1);}
    if (y0>y2){std::swap(y0,y2);std::swap(x0,x2);}
    if (y1>y2){std::swap(y1,y2);std::swap(x1,x2);}

    auto interp=[](int y,int ya,int xa,int yb,int xb)->int{
        if(yb==ya) return xa;
        return xa+(xb-xa)*(y-ya)/(yb-ya);
    };
    auto span=[&](int y,int xa,int xb){
        if(y<0||y>=H) return;
        if(xa>xb) std::swap(xa,xb);
        xa=std::max(0,xa); xb=std::min(W-1,xb);
        for(int x=xa;x<=xb;x++){
            int idx=(y*W+x)*4;
            img[idx]=r; img[idx+1]=g; img[idx+2]=b; img[idx+3]=255;
        }
    };
    for(int y=y0;y<=y1;y++) span(y, interp(y,y0,x0,y2,x2), interp(y,y0,x0,y1,x1));
    for(int y=y1;y<=y2;y++) span(y, interp(y,y0,x0,y2,x2), interp(y,y1,x1,y2,x2));
}

// ─────────────────────────────────────────────────────────────────────────────
// load_mesh
// ─────────────────────────────────────────────────────────────────────────────

PaperMesh load_mesh(const std::string& path)
{
    PaperMesh mesh;
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_vertex_status();

    OpenMesh::IO::Options opt;
    if (!OpenMesh::IO::read_mesh(mesh, path, opt))
        throw std::runtime_error("Failed to load mesh: " + path);

    // Remove degenerate (zero-area) faces
    for (auto fh : mesh.faces()) {
        auto fv = mesh.fv_begin(fh);
        auto p0 = mesh.point(*fv); ++fv;
        auto p1 = mesh.point(*fv); ++fv;
        auto p2 = mesh.point(*fv);
        auto cross = OpenMesh::cross(p1-p0, p2-p0);
        if (cross.length() < 1e-12f)
            mesh.delete_face(fh, true);
    }
    mesh.garbage_collection();
    return mesh;
}

// ─────────────────────────────────────────────────────────────────────────────
// repair_mesh
// ─────────────────────────────────────────────────────────────────────────────

void repair_mesh(PaperMesh& mesh)
{
    mesh.request_face_status();
    mesh.request_edge_status();
    mesh.request_vertex_status();

    for (auto vh : mesh.vertices())
        if (mesh.is_isolated(vh))
            mesh.delete_vertex(vh);

    mesh.garbage_collection();

    mesh.request_face_normals();
    mesh.request_vertex_normals();
    mesh.update_normals();
}

// ─────────────────────────────────────────────────────────────────────────────
// largest_component  (BFS over faces via face–face adjacency)
// ─────────────────────────────────────────────────────────────────────────────

PaperMesh largest_component(const PaperMesh& mesh)
{
    int nf = static_cast<int>(mesh.n_faces());
    if (nf == 0) return mesh;

    std::vector<bool> visited(nf, false);
    std::vector<std::vector<int>> components;

    for (int start = 0; start < nf; start++) {
        if (visited[start]) continue;
        auto fh_start = mesh.face_handle(start);
        if (!fh_start.is_valid()) continue;

        std::vector<int> comp;
        std::queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int fi = q.front(); q.pop();
            comp.push_back(fi);
            auto fh = mesh.face_handle(fi);
            for (auto ff = mesh.cff_begin(fh); ff != mesh.cff_end(fh); ++ff) {
                int fj = ff->idx();
                if (fj >= 0 && fj < nf && !visited[fj]) {
                    visited[fj] = true;
                    q.push(fj);
                }
            }
        }
        components.push_back(std::move(comp));
    }

    if (components.empty()) return mesh;

    auto& best = *std::max_element(components.begin(), components.end(),
        [](const auto& a, const auto& b){ return a.size() < b.size(); });

    std::unordered_set<int> keep(best.begin(), best.end());

    PaperMesh result;
    result.request_face_status();
    result.request_edge_status();
    result.request_vertex_status();

    std::vector<PaperMesh::VertexHandle> vmap(mesh.n_vertices());
    for (auto& vh : vmap) vh = PaperMesh::VertexHandle(-1);

    for (int fi : best) {
        auto fh = mesh.face_handle(fi);
        std::vector<PaperMesh::VertexHandle> fvhs;
        for (auto fv = mesh.cfv_begin(fh); fv != mesh.cfv_end(fh); ++fv) {
            int vi = fv->idx();
            if (!vmap[vi].is_valid())
                vmap[vi] = result.add_vertex(mesh.point(*fv));
            fvhs.push_back(vmap[vi]);
        }
        result.add_face(fvhs);
    }
    result.garbage_collection();
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_stats
// ─────────────────────────────────────────────────────────────────────────────

MeshStats compute_stats(const PaperMesh& mesh)
{
    MeshStats s;
    s.n_vertices = static_cast<int>(mesh.n_vertices());
    s.n_edges    = static_cast<int>(mesh.n_edges());
    s.n_faces    = static_cast<int>(mesh.n_faces());

    if (mesh.n_vertices() > 0) {
        auto pmin = mesh.point(*mesh.vertices_begin());
        auto pmax = pmin;
        for (auto vh : mesh.vertices()) {
            pmin = OpenMesh::min(pmin, mesh.point(vh));
            pmax = OpenMesh::max(pmax, mesh.point(vh));
        }
        s.bbox_diag = static_cast<double>((pmax-pmin).length());
    }

    if (mesh.n_edges() > 0) {
        double sum = 0.0;
        for (auto eh : mesh.edges()) {
            auto heh = mesh.halfedge_handle(eh, 0);
            sum += static_cast<double>(
                (mesh.point(mesh.to_vertex_handle(heh)) -
                 mesh.point(mesh.from_vertex_handle(heh))).length());
        }
        s.avg_edge_length = sum / mesh.n_edges();
    }

    // Boundary loops
    std::vector<bool> vis_heh(mesh.n_halfedges(), false);
    for (auto heh : mesh.halfedges()) {
        if (mesh.is_boundary(heh) && !vis_heh[heh.idx()]) {
            s.has_boundary = true;
            s.n_boundary_loops++;
            auto cur = heh;
            do { vis_heh[cur.idx()] = true; cur = mesh.next_halfedge_handle(cur); } while (cur != heh);
        }
    }

    // Components (BFS over faces)
    int nf = static_cast<int>(mesh.n_faces());
    std::vector<bool> vis(nf, false);
    for (int i = 0; i < nf; i++) {
        if (vis[i]) continue;
        s.n_components++;
        std::queue<int> q; q.push(i); vis[i]=true;
        while (!q.empty()) {
            int fi=q.front(); q.pop();
            auto fh=mesh.face_handle(fi);
            for (auto ff=mesh.cff_begin(fh); ff!=mesh.cff_end(fh); ++ff) {
                int fj=ff->idx();
                if (fj>=0 && fj<nf && !vis[fj]) { vis[fj]=true; q.push(fj); }
            }
        }
    }
    s.is_manifold = true; // simplified: trust OpenMesh construction
    return s;
}

// ─────────────────────────────────────────────────────────────────────────────
// save_mesh
// ─────────────────────────────────────────────────────────────────────────────

void save_mesh(const PaperMesh& mesh, const std::string& path)
{
    if (!OpenMesh::IO::write_mesh(mesh, path))
        throw std::runtime_error("Failed to save mesh: " + path);
}

// ─────────────────────────────────────────────────────────────────────────────
// render_mesh_png
// ─────────────────────────────────────────────────────────────────────────────

void render_mesh_png(const PaperMesh& mesh, const std::string& out_path)
{
    const int W=800, H=800;
    std::vector<uint8_t> img(W*H*4);
    // light-grey background
    for (int i=0;i<W*H*4;i+=4){img[i]=220;img[i+1]=220;img[i+2]=220;img[i+3]=255;}

    if (mesh.n_vertices()==0){
        stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
        return;
    }

    // Orthographic projection along view = (1,1,1)
    Eigen::Vector3d right = Eigen::Vector3d(-1,1,0).normalized();
    Eigen::Vector3d view  = Eigen::Vector3d(1,1,1).normalized();
    Eigen::Vector3d up    = view.cross(right).normalized();

    int nv = static_cast<int>(mesh.n_vertices());
    std::vector<double> px(nv), py(nv);
    double xmin=1e18,xmax=-1e18,ymin=1e18,ymax=-1e18;
    for (auto vh : mesh.vertices()) {
        auto p = mesh.point(vh);
        Eigen::Vector3d pt(p[0],p[1],p[2]);
        int i = vh.idx();
        px[i]=pt.dot(right); py[i]=pt.dot(up);
        xmin=std::min(xmin,px[i]); xmax=std::max(xmax,px[i]);
        ymin=std::min(ymin,py[i]); ymax=std::max(ymax,py[i]);
    }

    const double margin=0.05;
    double rx=xmax-xmin, ry=ymax-ymin;
    if(rx<1e-12) rx=1.0; if(ry<1e-12) ry=1.0;
    double scale=std::min((W*(1-2*margin))/rx,(H*(1-2*margin))/ry);
    double ox=W*margin-xmin*scale, oy=H*margin-ymin*scale;

    auto topx=[&](int vi)->int{ return (int)(px[vi]*scale+ox); };
    auto topy=[&](int vi)->int{ return H-1-(int)(py[vi]*scale+oy); };

    // Shade triangles
    PaperMesh& m = const_cast<PaperMesh&>(mesh);
    m.request_face_normals(); m.update_face_normals();

    for (auto fh : mesh.faces()){
        std::vector<int> vids;
        for (auto fv=mesh.cfv_begin(fh);fv!=mesh.cfv_end(fh);++fv) vids.push_back(fv->idx());
        if (vids.size()<3) continue;
        auto n=m.normal(fh);
        Eigen::Vector3d nv(n[0],n[1],n[2]);
        double dot=std::abs(nv.dot(view));
        auto shade=static_cast<uint8_t>(100+120*clamp(dot,0.0,1.0));
        fill_tri_img(img,W,H, topx(vids[0]),topy(vids[0]),
                     topx(vids[1]),topy(vids[1]),
                     topx(vids[2]),topy(vids[2]), shade,shade,static_cast<uint8_t>(std::min(255,shade+20)));
    }
    // Wireframe
    for (auto eh : mesh.edges()){
        auto heh=mesh.halfedge_handle(eh,0);
        int vi=mesh.from_vertex_handle(heh).idx();
        int vj=mesh.to_vertex_handle(heh).idx();
        draw_line_img(img,W,H, topx(vi),topy(vi), topx(vj),topy(vj), 60,60,60);
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}
