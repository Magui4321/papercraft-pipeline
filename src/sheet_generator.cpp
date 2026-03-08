/**
 * @file sheet_generator.cpp
 * @brief Printable sheet layout implementation.
 */

#include "sheet_generator.h"
#include "utils.h"

#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <unordered_map>
#include <iostream>

// ─────────────────────────────────────────────────────────────────────────────
// scale_patches_to_sheet
// ─────────────────────────────────────────────────────────────────────────────

void scale_patches_to_sheet(std::vector<UnfoldResult>& results,
                             const Config& cfg)
{
    // Target: the diagonal of each patch's UV bounding box ≤ 80% of the
    // shorter sheet dimension.  Scale each patch individually.
    double target_mm = std::min(cfg.sheet_w_mm, cfg.sheet_h_mm) * 0.8;

    for (auto& r : results) {
        if (r.UV.rows() == 0) continue;
        double umin=r.UV.col(0).minCoeff(), umax=r.UV.col(0).maxCoeff();
        double vmin=r.UV.col(1).minCoeff(), vmax=r.UV.col(1).maxCoeff();
        double diag = std::sqrt((umax-umin)*(umax-umin)+(vmax-vmin)*(vmax-vmin));
        if (diag < 1e-12) continue;
        double s = target_mm / diag;
        r.UV *= s;
        // Translate so bottom-left at (0,0)
        double new_umin = r.UV.col(0).minCoeff();
        double new_vmin = r.UV.col(1).minCoeff();
        r.UV.col(0).array() -= new_umin;
        r.UV.col(1).array() -= new_vmin;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// compute_tabs
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Tab> compute_tabs(const UnfoldResult& result, double tab_width_mm)
{
    std::vector<Tab> tabs;
    const auto& UV = result.UV;
    const auto& F  = result.F;
    int nF = F.rows(), nV = UV.rows();
    if (nF == 0 || nV == 0) return tabs;

    // Find boundary edges (appear in exactly one triangle)
    std::map<std::pair<int,int>,int> edge_cnt;
    for (int f = 0; f < nF; f++)
        for (int e = 0; e < 3; e++) {
            int a=F(f,e), b=F(f,(e+1)%3);
            edge_cnt[{std::min(a,b), std::max(a,b)}]++;
        }

    int eid = 0;
    for (auto& [key, cnt] : edge_cnt) {
        if (cnt != 1) { eid++; continue; }
        int a = key.first, b = key.second;
        if (a >= nV || b >= nV) { eid++; continue; }

        Eigen::Vector2d p0(UV(a,0), UV(a,1));
        Eigen::Vector2d p1(UV(b,0), UV(b,1));
        Eigen::Vector2d edge_dir = (p1-p0);
        double len = edge_dir.norm();
        if (len < 1e-8) { eid++; continue; }
        edge_dir /= len;

        // Outward normal (rotate 90° — direction away from mesh interior)
        Eigen::Vector2d normal(-edge_dir.y(), edge_dir.x());
        Eigen::Vector2d t0 = p0 + normal * tab_width_mm;
        Eigen::Vector2d t1 = p1 + normal * tab_width_mm;

        Tab tab;
        tab.p0 = p0; tab.p1 = p1;
        tab.t0 = t0; tab.t1 = t1;
        tab.patch_id = result.patch_id;
        tab.edge_id  = eid++;
        tabs.push_back(tab);
    }
    return tabs;
}

// ─────────────────────────────────────────────────────────────────────────────
// pack_patches  (greedy shelf bin packing)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<SheetPage> pack_patches(const std::vector<UnfoldResult>& results,
                                     const Config& cfg)
{
    std::vector<SheetPage> pages;
    const double W = cfg.sheet_w_mm;
    const double H = cfg.sheet_h_mm;
    const double PAD = 5.0; // 5 mm padding between patches

    struct Shelf {
        double y;         // current shelf top (y offset on page)
        double x;         // current x cursor on shelf
        double shelf_h;   // height of tallest patch on this shelf
    };

    SheetPage cur_page;
    cur_page.page_number = 1;
    std::vector<Shelf> shelves = { {PAD, PAD, 0.0} };

    int n = static_cast<int>(results.size());
    for (int i = 0; i < n; i++) {
        const auto& r = results[i];
        if (r.UV.rows() == 0) continue;

        double pw = r.UV.col(0).maxCoeff() - r.UV.col(0).minCoeff();
        double ph = r.UV.col(1).maxCoeff() - r.UV.col(1).minCoeff();
        pw += PAD; ph += PAD;

        bool placed = false;
        for (auto& shelf : shelves) {
            if (shelf.x + pw <= W - PAD &&
                shelf.y + ph <= H - PAD) {
                cur_page.patch_ids.push_back(i);
                cur_page.offsets.push_back(Eigen::Vector2d(shelf.x, shelf.y));
                cur_page.scales.push_back(1.0);
                shelf.x += pw;
                shelf.shelf_h = std::max(shelf.shelf_h, ph);
                placed = true;
                break;
            }
        }

        if (!placed) {
            // Start new shelf on current page
            double new_shelf_y = shelves.back().y + shelves.back().shelf_h;
            if (new_shelf_y + ph <= H - PAD) {
                shelves.push_back({new_shelf_y, PAD, 0.0});
                auto& shelf = shelves.back();
                cur_page.patch_ids.push_back(i);
                cur_page.offsets.push_back(Eigen::Vector2d(shelf.x, shelf.y));
                cur_page.scales.push_back(1.0);
                shelf.x += pw;
                shelf.shelf_h = ph;
            } else {
                // Start new page
                pages.push_back(cur_page);
                cur_page = SheetPage();
                cur_page.page_number = static_cast<int>(pages.size()) + 1;
                shelves = { {PAD, PAD, 0.0} };

                cur_page.patch_ids.push_back(i);
                cur_page.offsets.push_back(Eigen::Vector2d(shelves[0].x, shelves[0].y));
                cur_page.scales.push_back(1.0);
                shelves[0].x += pw;
                shelves[0].shelf_h = ph;
            }
        }
    }

    if (!cur_page.patch_ids.empty())
        pages.push_back(cur_page);

    return pages;
}

// ─────────────────────────────────────────────────────────────────────────────
// detect_overlaps  (SAT on axis-aligned bounding boxes — simple version)
// ─────────────────────────────────────────────────────────────────────────────

bool detect_overlaps(const std::vector<UnfoldResult>& results,
                     const SheetPage& page)
{
    int n = static_cast<int>(page.patch_ids.size());

    // Axis-aligned bounding boxes in sheet coordinates
    struct AABB { double x0,y0,x1,y1; };
    std::vector<AABB> boxes(n);

    for (int i = 0; i < n; i++) {
        int pi = page.patch_ids[i];
        const auto& UV = results[pi].UV;
        Eigen::Vector2d off = page.offsets[i];
        double umin=UV.col(0).minCoeff(), umax=UV.col(0).maxCoeff();
        double vmin=UV.col(1).minCoeff(), vmax=UV.col(1).maxCoeff();
        boxes[i] = { umin+off.x(), vmin+off.y(), umax+off.x(), vmax+off.y() };
    }

    for (int i = 0; i < n; i++)
        for (int j = i+1; j < n; j++) {
            bool sep = boxes[i].x1 < boxes[j].x0 || boxes[j].x1 < boxes[i].x0
                    || boxes[i].y1 < boxes[j].y0 || boxes[j].y1 < boxes[i].y0;
            if (!sep) return true;
        }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// render_sheet_png  (cut lines black, fold lines dashed blue, tabs light grey)
// ─────────────────────────────────────────────────────────────────────────────

void render_sheet_png(const std::vector<UnfoldResult>& results,
                      const std::vector<Tab>& all_tabs,
                      const SheetPage& page,
                      const Config& cfg,
                      const std::string& out_path)
{
    // Convert mm → pixels using sheet_dpi
    double mm_per_inch = 25.4;
    double ppm = cfg.sheet_dpi / mm_per_inch; // pixels per mm

    int W = static_cast<int>(cfg.sheet_w_mm * ppm);
    int H = static_cast<int>(cfg.sheet_h_mm * ppm);
    W = clamp(W, 100, 4000);
    H = clamp(H, 100, 5600);

    std::vector<uint8_t> img(W*H*4);
    // White background
    std::fill(img.begin(), img.end(), 255);
    for (int i = 3; i < W*H*4; i += 4) img[i] = 255;

    auto mm2px=[&](double mm_val)->int{ return static_cast<int>(mm_val * ppm); };

    // Bresenham line drawing
    auto draw_line=[&](double x0mm,double y0mm,double x1mm,double y1mm,
                        uint8_t r,uint8_t g,uint8_t b){
        int x0=mm2px(x0mm), y0=H-1-mm2px(y0mm);
        int x1=mm2px(x1mm), y1=H-1-mm2px(y1mm);
        int dx=std::abs(x1-x0),dy=std::abs(y1-y0);
        int sx=(x0<x1)?1:-1,sy=(y0<y1)?1:-1,err=dx-dy;
        while(true){
            if(x0>=0&&x0<W&&y0>=0&&y0<H){int idx=(y0*W+x0)*4;img[idx]=r;img[idx+1]=g;img[idx+2]=b;img[idx+3]=255;}
            if(x0==x1&&y0==y1) break;
            int e2=2*err;
            if(e2>-dy){err-=dy;x0+=sx;}
            if(e2< dx){err+=dx;y0+=sy;}
        }
    };

    // Draw each patch on this page
    for (int i = 0; i < static_cast<int>(page.patch_ids.size()); i++) {
        int pi      = page.patch_ids[i];
        auto off    = page.offsets[i];
        const auto& r = results[pi];
        const auto& UV = r.UV;
        const auto& F  = r.F;

        if (UV.rows()==0) continue;

        // Draw cut lines (black, solid)
        std::map<std::pair<int,int>,int> edge_cnt;
        for (int f=0;f<F.rows();f++)
            for (int e=0;e<3;e++){
                int a=F(f,e),b=F(f,(e+1)%3);
                edge_cnt[{std::min(a,b),std::max(a,b)}]++;
            }

        for (auto& [key,cnt]:edge_cnt){
            int a=key.first,b=key.second;
            if(a>=UV.rows()||b>=UV.rows()) continue;
            double x0=UV(a,0)+off.x(), y0=UV(a,1)+off.y();
            double x1=UV(b,0)+off.x(), y1=UV(b,1)+off.y();
            if(cnt==1){
                // boundary = cut line (black)
                draw_line(x0,y0,x1,y1,0,0,0);
            } else {
                // interior = fold line (dashed blue)
                double len=std::sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0));
                int steps=std::max(1,(int)(len/2.0));
                for(int s=0;s<steps;s+=2){
                    double ta=static_cast<double>(s)/steps;
                    double tb=static_cast<double>(std::min(s+1,steps))/steps;
                    draw_line(x0+ta*(x1-x0),y0+ta*(y1-y0),
                              x0+tb*(x1-x0),y0+tb*(y1-y0), 30,100,200);
                }
            }
        }
    }

    // Draw tabs — build patch_id → page_idx map once for O(n+m) lookup
    std::unordered_map<int,int> patch_id_to_page_idx;
    for (int i = 0; i < static_cast<int>(page.patch_ids.size()); i++)
        patch_id_to_page_idx[results[page.patch_ids[i]].patch_id] = i;

    for (auto& tab : all_tabs) {
        auto it = patch_id_to_page_idx.find(tab.patch_id);
        if (it == patch_id_to_page_idx.end()) continue;
        int page_idx = it->second;

        auto off = page.offsets[page_idx];
        auto translate=[&](Eigen::Vector2d p)->Eigen::Vector2d{ return p+off; };

        auto p0=translate(tab.p0), p1=translate(tab.p1), t0=translate(tab.t0), t1=translate(tab.t1);

        // Outline the tab trapezoid in dark grey
        draw_line(p0.x(),p0.y(),p1.x(),p1.y(),100,100,100);
        draw_line(p1.x(),p1.y(),t1.x(),t1.y(),100,100,100);
        draw_line(t1.x(),t1.y(),t0.x(),t0.y(),100,100,100);
        draw_line(t0.x(),t0.y(),p0.x(),p0.y(),100,100,100);
    }

    stbi_write_png(out_path.c_str(),W,H,4,img.data(),W*4);
}

// ─────────────────────────────────────────────────────────────────────────────
// generate_sheet
// ─────────────────────────────────────────────────────────────────────────────

std::vector<SheetPage> generate_sheet(std::vector<UnfoldResult>& results,
                                       const Config& cfg)
{
    scale_patches_to_sheet(results, cfg);
    auto pages = pack_patches(results, cfg);

    for (auto& page : pages) {
        bool overlap = detect_overlaps(results, page);
        if (overlap)
            std::cout << "[WARN]  Page " << page.page_number
                      << " has overlapping patches\n";
    }
    return pages;
}
