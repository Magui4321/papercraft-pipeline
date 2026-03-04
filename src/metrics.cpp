/**
 * @file metrics.cpp
 * @brief Pipeline metrics reporting implementation.
 */

#include "metrics.h"
#include "utils.h"

#include <nlohmann/json.hpp>
#include <stb_image_write.h>

#include <fstream>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <cstring>

// ─────────────────────────────────────────────────────────────────────────────
// print_table
// ─────────────────────────────────────────────────────────────────────────────

void PipelineMetrics::print_table() const
{
    auto row=[](const std::string& label, const std::string& val){
        std::cout << "  " << std::left << std::setw(28) << label
                  << val << "\n";
    };
    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║      PAPERCRAFT PIPELINE METRICS      ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n";
    row("Input file",          input_path);
    row("Input  V / F",        std::to_string(input_vertices)+" / "+std::to_string(input_faces));
    row("Simplified  V / F",   std::to_string(simp_vertices)+" / "+std::to_string(simp_faces));
    row("Fold preserve ratio", std::to_string(fold_preserve).substr(0,6));
    row("Patches",             std::to_string(n_patches));
    row("Seg distortion",      std::to_string(seg_distortion).substr(0,8));
    row("Mean UV distortion",  std::to_string(mean_distortion).substr(0,8));
    row("Max  UV distortion",  std::to_string(max_distortion).substr(0,8));
    row("Pages",               std::to_string(n_pages));
    row("Overlaps detected",   has_overlaps ? "YES" : "no");
    std::cout << "  ────────────────────────────────────\n";
    row("t_load_ms",           std::to_string((int)t_load_ms));
    row("t_simplify_ms",       std::to_string((int)t_simplify_ms));
    row("t_segment_ms",        std::to_string((int)t_segment_ms));
    row("t_unfold_ms",         std::to_string((int)t_unfold_ms));
    row("t_sheet_ms",          std::to_string((int)t_sheet_ms));
    row("t_total_ms",          std::to_string((int)t_total_ms));
    std::cout << "  ════════════════════════════════════\n\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// save_json
// ─────────────────────────────────────────────────────────────────────────────

void PipelineMetrics::save_json(const std::string& path) const
{
    nlohmann::json j;
    j["input_path"]       = input_path;
    j["input_vertices"]   = input_vertices;
    j["input_faces"]      = input_faces;
    j["simp_vertices"]    = simp_vertices;
    j["simp_faces"]       = simp_faces;
    j["fold_preserve"]    = fold_preserve;
    j["n_patches"]        = n_patches;
    j["seg_distortion"]   = seg_distortion;
    j["mean_distortion"]  = mean_distortion;
    j["max_distortion"]   = max_distortion;
    j["n_pages"]          = n_pages;
    j["has_overlaps"]     = has_overlaps;
    j["t_load_ms"]        = t_load_ms;
    j["t_simplify_ms"]    = t_simplify_ms;
    j["t_segment_ms"]     = t_segment_ms;
    j["t_unfold_ms"]      = t_unfold_ms;
    j["t_sheet_ms"]       = t_sheet_ms;
    j["t_total_ms"]       = t_total_ms;

    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "[ERROR] Cannot write " << path << "\n"; return; }
    f << j.dump(4) << "\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// save_report_txt
// ─────────────────────────────────────────────────────────────────────────────

void PipelineMetrics::save_report_txt(const std::string& path) const
{
    std::ofstream f(path);
    if (!f.is_open()) { std::cerr << "[ERROR] Cannot write " << path << "\n"; return; }

    f << "Papercraft Pipeline Report\n"
      << "==========================\n"
      << "Input:              " << input_path     << "\n"
      << "Input  V / F:       " << input_vertices << " / " << input_faces   << "\n"
      << "Simplified V / F:   " << simp_vertices  << " / " << simp_faces    << "\n"
      << "Fold preserve:      " << fold_preserve  << "\n"
      << "Patches:            " << n_patches      << "\n"
      << "Seg distortion:     " << seg_distortion << "\n"
      << "Mean UV distortion: " << mean_distortion<< "\n"
      << "Max  UV distortion: " << max_distortion << "\n"
      << "Pages:              " << n_pages        << "\n"
      << "Overlaps:           " << (has_overlaps?"YES":"no") << "\n"
      << "\nTiming:\n"
      << "  Load:       " << t_load_ms     << " ms\n"
      << "  Simplify:   " << t_simplify_ms << " ms\n"
      << "  Segment:    " << t_segment_ms  << " ms\n"
      << "  Unfold:     " << t_unfold_ms   << " ms\n"
      << "  Sheet:      " << t_sheet_ms    << " ms\n"
      << "  Total:      " << t_total_ms    << " ms\n";
}

// ─────────────────────────────────────────────────────────────────────────────
// render_report_png  (simple coloured text box)
// ─────────────────────────────────────────────────────────────────────────────

void PipelineMetrics::render_report_png(const std::string& path) const
{
    const int W=600, H=500;
    std::vector<uint8_t> img(W*H*4);
    // Dark background
    for (int i=0;i<W*H*4;i+=4){img[i]=30;img[i+1]=30;img[i+2]=50;img[i+3]=255;}

    // Draw a simple horizontal bar for each metric
    struct Bar { std::string label; double value; double max_val;
                 uint8_t r,g,b; };
    std::vector<Bar> bars = {
        {"fold_preserve",   fold_preserve,   1.0,  80,180,80 },
        {"seg_distortion",  seg_distortion,  1.0,  200,100,60},
        {"mean_distortion", mean_distortion, 1.0,  100,150,220},
        {"max_distortion",  max_distortion,  1.0,  220,80,80 },
    };

    int bar_h=30, gap=15, y_start=80;
    for (int bi=0;bi<(int)bars.size();bi++){
        auto& b=bars[bi];
        int y=y_start+bi*(bar_h+gap);
        double frac=clamp(b.value/b.max_val,0.0,1.0);
        int bar_len=static_cast<int>(frac*(W-200));
        // Draw bar
        for(int dy=0;dy<bar_h;dy++)
            for(int dx=0;dx<bar_len;dx++){
                int idx=((y+dy)*W+(100+dx))*4;
                if(idx+3<(int)img.size()){
                    img[idx]=b.r;img[idx+1]=b.g;img[idx+2]=b.b;img[idx+3]=255;
                }
            }
    }

    stbi_write_png(path.c_str(),W,H,4,img.data(),W*4);
}

// ─────────────────────────────────────────────────────────────────────────────
// build_pipeline_metrics
// ─────────────────────────────────────────────────────────────────────────────

void build_pipeline_metrics(PipelineMetrics& metrics,
                             const std::vector<double>& distortions)
{
    if (distortions.empty()) return;
    double sum = 0.0;
    double mx  = 0.0;
    for (double d : distortions) {
        sum += d;
        mx   = std::max(mx, d);
    }
    metrics.mean_distortion = sum / distortions.size();
    metrics.max_distortion  = mx;
}
