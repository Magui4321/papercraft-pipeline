/**
 * @file main.cpp
 * @brief Text-to-Papercraft pipeline entry point.
 *
 * Usage:
 *   papercraft <input_mesh> [--config config.json] [options]
 *
 * Pipeline stages:
 *   1. Load & repair mesh
 *   2. Fold-aware simplification (Constraint-Plane QEM)
 *   3. Spectral segmentation
 *   4. LSCM UV unfolding (Iterative Hierarchical Splitting)
 *   5. Sheet layout & tab generation
 *   6. Output PNGs + metrics
 */

#include "config.h"
#include "utils.h"
#include "mesh_loader.h"
#include "simplification.h"
#include "segmentation.h"
#include "unfolding.h"
#include "sheet_generator.h"
#include "metrics.h"

#include <iostream>
#include <string>
#include <filesystem>
#include <stdexcept>

int main(int argc, char** argv)
{
    // ── Parse arguments ──────────────────────────────────────────────────
    if (argc < 2) {
        std::cerr << "Usage: papercraft <input_mesh> [--config config.json] [options]\n";
        return 1;
    }

    std::string input_path = argv[1];

    Config cfg;
    // Look for --config flag
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--config" && i+1 < argc) {
            cfg = Config::from_json(argv[i+1]);
            break;
        }
    }
    // Apply any remaining CLI overrides
    cfg = Config::from_args(argc, argv);

    try { cfg.validate(); }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Config validation: " << e.what() << "\n";
        return 1;
    }

    if (cfg.verbose) cfg.print();

    // ── Prepare output directory ─────────────────────────────────────────
    ensure_dir(cfg.output_dir);
    std::string base = stem(input_path);

    PipelineMetrics metrics;
    metrics.input_path = input_path;

    Timer t_total;

    // ════════════════════════════════════════════════════════════════════
    // STAGE 1: Load & repair
    // ════════════════════════════════════════════════════════════════════
    log(LogLevel::INFO, "Stage 1: Loading mesh: " + input_path);
    Timer t1;
    PaperMesh mesh;
    try {
        mesh = load_mesh(input_path);
    } catch (const std::exception& e) {
        log(LogLevel::ERROR, std::string("Load failed: ") + e.what());
        return 1;
    }

    repair_mesh(mesh);
    mesh = largest_component(mesh);

    auto stats0 = compute_stats(mesh);
    metrics.input_vertices = stats0.n_vertices;
    metrics.input_faces    = stats0.n_faces;
    metrics.t_load_ms      = t1.elapsed_ms();

    log_metric("input_vertices", stats0.n_vertices);
    log_metric("input_faces",    stats0.n_faces);
    log_metric("input_components", stats0.n_components);
    log_metric("bbox_diagonal",  stats0.bbox_diag);

    if (!cfg.dry_run)
        render_mesh_png(mesh, cfg.output_dir+"/"+base+"_01_input.png");

    // ════════════════════════════════════════════════════════════════════
    // STAGE 2: Fold-aware simplification (Constraint-Plane QEM)
    // ════════════════════════════════════════════════════════════════════
    log(LogLevel::INFO, "Stage 2: Fold-aware simplification → " +
        std::to_string(cfg.target_face_count) + " faces");
    Timer t2;

    auto fold_edges = detect_fold_edges(mesh, cfg.fold_angle_thresh);
    auto simp       = fold_aware_simplify(mesh, fold_edges, cfg);
    simp.print();

    metrics.fold_preserve   = simp.fold_preservation_ratio;
    metrics.t_simplify_ms   = t2.elapsed_ms();

    auto stats1 = compute_stats(simp.mesh);
    metrics.simp_vertices = stats1.n_vertices;
    metrics.simp_faces    = stats1.n_faces;

    log_metric("simp_faces",          stats1.n_faces);
    log_metric("fold_preserve_ratio", simp.fold_preservation_ratio);
    log_metric("fold_edges",          static_cast<double>(simp.fold_edges.total_detected));

    if (!cfg.dry_run) {
        simp.save_metrics_json(cfg.output_dir + "/stage2_metrics.json");
        simp.fold_edges.save_json(cfg.output_dir + "/stage2_fold_edges.json");
        render_foldlines_png(simp.mesh, simp.fold_edges,
                             cfg.output_dir + "/stage2_foldlines.png");
        save_mesh(simp.mesh, cfg.output_dir+"/"+base+"_simplified.obj");
    }

    if (simp.fold_preservation_ratio < cfg.fold_preserve_ratio)
        log(LogLevel::WARNING, "Fold preservation ratio below threshold: " +
            std::to_string(simp.fold_preservation_ratio));

    // ════════════════════════════════════════════════════════════════════
    // STAGE 3: Spectral segmentation
    // ════════════════════════════════════════════════════════════════════
    log(LogLevel::INFO, "Stage 3: Spectral segmentation");
    Timer t3;

    auto seg = segment_mesh(simp.mesh, cfg, simp.fold_edges);
    seg.print();

    metrics.n_patches    = static_cast<int>(seg.patches.size());
    metrics.t_segment_ms = t3.elapsed_ms();

    log_metric("n_patches", static_cast<double>(seg.patches.size()));

    if (!cfg.dry_run) {
        seg.save_json(cfg.output_dir + "/stage3_metrics.json");
        render_patches_png(simp.mesh, seg.patches,
                           cfg.output_dir+"/"+base+"_03_patches.png");
    }

    // ════════════════════════════════════════════════════════════════════
    // STAGE 4: LSCM UV unfolding (Iterative Hierarchical Splitting)
    // ════════════════════════════════════════════════════════════════════
    log(LogLevel::INFO, "Stage 4: LSCM UV unfolding");
    Timer t4;

    auto unfold = unfold_patches(simp.mesh, seg, cfg);
    unfold.print();

    metrics.t_unfold_ms = t4.elapsed_ms();

    std::vector<double> distortions;
    for (auto& r : unfold.patches) {
        distortions.push_back(r.distortion);
    }
    build_pipeline_metrics(metrics, distortions);

    log_metric("mean_uv_distortion", metrics.mean_distortion);
    log_metric("max_uv_distortion",  metrics.max_distortion);

    if (!cfg.dry_run) {
        unfold.save_json(cfg.output_dir + "/stage4_metrics.json");
        render_uv_layout_png(unfold.patches,
                             cfg.output_dir+"/"+base+"_04_uv.png");
    }

    // ════════════════════════════════════════════════════════════════════
    // STAGE 5: Sheet layout
    // ════════════════════════════════════════════════════════════════════
    log(LogLevel::INFO, "Stage 5: Sheet layout & tab generation");
    Timer t5;

    auto pages = generate_sheet(unfold.patches, cfg);
    metrics.n_pages    = static_cast<int>(pages.size());
    metrics.t_sheet_ms = t5.elapsed_ms();

    log_metric("n_pages", static_cast<double>(pages.size()));

    // Collect all tabs
    std::vector<Tab> all_tabs;
    for (auto& r : unfold.patches) {
        auto tabs = compute_tabs(r, cfg.tab_width_mm);
        all_tabs.insert(all_tabs.end(), tabs.begin(), tabs.end());
    }

    if (!cfg.dry_run) {
        for (auto& page : pages) {
            std::string out = cfg.output_dir+"/"+base+"_05_sheet_p"+
                              std::to_string(page.page_number)+".png";
            render_sheet_png(unfold.patches, all_tabs, page, cfg, out);
        }
    }

    // Overlap check across all pages
    for (auto& page : pages)
        if (detect_overlaps(unfold.patches, page)) {
            metrics.has_overlaps = true;
            break;
        }

    // ════════════════════════════════════════════════════════════════════
    // STAGE 6: Metrics output
    // ════════════════════════════════════════════════════════════════════
    metrics.t_total_ms = t_total.elapsed_ms();
    metrics.print_table();

    if (!cfg.dry_run) {
        metrics.save_json(cfg.output_dir+"/"+base+"_metrics.json");
        metrics.save_report_txt(cfg.output_dir+"/"+base+"_report.txt");
        metrics.render_report_png(cfg.output_dir+"/"+base+"_06_report.png");
    }

    log(LogLevel::INFO, "Done. Total time: " + std::to_string((int)metrics.t_total_ms) + " ms");
    return 0;
}
