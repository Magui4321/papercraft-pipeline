#pragma once
/**
 * @file metrics.h
 * @brief Pipeline run metrics collection and reporting.
 */

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

/**
 * @brief Timing and quality metrics for a single pipeline run.
 */
struct PipelineMetrics {
    // ── Input ──────────────────────────────────────────────
    std::string input_path;
    int  input_vertices     = 0;
    int  input_faces        = 0;

    // ── After simplification ───────────────────────────────
    int  simp_vertices      = 0;
    int  simp_faces         = 0;
    double fold_preserve    = 0.0; ///< Fraction of fold edges preserved

    // ── Segmentation ───────────────────────────────────────
    int  n_patches          = 0;
    double seg_distortion   = 0.0;

    // ── Unfolding ──────────────────────────────────────────
    double mean_distortion  = 0.0;
    double max_distortion   = 0.0;

    // ── Sheet layout ───────────────────────────────────────
    int  n_pages            = 0;
    bool has_overlaps       = false;

    // ── Timing (ms) ────────────────────────────────────────
    double t_load_ms        = 0.0;
    double t_simplify_ms    = 0.0;
    double t_segment_ms     = 0.0;
    double t_unfold_ms      = 0.0;
    double t_sheet_ms       = 0.0;
    double t_total_ms       = 0.0;

    /**
     * @brief Print a formatted table of metrics to stdout.
     */
    void print_table() const;

    /**
     * @brief Save metrics as JSON.
     * @param path Output JSON file path.
     */
    void save_json(const std::string& path) const;

    /**
     * @brief Save human-readable text report.
     * @param path Output text file path.
     */
    void save_report_txt(const std::string& path) const;

    /**
     * @brief Render a simple text-on-image PNG report.
     * @param path Output PNG file path.
     */
    void render_report_png(const std::string& path) const;
};

/**
 * @brief Populate a PipelineMetrics struct with distortion statistics.
 * @param metrics    Metrics struct to update (distortion fields).
 * @param distortions Per-patch distortion values.
 */
void build_pipeline_metrics(PipelineMetrics& metrics,
                             const std::vector<double>& distortions);
