#pragma once
/**
 * @file config.h
 * @brief Configuration management for the papercraft pipeline.
 */

#include <string>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

/**
 * @brief All configuration parameters for the papercraft pipeline.
 *
 * Can be loaded from a JSON file or parsed from command-line arguments.
 */
struct Config {
    std::string output_dir        = "outputs"; ///< Output directory path
    int         target_face_count = 2000;      ///< Target face count after decimation
    int         n_patches         = 0;         ///< Number of patches (0 = auto)
    double      dihedral_weight   = 1.0;       ///< Weight for dihedral angle in segmentation
    double      distortion_weight = 0.5;       ///< Weight for distortion in segmentation
    double      fold_angle_thresh = 30.0;      ///< Fold edge dihedral angle threshold (degrees)
    double      fold_preserve_ratio = 0.80;    ///< Minimum fraction of fold edges to preserve
    double      tab_width_mm      = 3.0;       ///< Gluing tab width in millimeters
    double      max_distortion_warn = 0.25;    ///< Max UV distortion before warning
    int         sheet_dpi         = 300;       ///< Output sheet DPI
    double      sheet_w_mm        = 210.0;     ///< Sheet width in mm (A4)
    double      sheet_h_mm        = 297.0;     ///< Sheet height in mm (A4)
    int         threads           = 0;         ///< Number of threads (0 = auto)
    bool        verbose           = false;     ///< Enable verbose output
    bool        dry_run           = false;     ///< Dry run (no file output)

    /**
     * @brief Load configuration from a JSON file.
     * @param path Path to the JSON config file.
     * @return Loaded Config object.
     * @throws std::runtime_error if file cannot be opened or parsed.
     */
    static Config from_json(const std::string& path) {
        Config cfg;
        std::ifstream f(path);
        if (!f.is_open()) throw std::runtime_error("Cannot open config: " + path);
        nlohmann::json j;
        f >> j;
        if (j.contains("output_dir"))           cfg.output_dir           = j["output_dir"].get<std::string>();
        if (j.contains("target_face_count"))    cfg.target_face_count    = j["target_face_count"].get<int>();
        if (j.contains("n_patches"))            cfg.n_patches            = j["n_patches"].get<int>();
        if (j.contains("dihedral_weight"))      cfg.dihedral_weight      = j["dihedral_weight"].get<double>();
        if (j.contains("distortion_weight"))    cfg.distortion_weight    = j["distortion_weight"].get<double>();
        if (j.contains("fold_angle_thresh"))    cfg.fold_angle_thresh    = j["fold_angle_thresh"].get<double>();
        if (j.contains("fold_preserve_ratio"))  cfg.fold_preserve_ratio  = j["fold_preserve_ratio"].get<double>();
        if (j.contains("tab_width_mm"))         cfg.tab_width_mm         = j["tab_width_mm"].get<double>();
        if (j.contains("max_distortion_warn"))  cfg.max_distortion_warn  = j["max_distortion_warn"].get<double>();
        if (j.contains("sheet_dpi"))            cfg.sheet_dpi            = j["sheet_dpi"].get<int>();
        if (j.contains("sheet_w_mm"))           cfg.sheet_w_mm           = j["sheet_w_mm"].get<double>();
        if (j.contains("sheet_h_mm"))           cfg.sheet_h_mm           = j["sheet_h_mm"].get<double>();
        if (j.contains("threads"))              cfg.threads              = j["threads"].get<int>();
        if (j.contains("verbose"))              cfg.verbose              = j["verbose"].get<bool>();
        if (j.contains("dry_run"))              cfg.dry_run              = j["dry_run"].get<bool>();
        return cfg;
    }

    /**
     * @brief Parse configuration from command-line arguments.
     * @param argc Argument count.
     * @param argv Argument values.
     * @return Config with CLI-overridden values; loads JSON first if --config is given.
     */
    static Config from_args(int argc, char** argv) {
        // First pass: find --config path and load JSON defaults
        Config cfg;
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--config" && i+1 < argc) {
                cfg = Config::from_json(argv[i+1]);
                break;
            }
        }
        // Second pass: apply CLI overrides on top of (potentially JSON-loaded) cfg
        for (int i = 1; i < argc; i++) {
            std::string arg = argv[i];
            if      (arg == "--config"            && i+1 < argc) { ++i; /* already handled */ }
            else if (arg == "--output_dir"        && i+1 < argc) { cfg.output_dir = argv[++i]; }
            else if (arg == "--target_faces"      && i+1 < argc) { cfg.target_face_count = std::stoi(argv[++i]); }
            else if (arg == "--n_patches"         && i+1 < argc) { cfg.n_patches = std::stoi(argv[++i]); }
            else if (arg == "--threads"           && i+1 < argc) { cfg.threads = std::stoi(argv[++i]); }
            else if (arg == "--fold_angle_thresh" && i+1 < argc) { cfg.fold_angle_thresh = std::stod(argv[++i]); }
            else if (arg == "--tab_width_mm"      && i+1 < argc) { cfg.tab_width_mm = std::stod(argv[++i]); }
            else if (arg == "--verbose")  { cfg.verbose = true; }
            else if (arg == "--dry_run")  { cfg.dry_run = true; }
        }
        return cfg;
    }

    /**
     * @brief Validate configuration parameters.
     * @throws std::invalid_argument if any parameter is out of valid range.
     */
    void validate() const {
        if (target_face_count < 100)
            throw std::invalid_argument("target_face_count must be >= 100");
        if (fold_angle_thresh <= 0.0 || fold_angle_thresh >= 180.0)
            throw std::invalid_argument("fold_angle_thresh must be in (0, 180)");
        if (fold_preserve_ratio < 0.0 || fold_preserve_ratio > 1.0)
            throw std::invalid_argument("fold_preserve_ratio must be in [0, 1]");
        if (tab_width_mm <= 0.0)
            throw std::invalid_argument("tab_width_mm must be positive");
        if (sheet_dpi <= 0)
            throw std::invalid_argument("sheet_dpi must be positive");
        if (sheet_w_mm <= 0.0 || sheet_h_mm <= 0.0)
            throw std::invalid_argument("sheet dimensions must be positive");
        if (threads < 0)
            throw std::invalid_argument("threads must be >= 0");
    }

    /**
     * @brief Pretty-print configuration to stdout.
     */
    void print() const {
        std::cout << "=== Papercraft Pipeline Config ===\n"
                  << "  output_dir:           " << output_dir          << "\n"
                  << "  target_face_count:    " << target_face_count   << "\n"
                  << "  n_patches:            " << n_patches           << " (0=auto)\n"
                  << "  dihedral_weight:      " << dihedral_weight     << "\n"
                  << "  distortion_weight:    " << distortion_weight   << "\n"
                  << "  fold_angle_thresh:    " << fold_angle_thresh   << " deg\n"
                  << "  fold_preserve_ratio:  " << fold_preserve_ratio << "\n"
                  << "  tab_width_mm:         " << tab_width_mm        << " mm\n"
                  << "  max_distortion_warn:  " << max_distortion_warn << "\n"
                  << "  sheet_dpi:            " << sheet_dpi           << "\n"
                  << "  sheet_w_mm:           " << sheet_w_mm          << " mm\n"
                  << "  sheet_h_mm:           " << sheet_h_mm          << " mm\n"
                  << "  threads:              " << threads             << " (0=auto)\n"
                  << "  verbose:              " << (verbose ? "true" : "false") << "\n"
                  << "  dry_run:              " << (dry_run  ? "true" : "false") << "\n"
                  << "==================================\n";
    }
};
