#pragma once
/**
 * @file sheet_generator.h
 * @brief Printable sheet layout: scaling, tab generation, packing and rendering.
 */

#include "unfolding.h"
#include "config.h"

#include <vector>
#include <Eigen/Dense>

/**
 * @brief One printed page containing multiple packed patches.
 */
struct SheetPage {
    int                       page_number = 0;
    std::vector<int>          patch_ids;   ///< Indices into the UnfoldResult vector
    std::vector<Eigen::Vector2d> offsets;  ///< Translation per patch (mm in sheet space)
    std::vector<double>       scales;      ///< Uniform scale per patch (mm/UV unit)
};

/**
 * @brief Gluing tab along one UV edge.
 */
struct Tab {
    Eigen::Vector2d p0, p1;  ///< Edge end-points (mm)
    Eigen::Vector2d t0, t1;  ///< Tab outer corners (mm)
    int             patch_id = 0;
    int             edge_id  = 0;
};

/**
 * @brief Scale all patch UV coordinates so they fit within the sheet dimensions.
 *
 * Patches are scaled uniformly so their largest extent fills at most
 * 80 % of the shorter sheet dimension.
 *
 * @param results  Unfolding results (UV modified in-place).
 * @param cfg      Pipeline configuration.
 */
void scale_patches_to_sheet(std::vector<UnfoldResult>& results,
                             const Config& cfg);

/**
 * @brief Compute gluing tabs for all boundary edges of a patch.
 * @param result       Unfolding result for the patch.
 * @param tab_width_mm Tab width in millimetres.
 * @return             Vector of Tab objects.
 */
std::vector<Tab> compute_tabs(const UnfoldResult& result,
                               double tab_width_mm);

/**
 * @brief Pack patches onto A4 pages using a greedy shelf-bin algorithm.
 * @param results  Scaled unfolding results.
 * @param cfg      Pipeline configuration.
 * @return         Vector of SheetPage objects.
 */
std::vector<SheetPage> pack_patches(const std::vector<UnfoldResult>& results,
                                     const Config& cfg);

/**
 * @brief Test whether any two patches overlap on a page using SAT.
 * @param results  Unfolding results (UV in mm).
 * @param page     The page to test.
 * @return         true if at least one overlap is detected.
 */
bool detect_overlaps(const std::vector<UnfoldResult>& results,
                     const SheetPage& page);

/**
 * @brief Render one sheet page as a PNG with cut/fold/tab lines.
 * @param results   All unfolding results.
 * @param all_tabs  All pre-computed tabs.
 * @param page      The page descriptor.
 * @param cfg       Pipeline configuration.
 * @param out_path  Output PNG path.
 */
void render_sheet_png(const std::vector<UnfoldResult>& results,
                      const std::vector<Tab>& all_tabs,
                      const SheetPage& page,
                      const Config& cfg,
                      const std::string& out_path);

/**
 * @brief Full sheet generation: scale → tab computation → packing → output.
 * @param results  Unfolding results (modified in-place for scaling).
 * @param cfg      Pipeline configuration.
 * @return         Vector of SheetPage objects.
 */
std::vector<SheetPage> generate_sheet(std::vector<UnfoldResult>& results,
                                       const Config& cfg);
