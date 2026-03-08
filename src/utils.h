#pragma once
/**
 * @file utils.h
 * @brief Header-only utility functions for the papercraft pipeline.
 */

#include <chrono>
#include <string>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <algorithm>

/**
 * @brief High-resolution wall-clock timer.
 */
struct Timer {
    std::chrono::high_resolution_clock::time_point start_time;

    /// Constructs and starts the timer.
    Timer() : start_time(std::chrono::high_resolution_clock::now()) {}

    /// Returns milliseconds elapsed since construction.
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_time).count();
    }

    /// Prints elapsed time with a descriptive label.
    void print(const std::string& label) const {
        std::cout << "[Timer] " << label << ": " << elapsed_ms() << " ms\n";
    }
};

/**
 * @brief Log severity levels.
 */
enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

/**
 * @brief Log a message at the specified severity level.
 * @param level  Severity of the message.
 * @param msg    Message text.
 */
inline void log(LogLevel level, const std::string& msg) {
    const char* prefix = "";
    switch (level) {
        case LogLevel::DEBUG:   prefix = "[DEBUG] "; break;
        case LogLevel::INFO:    prefix = "[INFO]  "; break;
        case LogLevel::WARNING: prefix = "[WARN]  "; break;
        case LogLevel::ERROR:   prefix = "[ERROR] "; break;
    }
    if (level == LogLevel::ERROR)
        std::cerr << prefix << msg << "\n";
    else
        std::cout << prefix << msg << "\n";
}

/**
 * @brief Log a named numeric metric with optional unit.
 * @param name  Metric name.
 * @param value Numeric value.
 * @param unit  Optional unit string (e.g. "ms", "faces").
 */
inline void log_metric(const std::string& name, double value,
                        const std::string& unit = "") {
    std::cout << "[METRIC] " << name << " = " << value;
    if (!unit.empty()) std::cout << " " << unit;
    std::cout << "\n";
}

/**
 * @brief Ensure a directory (and all parents) exists.
 * @param path Directory path to create.
 */
inline void ensure_dir(const std::string& path) {
    std::filesystem::create_directories(path);
}

/**
 * @brief Extract the filename stem (no extension) from a path.
 * @param path File path.
 * @return Stem string.
 */
inline std::string stem(const std::string& path) {
    return std::filesystem::path(path).stem().string();
}

/**
 * @brief Convert degrees to radians.
 * @param deg Angle in degrees.
 * @return Angle in radians.
 */
inline double deg2rad(double deg) {
    return deg * M_PI / 180.0;
}

/**
 * @brief Convert radians to degrees.
 * @param rad Angle in radians.
 * @return Angle in degrees.
 */
inline double rad2deg(double rad) {
    return rad * 180.0 / M_PI;
}

/**
 * @brief Clamp a value to the closed interval [lo, hi].
 * @tparam T Numeric type.
 * @param val Value to clamp.
 * @param lo  Lower bound.
 * @param hi  Upper bound.
 * @return Clamped value.
 */
template<typename T>
inline T clamp(T val, T lo, T hi) {
    return std::max(lo, std::min(hi, val));
}
