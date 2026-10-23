// Aligned D405 frames using the installed Homebrew librealsense library.
#include <cstdint>
#include <functional>
#include <iostream>
#include <librealsense2/rs.hpp>
#include <memory>

int main() {
    try {
        std::cerr << "stage: searching for D405" << std::endl;
        rs2::context context;
        auto devices = context.query_devices();
        if (devices.size() == 0) {
            std::cerr << "No RealSense USB device found; check the D405 cable and close other camera apps\n";
            return 2;
        }
        std::cerr << "stage: configuring color and depth streams" << std::endl;
        rs2::pipeline pipeline;
        rs2::config config;
        const auto serial = devices.front().get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
        const struct { int width, height, fps; } modes[] = {
            {640, 480, 30}, {640, 480, 15}, {640, 480, 5},
            {640, 360, 30}, {640, 360, 15}, {640, 360, 5},
            {424, 240, 30}, {424, 240, 15}, {424, 240, 5}
        };
        bool resolved = false;
        for (const auto& mode : modes) {
            rs2::config candidate;
            candidate.enable_device(serial);
            candidate.enable_stream(RS2_STREAM_COLOR, mode.width, mode.height, RS2_FORMAT_YUYV, mode.fps);
            candidate.enable_stream(RS2_STREAM_DEPTH, mode.width, mode.height, RS2_FORMAT_Z16, mode.fps);
            if (candidate.can_resolve(pipeline)) {
                config = candidate;
                std::cerr << "stage: selected " << mode.width << 'x' << mode.height
                          << " YUYV+Z16 @ " << mode.fps << " fps" << std::endl;
                resolved = true;
                break;
            }
        }
        bool use_pipeline = false;
        float scale_mm = 0;
        if (resolved) {
            try {
                std::cerr << "stage: starting D405 pipeline streams" << std::endl;
                auto profile = pipeline.start(config);
                scale_mm = profile.get_device().first<rs2::depth_sensor>().get_depth_scale() * 1000.0f;
                use_pipeline = true;
            } catch (const rs2::error& error) {
                std::cerr << "Pipeline stream start failed: " << error.what() << std::endl;
            }
        }
        rs2::sensor direct_sensor;
        std::unique_ptr<rs2::syncer> syncer;
        if (!use_pipeline) {
            std::cerr << "stage: opening D405 sensor profiles directly" << std::endl;
            for (auto sensor : devices.front().query_sensors()) {
                const auto profiles = sensor.get_stream_profiles();
                for (const auto& mode : modes) {
                    rs2::stream_profile color_profile, depth_profile;
                    for (const auto& stream : profiles) {
                        const auto type = stream.stream_type();
                        if (type != RS2_STREAM_COLOR && type != RS2_STREAM_DEPTH) continue;
                        const auto video = stream.as<rs2::video_stream_profile>();
                        if (video.width() != mode.width || video.height() != mode.height ||
                            stream.fps() != mode.fps) continue;
                        if (type == RS2_STREAM_COLOR && stream.format() == RS2_FORMAT_YUYV)
                            color_profile = stream;
                        if (type == RS2_STREAM_DEPTH && stream.format() == RS2_FORMAT_Z16)
                            depth_profile = stream;
                    }
                    if (!color_profile || !depth_profile) continue;
                    try {
                        sensor.open({color_profile, depth_profile});
                        syncer = std::make_unique<rs2::syncer>(4);
                        sensor.start([&syncer](rs2::frame frame) { (*syncer)(std::move(frame)); });
                        direct_sensor = sensor;
                        scale_mm = sensor.as<rs2::depth_sensor>().get_depth_scale() * 1000.0f;
                        std::cerr << "stage: direct sensor started " << mode.width << 'x'
                                  << mode.height << " @ " << mode.fps << " fps" << std::endl;
                        break;
                    } catch (const rs2::error& error) {
                        std::cerr << "Direct " << mode.width << 'x' << mode.height << " @ "
                                  << mode.fps << " fps failed: " << error.what() << std::endl;
                        try { sensor.stop(); } catch (...) {}
                        try { sensor.close(); } catch (...) {}
                        syncer.reset();
                    }
                }
                if (direct_sensor) break;
            }
            if (!direct_sensor) {
                std::cerr << "No simultaneous D405 color+depth mode could be opened\n";
                try {
                    std::cerr << "stage: checking depth-only stream" << std::endl;
                    rs2::pipeline depth_probe;
                    rs2::config depth_config;
                    depth_config.enable_device(serial);
                    depth_config.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
                    depth_probe.start(depth_config);
                    auto depth_frame = depth_probe.wait_for_frames(3000).get_depth_frame();
                    std::cerr << "Depth-only frame: " << depth_frame.get_width() << 'x'
                              << depth_frame.get_height() << std::endl;
                    depth_probe.stop();
                } catch (const rs2::error& error) {
                    std::cerr << "Depth-only stream failed: " << error.what() << std::endl;
                }
                return 2;
            }
        }
        std::cerr << "stage: streams started" << std::endl;
        rs2::align align_to_color(RS2_STREAM_COLOR);
        char command;
        while (std::cin.get(command) && command == 'R') {
            std::cerr << "stage: waiting for aligned frame" << std::endl;
            auto raw_frames = use_pipeline ? pipeline.wait_for_frames(3000) : syncer->wait_for_frames(3000);
            auto frames = align_to_color.process(raw_frames);
            auto color = frames.get_color_frame();
            auto depth = frames.get_depth_frame();
            if (!color || !depth) {
                std::cerr << "Missing aligned color or depth frame\n";
                return 2;
            }
            const uint32_t width = static_cast<uint32_t>(color.get_width());
            const uint32_t height = static_cast<uint32_t>(color.get_height());
            if (depth.get_width() != static_cast<int>(width) || depth.get_height() != static_cast<int>(height)) {
                std::cerr << "Aligned depth size differs from color size\n";
                return 2;
            }
            const auto intr = color.get_profile().as<rs2::video_stream_profile>().get_intrinsics();
            const float values[] = {intr.fx, intr.fy, intr.ppx, intr.ppy, scale_mm};
            std::cout.write(reinterpret_cast<const char*>(&width), sizeof(width));
            std::cout.write(reinterpret_cast<const char*>(&height), sizeof(height));
            std::cout.write(reinterpret_cast<const char*>(values), sizeof(values));
            std::cout.write(reinterpret_cast<const char*>(color.get_data()), width * height * 2);
            std::cout.write(reinterpret_cast<const char*>(depth.get_data()), width * height * 2);
            std::cout.flush();
        }
        if (use_pipeline) pipeline.stop();
        else { direct_sensor.stop(); direct_sensor.close(); }
        return 0;
    } catch (const rs2::error& error) {
        std::cerr << "RealSense SDK: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "Depth bridge: " << error.what() << '\n';
        return 2;
    }
}
