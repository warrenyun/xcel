#include "vehicle_fmi.hpp"

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/transport/Node.hh>

#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <thread>

namespace {

struct AppConfig {
  std::string world_name = "track.sdf";
  std::string model_name = "vehicle";
  std::string fmu_path = "two_track_modelica_model.fmu";
  double step_s = 0.01;
  double drive_torque_nm = 120.0;
  double steering_rad = 0.0;
  double z_world_m = 0.2;
};

AppConfig parse_args(int argc, char** argv) {
  AppConfig config;
  if (argc > 1) {
    config.fmu_path = argv[1];
  }
  if (argc > 2) {
    config.world_name = argv[2];
  }
  if (argc > 3) {
    config.model_name = argv[3];
  }
  if (argc > 4) {
    config.step_s = std::stod(argv[4]);
  }
  if (argc > 5) {
    config.drive_torque_nm = std::stod(argv[5]);
  }
  if (argc > 6) {
    config.steering_rad = std::stod(argv[6]);
  }
  return config;
}

void fill_pose_request(
    const std::string& model_name,
    const sim::VehicleState& state,
    const double z_world_m,
    gz::msgs::Pose& request) {
  request.Clear();
  request.set_name(model_name);

  auto* position = request.mutable_position();
  position->set_x(state.x_world);
  position->set_y(state.y_world);
  position->set_z(z_world_m);

  const double half_yaw = state.psi_world * 0.5;
  auto* orientation = request.mutable_orientation();
  orientation->set_w(std::cos(half_yaw));
  orientation->set_x(0.0);
  orientation->set_y(0.0);
  orientation->set_z(std::sin(half_yaw));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const AppConfig config = parse_args(argc, argv);
    if (config.step_s <= 0.0) {
      std::cerr << "step_s must be > 0\n";
      return 1;
    }

    sim::Vehicle vehicle(config.fmu_path, config.step_s, 0.0);
    sim::VehicleInput input;
    input.torque_fl = config.drive_torque_nm;
    input.torque_fr = config.drive_torque_nm;
    input.torque_rl = config.drive_torque_nm;
    input.torque_rr = config.drive_torque_nm;
    input.wheel_steer_rad = config.steering_rad;

    gz::transport::Node node;
    const std::string service_name = "/world/" + config.world_name + "/set_pose";

    std::cout << "FMU path: " << config.fmu_path << "\n";
    std::cout << "Driving model '" << config.model_name << "' in world '"
              << config.world_name << "' via " << service_name << "\n";
    std::cout << "step_s=" << config.step_s
              << " torque=" << config.drive_torque_nm
              << " steering_rad=" << config.steering_rad << "\n";

    std::size_t step_idx = 0;
    while (true) {
      vehicle.update_state(input, config.step_s);
      const sim::VehicleState state = vehicle.state();

      gz::msgs::Pose request;
      fill_pose_request(config.model_name, state, config.z_world_m, request);

      gz::msgs::Boolean response;
      bool request_succeeded = false;
      const bool service_called = node.Request(
          service_name, request, 1000, response, request_succeeded);

      if (!service_called || !request_succeeded || !response.data()) {
        std::cerr << "Failed set_pose request on " << service_name << "\n";
      } else if (step_idx % 100 == 0) {
        std::cout << "t=" << vehicle.simulation_time_s()
                  << " x=" << state.x_world
                  << " y=" << state.y_world
                  << " yaw=" << state.psi_world << "\n";
      }

      ++step_idx;
      std::this_thread::sleep_for(std::chrono::duration<double>(config.step_s));
    }
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    std::cerr << "usage: dv_sim [fmu_path] [world_name] [model_name] [step_s] "
                 "[drive_torque_nm] [steering_rad]\n";
    return 1;
  }
}
