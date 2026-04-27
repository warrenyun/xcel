#include "vehicle_fmi.hpp"

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/float_v.pb.h>
#include <gz/msgs/pose.pb.h>
#include <gz/msgs/world_control.pb.h>
#include <gz/transport/Node.hh>

#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

static constexpr char WORLD_NAME[] = "track";
static constexpr char MODEL_NAME[] = "vehicle";
static constexpr char FMU_PATH[] = "two_track_modelica_model.fmu";
static constexpr double STEP_S = 0.01;
static constexpr double Z_WORLD_M = 0.0;
static constexpr auto INPUT_TIMEOUT = std::chrono::milliseconds(100);

static void fill_pose(const sim::VehicleState& state, gz::msgs::Pose& msg) {
  msg.Clear();
  msg.set_name(MODEL_NAME);
  auto* p = msg.mutable_position();
  p->set_x(state.x_world);
  p->set_y(state.y_world);
  p->set_z(Z_WORLD_M);
  const double half_yaw = state.psi_world * 0.5;
  auto* q = msg.mutable_orientation();
  q->set_w(std::cos(half_yaw));
  q->set_x(0.0);
  q->set_y(0.0);
  q->set_z(std::sin(half_yaw));
}

int main() {
  try {
    sim::Vehicle vehicle(FMU_PATH, STEP_S);

    std::mutex input_mu;
    sim::VehicleInput shared_input{};
    auto last_input_time = std::chrono::steady_clock::now();

    std::atomic<bool> running{true};

    std::thread sim_thread([&] {
      gz::transport::Node node;
      const std::string pose_srv =
          std::string("/world/") + WORLD_NAME + "/set_pose";
      const std::string ctrl_srv =
          std::string("/world/") + WORLD_NAME + "/control";

      gz::msgs::WorldControl pause_msg;
      pause_msg.set_pause(true);
      gz::msgs::Boolean resp;
      bool ok = false;
      node.Request(ctrl_srv, pause_msg, 1000, resp, ok);

      std::size_t step_idx = 0;

      while (running.load(std::memory_order_relaxed)) {
        sim::VehicleInput input;
        {
          std::lock_guard lock(input_mu);
          input = shared_input;
          auto age = std::chrono::steady_clock::now() - last_input_time;
          if (age > INPUT_TIMEOUT) {
            input.torque_fl = 0.0;
            input.torque_fr = 0.0;
            input.torque_rl = 0.0;
            input.torque_rr = 0.0;
          }
        }

        vehicle.update_state(input);
        const auto state = vehicle.state();

        gz::msgs::Pose pose_req;
        fill_pose(state, pose_req);
        bool req_ok = false;
        node.Request(pose_srv, pose_req, 1000, resp, req_ok);

        if (!req_ok || !resp.data()) {
          std::cerr << "set_pose failed\n";
        } else if (step_idx % 100 == 0) {
          std::cout << "t=" << vehicle.simulation_time_s()
                    << " x=" << state.x_world
                    << " y=" << state.y_world
                    << " yaw=" << state.psi_world << "\n";
        }

        gz::msgs::WorldControl step_msg;
        step_msg.set_multi_step(1);
        node.Request(ctrl_srv, step_msg, 1000, resp, ok);

        ++step_idx;
        std::this_thread::sleep_for(std::chrono::duration<double>(STEP_S));
      }
    });

    gz::transport::Node input_node;
    input_node.Subscribe<gz::msgs::Float_V>(
        "/vehicle/cmd",
        [&](const gz::msgs::Float_V& msg) {
          if (msg.data_size() < 5) return;
          std::lock_guard lock(input_mu);
          shared_input.torque_fl = msg.data(0);
          shared_input.torque_fr = msg.data(1);
          shared_input.torque_rl = msg.data(2);
          shared_input.torque_rr = msg.data(3);
          shared_input.wheel_steer_rad= msg.data(4);
          last_input_time = std::chrono::steady_clock::now();
        });

    std::cout << "listening on /vehicle/cmd (Float_V: fl fr rl rr steer)\n";
    gz::transport::waitForShutdown();

    running.store(false, std::memory_order_relaxed);
    sim_thread.join();

  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }
}
