#pragma once

#include <fmi4cpp/fmi4cpp.hpp>

#include <memory>
#include <string>
#include <vector>

namespace sim {

struct VehicleState {
  double ax_world = 0.0;
  double ay_world = 0.0;
  double psi_ddot_world = 0.0;
  double ax_body = 0.0;
  double ay_body = 0.0;
  double psi_ddot_body = 0.0;
  double vx_world = 0.0;
  double vy_world = 0.0;
  double psi_dot_world = 0.0;
  double vx_body = 0.0;
  double vy_body = 0.0;
  double psi_dot_body = 0.0;
  double x_world = 0.0;
  double y_world = 0.0;
  double psi_world = 0.0;
};

struct VehicleInput {
  double torque_fl = 0.0;
  double torque_fr = 0.0;
  double torque_rl = 0.0;
  double torque_rr = 0.0;
  double wheel_steer_rad = 0.0;
};

class Vehicle {
public:
  explicit Vehicle(
      const std::string& fmi_path,
      double fixed_step_s = 0.001,
      double start_time_s = 0.0);

  void update_state(const VehicleInput& input);
  void update_state(const VehicleInput& input, double step_size_s);

  VehicleState state() const;
  double simulation_time_s() const;

  std::vector<fmi4cpp::fmi4cppByte> serialize_fmu_state() const;
  void deserialize_fmu_state(const std::vector<fmi4cpp::fmi4cppByte>& serialized_state);

private:
  using ValueRef = fmi4cpp::fmi4cppValueReference;

  ValueRef get_real_value_reference(const std::string& variable_name) const;
  void resolve_variable_references();
  void write_input_to_fmu(const VehicleInput& input);
  void unpack_fmu_state();

  fmi4cpp::fmi2::fmu _fmu;
  std::unique_ptr<fmi4cpp::fmi2::cs_fmu> _cs_fmu;
  std::unique_ptr<fmi4cpp::fmi2::cs_slave> _instance;

  double _fixed_step_s = 0.001;

  struct {
    ValueRef torque_fl = 0;
    ValueRef torque_fr = 0;
    ValueRef torque_rl = 0;
    ValueRef torque_rr = 0;
    ValueRef wheel_steer_rad = 0;
  } _input_vr;

  struct {
    ValueRef ax_body = 0;
    ValueRef ax_world = 0;
    ValueRef ay_body = 0;
    ValueRef ay_world = 0;
    ValueRef psi_ddot_body = 0;
    ValueRef psi_ddot_world = 0;
    ValueRef psi_dot_body = 0;
    ValueRef psi_dot_world = 0;
    ValueRef psi_world = 0;
    ValueRef vx_body = 0;
    ValueRef vx_world = 0;
    ValueRef vy_body = 0;
    ValueRef vy_world = 0;
    ValueRef x_world = 0;
    ValueRef y_world = 0;
  } _state_vr;

  VehicleState _current_state;
  VehicleInput _last_input;
};

}  // namespace sim
