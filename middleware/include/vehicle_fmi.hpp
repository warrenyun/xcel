#pragma once

#include <fmi4cpp/fmi2/cs_fmu.hpp>
#include <fmi4cpp/fmi2/cs_slave.hpp>
#include <fmi4cpp/fmi4cpp.hpp>
#include <string>
#include <vector>

namespace sim {

/* The simulator instance's vehicle inputs. */
struct VehicleInput {
  double torque_fl;
  double torque_fr;
  double torque_rl;
  double torque_rr;
  double wheel_steer_rad;
};

/* The simulator instance's vehicle state. */
struct VehicleState {
  double ax_world;
  double ay_world;
  double psi_ddot_world;
  double ax_body;
  double ay_body;
  double psi_ddot_body;
  double vx_world;
  double vy_world;
  double psi_dot_world;
  double vx_body;
  double vy_body;
  double psi_dot_body;
  double x_world;
  double y_world;
  double psi_world;
};

/* The FMU */
struct VehicleInputReference {
  fmi2ValueReference torque_fl;
  fmi2ValueReference torque_fr;
  fmi2ValueReference torque_rl;
  fmi2ValueReference torque_rr;
  fmi2ValueReference wheel_steer_rad;
};

struct VehicleStateReference {
  fmi2ValueReference ax_world;
  fmi2ValueReference ay_world;
  fmi2ValueReference psi_ddot_world;
  fmi2ValueReference ax_body;
  fmi2ValueReference ay_body;
  fmi2ValueReference psi_ddot_body;
  fmi2ValueReference vx_world;
  fmi2ValueReference vy_world;
  fmi2ValueReference psi_dot_world;
  fmi2ValueReference vx_body;
  fmi2ValueReference vy_body;
  fmi2ValueReference psi_dot_body;
  fmi2ValueReference x_world;
  fmi2ValueReference y_world;
  fmi2ValueReference psi_world;
};

class Vehicle {
public:
  Vehicle(const std::string& fmi_path, double fixed_step_s, double start_time_s = 0.0);

  void update_state(const VehicleInput& input);
  void update_state(const VehicleInput& input, double step_size_s);
  VehicleState state() const;
  double simulation_time_s() const;

  std::vector<fmi4cpp::fmi4cppByte> serialize_fmu_state() const;
  void deserialize_fmu_state(const std::vector<fmi4cpp::fmi4cppByte>& serialized);

private:
  fmi2ValueReference _get_value_reference(const std::string& name) const;
  void _init_variable_references();
  void _write_fmu_input(const VehicleInput& input);
  void _unpack_fmu_state();

  fmi4cpp::fmi2::fmu _fmu;
  std::shared_ptr<fmi4cpp::fmi2::cs_fmu> _cs_fmu;
  std::shared_ptr<fmi4cpp::fmi2::cs_slave> _instance;

  double _fixed_step_s;
  VehicleInput _last_input;
  VehicleInputReference _input_vr;
  VehicleStateReference _state_vr;
  VehicleState _current_state;
};

} // namespace sim
