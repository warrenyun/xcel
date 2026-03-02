#include "vehicle_fmi.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace sim {

namespace {
void require_success(bool ok, const std::string& message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}
}

Vehicle::Vehicle(
    const std::string& fmi_path,
    const double fixed_step_s,
    const double start_time_s)
    : _fmu(fmi_path), _fixed_step_s(fixed_step_s) {
  if (_fixed_step_s <= 0.0) {
    throw std::invalid_argument("Vehicle fixed step must be > 0");
  }
  if (!_fmu.supports_cs()) {
    throw std::runtime_error(
        "FMU does not support co-simulation ");
  }

  _cs_fmu = _fmu.as_cs_fmu();
  _instance = _cs_fmu->new_instance(false, false);
  if (!_instance) {
    throw std::runtime_error("Failed to create FMU co-simulation instance");
  }

  resolve_variable_references();

  require_success(
      _instance->setup_experiment(start_time_s),
      "FMU setup_experiment failed");
  require_success(
      _instance->enter_initialization_mode(),
      "FMU enter_initialization_mode failed");

  write_input_to_fmu(_last_input); // zero inputs
  require_success(
      _instance->exit_initialization_mode(),
      "FMU exit_initialization_mode failed");

  read_state_from_fmu();
}

VehicleState Vehicle::state() const { return _current_state; }

double Vehicle::simulation_time_s() const {
  return _instance ? _instance->get_simulation_time() : 0.0;
}

void Vehicle::update_state(const VehicleInput& input) {
  update_state(input, _fixed_step_s);
}

void Vehicle::update_state(const VehicleInput& input, const double step_size_s) {
  if (step_size_s <= 0.0) {
    throw std::invalid_argument("FMU step_size_s must be > 0");
  }

  write_input_to_fmu(input);
  require_success(_instance->step(step_size_s), "FMU step failed");

  _last_input = input;
  read_state_from_fmu();
}

std::vector<fmi4cpp::fmi4cppByte> Vehicle::serialize_fmu_state() const {
  fmi4cpp::fmi4cppFMUstate raw_state = nullptr;
  require_success(_instance->get_fmu_state(raw_state), "FMU get_fmu_state failed");

  std::vector<fmi4cpp::fmi4cppByte> serialized_state;
  const bool serialize_ok =
      _instance->serialize_fmu_state(raw_state, serialized_state);
  const bool free_ok = _instance->free_fmu_state(raw_state);

  if (!serialize_ok) {
    throw std::runtime_error("FMU serialize_fmu_state failed");
  }
  if (!free_ok) {
    throw std::runtime_error("FMU free_fmu_state failed after serialization");
  }

  return serialized_state;
}

void Vehicle::deserialize_fmu_state(
    const std::vector<fmi4cpp::fmi4cppByte>& serialized_state) {
  if (serialized_state.empty()) {
    throw std::invalid_argument("serialized_state must not be empty");
  }

  fmi4cpp::fmi4cppFMUstate raw_state = nullptr;
  require_success(
      _instance->de_serialize_fmu_state(raw_state, serialized_state),
      "FMU de_serialize_fmu_state failed");

  const bool set_ok = _instance->set_fmu_state(raw_state);
  const bool free_ok = _instance->free_fmu_state(raw_state);
  if (!set_ok) {
    throw std::runtime_error("FMU set_fmu_state failed");
  }
  if (!free_ok) {
    throw std::runtime_error("FMU free_fmu_state failed after restore");
  }

  read_state_from_fmu();
}

Vehicle::ValueRef Vehicle::get_real_value_reference(
    const std::string& variable_name) const {
  const auto& variable =
      _instance->get_model_description()->get_variable_by_name(variable_name);
  if (!variable.is_real()) {
    throw std::runtime_error("FMU variable is not Real: " + variable_name);
  }
  return variable.as_real().valueReference();
}

void Vehicle::resolve_variable_references() {
  _input_vr.torque_fl = get_real_value_reference("torque_fl");
  _input_vr.torque_fr = get_real_value_reference("torque_fr");
  _input_vr.torque_rl = get_real_value_reference("torque_rl");
  _input_vr.torque_rr = get_real_value_reference("torque_rr");
  _input_vr.wheel_steer_rad = get_real_value_reference("wheel_steer_rad");

  _state_vr.ax_body = get_real_value_reference("a_x_body_m_s2");
  _state_vr.ax_world = get_real_value_reference("a_x_world_m_s2");
  _state_vr.ay_body = get_real_value_reference("a_y_body_m_s2");
  _state_vr.ay_world = get_real_value_reference("a_y_world_m_s2");
  _state_vr.psi_ddot_body = get_real_value_reference("psi_ddot_body_rad_s2");
  _state_vr.psi_ddot_world = get_real_value_reference("psi_ddot_world_rad_s2");
  _state_vr.psi_dot_body = get_real_value_reference("psi_dot_body_rad_s");
  _state_vr.psi_dot_world = get_real_value_reference("psi_dot_world_rad_s");
  _state_vr.psi_world = get_real_value_reference("psi_world_rad");
  _state_vr.vx_body = get_real_value_reference("v_x_body_m_s");
  _state_vr.vx_world = get_real_value_reference("v_x_world_m_s");
  _state_vr.vy_body = get_real_value_reference("v_y_body_m_s");
  _state_vr.vy_world = get_real_value_reference("v_y_world_m_s");
  _state_vr.x_world = get_real_value_reference("x_world_m");
  _state_vr.y_world = get_real_value_reference("y_world_m");
}

void Vehicle::write_input_to_fmu(const VehicleInput& input) {
  const std::vector<ValueRef> refs = {
      _input_vr.torque_fl,
      _input_vr.torque_fr,
      _input_vr.torque_rl,
      _input_vr.torque_rr,
      _input_vr.wheel_steer_rad};
  const std::vector<double> values = {
      input.torque_fl,
      input.torque_fr,
      input.torque_rl,
      input.torque_rr,
      input.wheel_steer_rad};

  require_success(_instance->write_real(refs, values), "FMU write_real inputs failed");
}

void Vehicle::read_state_from_fmu() {
  const std::vector<ValueRef> refs = {
      _state_vr.ax_world,
      _state_vr.ay_world,
      _state_vr.psi_ddot_world,
      _state_vr.ax_body,
      _state_vr.ay_body,
      _state_vr.psi_ddot_body,
      _state_vr.vx_world,
      _state_vr.vy_world,
      _state_vr.psi_dot_world,
      _state_vr.vx_body,
      _state_vr.vy_body,
      _state_vr.psi_dot_body,
      _state_vr.x_world,
      _state_vr.y_world,
      _state_vr.psi_world};
  std::vector<double> values(refs.size(), 0.0);

  require_success(_instance->read_real(refs, values), "FMU read_real outputs failed");

  _current_state.ax_world = values[0];
  _current_state.ay_world = values[1];
  _current_state.psi_ddot_world = values[2];
  _current_state.ax_body = values[3];
  _current_state.ay_body = values[4];
  _current_state.psi_ddot_body = values[5];
  _current_state.vx_world = values[6];
  _current_state.vy_world = values[7];
  _current_state.psi_dot_world = values[8];
  _current_state.vx_body = values[9];
  _current_state.vy_body = values[10];
  _current_state.psi_dot_body = values[11];
  _current_state.x_world = values[12];
  _current_state.y_world = values[13];
  _current_state.psi_world = values[14];
}

}
