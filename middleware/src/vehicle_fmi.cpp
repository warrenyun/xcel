#include "vehicle_fmi.hpp"

#include <fmi4cpp/fmi2/fmi2TypesPlatform.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace sim {

/* Mappings from a simulator instance's inputs to the loaded FMU inputs. */
constexpr struct {
  const char* fmu_field;
  double VehicleInput::* value;
  fmi2ValueReference VehicleInputReference::* vr;
} input_map[] = {
  {"torque_fl",       &VehicleInput::torque_fl,       &VehicleInputReference::torque_fl},
  {"torque_fr",       &VehicleInput::torque_fr,       &VehicleInputReference::torque_fr},
  {"torque_rl",       &VehicleInput::torque_rl,       &VehicleInputReference::torque_rl},
  {"torque_rr",       &VehicleInput::torque_rr,       &VehicleInputReference::torque_rr},
  {"wheel_steer_rad", &VehicleInput::wheel_steer_rad, &VehicleInputReference::wheel_steer_rad},
};

/* Mappings from a simulator instance's vehicle state to the loaded FMU vehicle state. */
constexpr struct {
  const char* fmu_field;
  double VehicleState::* value;
  fmi2ValueReference VehicleStateReference::* vr;
} state_map[] = {
  {"a_x_world_m_s2",        &VehicleState::ax_world,      &VehicleStateReference::ax_world},
  {"a_y_world_m_s2",        &VehicleState::ay_world,      &VehicleStateReference::ay_world},
  {"psi_ddot_world_rad_s2", &VehicleState::psi_ddot_world,&VehicleStateReference::psi_ddot_world},
  {"a_x_body_m_s2",         &VehicleState::ax_body,       &VehicleStateReference::ax_body},
  {"a_y_body_m_s2",         &VehicleState::ay_body,       &VehicleStateReference::ay_body},
  {"psi_ddot_body_rad_s2",  &VehicleState::psi_ddot_body, &VehicleStateReference::psi_ddot_body},
  {"v_x_world_m_s",         &VehicleState::vx_world,      &VehicleStateReference::vx_world},
  {"v_y_world_m_s",         &VehicleState::vy_world,      &VehicleStateReference::vy_world},
  {"psi_dot_world_rad_s",   &VehicleState::psi_dot_world, &VehicleStateReference::psi_dot_world},
  {"v_x_body_m_s",          &VehicleState::vx_body,       &VehicleStateReference::vx_body},
  {"v_y_body_m_s",          &VehicleState::vy_body,       &VehicleStateReference::vy_body},
  {"psi_dot_body_rad_s",    &VehicleState::psi_dot_body,  &VehicleStateReference::psi_dot_body},
  {"x_world_m",             &VehicleState::x_world,       &VehicleStateReference::x_world},
  {"y_world_m",             &VehicleState::y_world,       &VehicleStateReference::y_world},
  {"psi_world_rad",         &VehicleState::psi_world,     &VehicleStateReference::psi_world},
};

Vehicle::Vehicle(const std::string& fmu_path, double fixed_step_s, double start_time_s)
    : _fmu(fmu_path), _fixed_step_s(fixed_step_s) {
  if (_fixed_step_s <= 0.0) throw std::invalid_argument("Vehicle fixed step must be > 0");
  if (!_fmu.supports_cs()) throw std::runtime_error("FMU does not support co-simulation");

  _cs_fmu = _fmu.as_cs_fmu();
  _instance = _cs_fmu->new_instance(false, false);
  if (!_instance) throw std::runtime_error("Failed to create FMU co-simulation instance");

  _init_variable_references();
  _instance->setup_experiment(start_time_s);
  _instance->enter_initialization_mode();
  _instance->exit_initialization_mode();
  _write_fmu_input(_last_input);
  _unpack_fmu_state();
}

VehicleState Vehicle::state() const { return _current_state; }

double Vehicle::simulation_time_s() const { return _instance ? _instance->get_simulation_time() : 0.0; }

void Vehicle::update_state(const VehicleInput& input) { update_state(input, _fixed_step_s); }

void Vehicle::update_state(const VehicleInput& input, double step_size_s) {
  if (step_size_s <= 0.0) throw std::invalid_argument("FMU step_size_s must be > 0");

  _write_fmu_input(input);
  _instance->step(step_size_s);
  _last_input = input;
  _unpack_fmu_state();
}

fmi2ValueReference Vehicle::_get_value_reference(const std::string& name) const {
  const auto& var = _instance->get_model_description()->get_variable_by_name(name);
  if (!var.is_real()) throw std::runtime_error("FMU variable is not real: " + name);
  return var.as_real().valueReference();
}

void Vehicle::_init_variable_references() {
  for (auto& e : input_map)
    _input_vr.*e.vr = _get_value_reference(e.fmu_field);
  for (auto& e : state_map)
    _state_vr.*e.vr = _get_value_reference(e.fmu_field);
}

void Vehicle::_write_fmu_input(const VehicleInput& input) {
  std::vector<fmi2ValueReference> refs;
  std::vector<double> values;
  for (auto& e : input_map) {
    refs.push_back(_input_vr.*e.vr);
    values.push_back(input.*e.value);
  }
  _instance->write_real(refs, values);
}

void Vehicle::_unpack_fmu_state() {
  std::vector<fmi2ValueReference> refs;
  for (auto& e : state_map)
    refs.push_back(_state_vr.*e.vr);

  std::vector<double> values(refs.size(), 0.0);
  _instance->read_real(refs, values);

  for (size_t i = 0; i < std::size(state_map); ++i)
    _current_state.*state_map[i].value = values[i];
}

std::vector<fmi4cpp::fmi4cppByte> Vehicle::serialize_fmu_state() const {
  fmi4cpp::fmi4cppFMUstate raw = nullptr;
  _instance->get_fmu_state(raw);
  std::vector<fmi4cpp::fmi4cppByte> serialized;
  if (!_instance->serialize_fmu_state(raw, serialized)) throw std::runtime_error("FMU serialize_fmu_state failed");
  if (!_instance->free_fmu_state(raw)) throw std::runtime_error("FMU free_fmu_state failed after serialization");
  return serialized;
}

void Vehicle::deserialize_fmu_state(const std::vector<fmi4cpp::fmi4cppByte>& serialized) {
  if (serialized.empty()) throw std::invalid_argument("serialized_state must not be empty");
  fmi4cpp::fmi4cppFMUstate raw = nullptr;
  _instance->de_serialize_fmu_state(raw, serialized);
  if (!_instance->set_fmu_state(raw)) throw std::runtime_error("FMU set_fmu_state failed");
  if (!_instance->free_fmu_state(raw)) throw std::runtime_error("FMU free_fmu_state failed after restore");
  _unpack_fmu_state();
}

} // namespace sim
