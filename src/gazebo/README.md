# Gazebo Models

Folder containing SDF models and meshes to build the simulated driverless track.

Pretty much all of these models were imported and refactored for Gazebo Harmonic from AMZ Racing's [fssim](https://github.com/AMZ-Racing/fssim). Massive thanks to them!!!

## Run and spawn the track

The world at `src/gazebo/worlds/track.sdf` now includes:

- `model://gazebo/models/track`

To make `model://gazebo/...` resolve on host, set the resource path to the repo `src` directory before launching Gazebo:

```bash
export GZ_SIM_RESOURCE_PATH="/path/to/stupid-fucking-simulator/src:${GZ_SIM_RESOURCE_PATH}"
gz sim "/path/to/stupid-fucking-simulator/src/gazebo/worlds/track.sdf"
```

Notes:

- The track SDF files under `src/gazebo/models/track/*.sdf` use Gazebo auto-generated include names (no manual `<name>` tags needed).
- In the container image, `GZ_SIM_RESOURCE_PATH=/workspace/src` is already set.
