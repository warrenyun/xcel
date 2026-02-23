cmake_minimum_required(VERSION 3.20)
project(ht_dv_sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

include_directories(/usr/local/include/fmu_tools)

add_library(my_plugin SHARED
    src/main.cpp
)

target_link_libraries(my_plugin
    gz-sim8::gz-sim8
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
)

# Install plugin to where Gazebo can find it
install(TARGETS my_plugin
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/gz-sim-8/plugins
)
