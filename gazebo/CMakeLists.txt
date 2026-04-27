cmake_minimum_required(VERSION 3.20)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)
find_package(fmi4cpp REQUIRED)


target_include_directories(gazebo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(gazebo PUBLIC
    gz-sim8::gz-sim8
    gz-transport13::gz-transport13
    gz-msgs10::gz-msgs10
    fmi4cpp::fmi4cpp
)
