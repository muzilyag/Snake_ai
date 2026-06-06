cmake_minimum_required(VERSION 3.12)
project(snake_cpp_engine)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 CONFIG REQUIRED PATHS ${pybind11_DIR})

pybind11_add_module(snake_cpp 
    src/engine/pybind_module.cpp 
    src/engine/vec_env.cpp 
    src/engine/systems/physics.cpp
    src/engine/systems/radar.cpp
    src/engine/systems/rewards.cpp
)

set_target_properties(snake_cpp PROPERTIES SUFFIX ".pyd")

if(MSVC)
    target_compile_options(snake_cpp PRIVATE /O2 /fp:fast)
else()
    target_compile_options(snake_cpp PRIVATE -O3 -ffast-math -march=native)
    target_link_options(snake_cpp PRIVATE -static -static-libgcc -static-libstdc++)
endif()