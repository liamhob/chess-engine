cmake_minimum_required(VERSION 3.20)
project(alphazero_chess LANGUAGES CXX)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(chess_core STATIC src/position.cpp src/movegen.cpp src/zobrist.cpp src/attacks.cpp src/mcts.cpp src/concurrency.cpp src/search.cpp src/inference.cpp)
target_include_directories(chess_core PUBLIC include)
target_compile_features(chess_core PUBLIC cxx_std_20)

add_executable(phase1_foundations_test tests/phase1_foundations_test.cpp)
target_link_libraries(phase1_foundations_test PRIVATE chess_core)

add_executable(position_test tests/position_test.cpp)
target_link_libraries(position_test PRIVATE chess_core)

add_executable(movegen_test tests/movegen_test.cpp)
target_link_libraries(movegen_test PRIVATE chess_core)

add_executable(special_moves_test tests/special_moves_test.cpp)
target_link_libraries(special_moves_test PRIVATE chess_core)

add_executable(zobrist_test tests/zobrist_test.cpp)
target_link_libraries(zobrist_test PRIVATE chess_core)

add_executable(perft_gate tests/perft_gate.cpp)
target_link_libraries(perft_gate PRIVATE chess_core)

add_executable(phase1_state_test tests/phase1_state_test.cpp)
target_link_libraries(phase1_state_test PRIVATE chess_core)

add_executable(mcts_node_test tests/mcts_node_test.cpp)
target_link_libraries(mcts_node_test PRIVATE chess_core)

add_executable(mcts_concurrency_test tests/mcts_concurrency_test.cpp)
target_link_libraries(mcts_concurrency_test PRIVATE chess_core)

add_executable(mcts_search_test tests/mcts_search_test.cpp)
target_link_libraries(mcts_search_test PRIVATE chess_core)

add_executable(inference_queue_test tests/inference_queue_test.cpp)
target_link_libraries(inference_queue_test PRIVATE chess_core)

add_executable(fifty_move_test tests/fifty_move_test.cpp)
target_link_libraries(fifty_move_test PRIVATE chess_core)

pybind11_add_module(alphazero_cpp src/python_bindings.cpp)
target_link_libraries(alphazero_cpp PRIVATE chess_core)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME phase1_foundations COMMAND phase1_foundations_test)
    add_test(NAME position COMMAND position_test)
    add_test(NAME movegen COMMAND movegen_test)
    add_test(NAME special_moves COMMAND special_moves_test)
    add_test(NAME zobrist COMMAND zobrist_test)
    add_test(NAME perft_gate COMMAND perft_gate)
    add_test(NAME phase1_state COMMAND phase1_state_test)
    add_test(NAME mcts_node COMMAND mcts_node_test)
    add_test(NAME mcts_concurrency COMMAND mcts_concurrency_test)
    add_test(NAME mcts_search COMMAND mcts_search_test)
    add_test(NAME inference_queue COMMAND inference_queue_test)
    add_test(NAME fifty_move COMMAND fifty_move_test)
endif()
