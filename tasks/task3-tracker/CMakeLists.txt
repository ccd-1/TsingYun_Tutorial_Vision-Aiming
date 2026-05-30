set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(hw_task3_tracker
  src/kalman_tracker.cpp
)

target_include_directories(hw_task3_tracker
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(hw_task3_tracker_shared SHARED
  src/kalman_tracker.cpp
  src/kalman_tracker_c_wrapper.cpp
)

target_include_directories(hw_task3_tracker_shared
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(HW_BUILD_TESTS)
  add_executable(task3_tracker_tests
    ../../tests/cpp/task3_tracker_tests.cpp
  )
  target_link_libraries(task3_tracker_tests PRIVATE hw_task3_tracker)
  add_test(NAME task3_tracker_tests COMMAND task3_tracker_tests)
endif()
