

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(Numpy REQUIRED)
#find_package(Boost COMPONENTS python REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

add_executable(benchmark_merge_labels benchmark_merge_labels.cxx)
target_link_libraries(benchmark_merge_labels
#    ${Boost_PYTHON_LIBRARIES}
    ${PYTHON_LIBRARY}
    ${VIGRA_IMPEX_LIBRARY}
#    ${RT_LIBRARY}
)

get_property(location TARGET benchmark_merge_labels PROPERTY LOCATION)
add_custom_command(TARGET benchmark_merge_labels
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${PROJECT_SOURCE_DIR}/benchmark/
)
