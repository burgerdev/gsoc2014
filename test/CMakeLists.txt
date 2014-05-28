

set(Python_ADDITIONAL_VERSIONS 2.7)
find_package(PythonLibs REQUIRED)
find_package(PythonInterp REQUIRED)
find_package(Numpy REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

include_directories(
    ${PYTHON_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PYTHON_NUMPY_INCLUDE_DIR}
)

add_executable(test_merge_labels testMergeLabels.cxx)
target_link_libraries(test_merge_labels
    ${Boost_PYTHON_LIBRARIES}
    ${PYTHON_LIBRARY}
    ${VIGRA_IMPEX_LIBRARY}
#    ${RT_LIBRARY}
)

add_test("test_merge_labels" test_merge_labels)