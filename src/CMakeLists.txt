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

add_library(_lazycc_cxx SHARED
    lazycc.cxx
)

target_link_libraries(_lazycc_cxx
    ${Boost_PYTHON_LIBRARIES}
    ${PYTHON_LIBRARY}
    ${VIGRA_IMPEX_LIBRARY}
#    ${RT_LIBRARY}
)

set_target_properties(_lazycc_cxx PROPERTIES OUTPUT_NAME "_lazycc_cxx" PREFIX "")

get_property(location TARGET _lazycc_cxx PROPERTY LOCATION)
add_custom_command(TARGET _lazycc_cxx
    POST_BUILD        
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        ${location}
        ${PROJECT_SOURCE_DIR}/lazycc/
)

