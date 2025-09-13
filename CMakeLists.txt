cmake_minimum_required(VERSION 3.24)
project(png2slime)

set(BINARY_DIR ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_DEBUG ${BINARY_DIR})
make_directory(${BINARY_DIR})

add_subdirectory(SDL)
add_executable(png2slime WIN32 main.c util.c stb.c spirv_reflect.c)
set_target_properties(png2slime PROPERTIES C_STANDARD 11)
target_link_libraries(png2slime SDL3::SDL3)
if(UNIX)
    target_link_libraries(png2slime m)
endif()

function(spirv FILE)
    set(OUTPUT ${BINARY_DIR}/${FILE})
    add_custom_command(
        OUTPUT ${OUTPUT}
        COMMAND glslc ${FILE} -o ${OUTPUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        DEPENDS ${FILE} config.h
        BYPRODUCTS ${OUTPUT}
        COMMENT ${FILE}
    )
    string(REPLACE . _ NAME ${FILE})
    add_custom_target(${NAME} DEPENDS ${OUTPUT})
    add_dependencies(png2slime ${NAME})
endfunction()
spirv(blur.comp)
spirv(draw.frag)
spirv(quad.vert)
spirv(update.comp)

configure_file(LICENSE.txt ${BINARY_DIR} COPYONLY)
configure_file(README.md ${BINARY_DIR} COPYONLY)