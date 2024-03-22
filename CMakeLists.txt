cmake_minimum_required(VERSION 3.13.4)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_definitions(-w)
set(PROJECT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PROJECT_BINARY_DIR ${PROJECT_SOURCE_DIR}/build)
set(TEST_PATH ${PROJECT_SOURCE_DIR}/test)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)


if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(POM LANGUAGES CXX C)
  # llvm
  set(LLVM_SOURCE_DIR ${PROJECT_SOURCE_DIR}/scalehls/polygeist/llvm-project/llvm)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
  message(STATUS "Using LLVM_EXTERNAL_LIT.cmake in: ${LLVM_EXTERNAL_LIT}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)
else ()
  set(LLVM_SOURCE_DIR ${LLVM_MAIN_SRC_DIR})
  set(MLIR_MAIN_SRC_DIR ${LLVM_MAIN_SRC_DIR}/../mlir)
  set(MLIR_INCLUDE_DIRS ${MLIR_MAIN_SRC_DIR}/include)
  set(MLIR_CMAKE_DIR ${MLIR_MAIN_SRC_DIR}/cmake/modules)
  set(MLIR_TABLEGEN_EXE $<TARGET_FILE:mlir-tblgen>)
  set(MLIR_TABLEGEN_OUTPUT_DIR ${LLVM_BINARY_DIR}/tools/mlir/include)
  include_directories(SYSTEM ${MLIR_INCLUDE_DIR})
  include_directories(SYSTEM ${MLIR_TABLEGEN_OUTPUT_DIR})
  message(STATUS "Using LLVM_SOURCE_DIR in: ${LLVM_SOURCE_DIR}")
  message(STATUS "Using MLIR_MAIN_SRC_DIR in: ${MLIR_MAIN_SRC_DIR}")
endif()

# include_directories("/home/POM/third_party/llvm-project/mlir/include/mlir/IR")
set(POM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(POM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(POM_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/include )
set(POM_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)


list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})



# ISL
#set(ISL_INCLUDE_DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/isl/build/include/ CACHE PATH #"Path to ISL include directory")
#set(ISL_LIB_DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/isl/build/.lib/ CACHE PATH "Path to #ISL library directory")

include_directories(${PROJECT_SOURCE_DIR}/include/polyhedral)
include_directories(${PROJECT_SOURCE_DIR}/lib/polyhedral)
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/testbench)
include_directories(${PROJECT_SOURCE_DIR}/scalehls/polygeist/llvm-project/mlir/include)
include_directories(${PROJECT_SOURCE_DIR}/scalehls/polygeist/llvm-project/llvm/include)
include_directories(${PROJECT_SOURCE_DIR}/scalehls/build/tools/scalehls/include)
include_directories($ISL_INCLUDE_DIRECTORY)
include_directories(${PROJECT_SOURCE_DIR}/scalehls/include)
#include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
include_directories(/usr/include/python3.8)


find_library(ISLLib isl PATHS /usr/local/lib NO_DEFAULT_PATH)
message(STATUS "Using ISLlib in: ${ISLLib}")



get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)


# find_library(MLIRSCF NAMES libMLIRSCF.a PATHS ${PROJECT_SOURCE_DIR}/scalehls/build/lib)
# message(STATUS "Using MLIRSCF in: ${MLIRSCF}")

find_library(MLIRHLS MLIRHLS PATHS ${PROJECT_SOURCE_DIR}/scalehls/build/lib)
message(STATUS "Using MLIRHLS in: ${MLIRHLS}")

find_library(MLIRScaleHLSSupport MLIRScaleHLSSupport PATHS ${PROJECT_SOURCE_DIR}/scalehls/build/lib)
message(STATUS "Using MLIRScaleHLSSupport in: ${MLIRScaleHLSSupport}")

find_library(MLIRScaleHLSTransforms MLIRScaleHLSTransforms PATHS ${PROJECT_SOURCE_DIR}/scalehls/build/lib)
message(STATUS "Using MLIRScaleHLSSupport in: ${MLIRScaleHLSTransforms}")


set(TEST_SOURCE_DIR ${PROJECT_SOURCE_DIR}/testbench)
file(GLOB_RECURSE  mains RELATIVE 
"${TEST_SOURCE_DIR}" 
"${TEST_SOURCE_DIR}/*.cpp"
)
message(STATUS "Using TEST_SOURCE_DIR in:${TEST_SOURCE_DIR}")

foreach(mainfile IN LISTS mains)
    # Get file name without directory
    get_filename_component(mainname ${mainfile} NAME_WE)
    # message(STATUS "get mainname in:${mainname}")
    add_executable(${mainname} ${TEST_SOURCE_DIR}/${mainfile})

    target_link_libraries(${mainname} Functions)
    target_link_libraries(${mainname} ${ISLLib})
    target_link_libraries(${mainname}
        ${dialect_libs}
        ${conversion_libs}
        ${translation_libs}
        ${MLIRScaleHLSTransforms}
        ${MLIRScaleHLSSupport}
        ${MLIRHLS}
        MLIRAffineTransforms
        MLIROptLib
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRIR
        MLIRParser
        MLIRPass
        MLIRSideEffectInterfaces
        MLIRSupport
        MLIRTransforms)
endforeach()

# add_executable(test test.cpp)
# add_executable(bicg bicg.cpp)
# target_link_libraries(test Functions)
# target_link_libraries(test ${ISLLib})
# target_link_libraries(test
#     ${dialect_libs}
#     ${conversion_libs}
#     ${translation_libs}
#     ${MLIRScaleHLSTransforms}
#     ${MLIRScaleHLSSupport}
#     ${MLIRHLSCpp}
#     MLIRAffineTransforms
#     MLIROptLib
#     MLIRAnalysis
#     MLIRCallInterfaces
#     MLIRCastInterfaces
#     MLIRIR
#     MLIRParser
#     MLIRPass
#     MLIRSideEffectInterfaces
#     MLIRSupport
#     MLIRTransforms)



add_subdirectory(include)
add_subdirectory(lib)
#add_subdirectory(test)
#add_subdirectory(standalone-opt)
#add_subdirectory(standalone-translate)
#add_subdirectory(POM-isl)
# add_subdirectory(pybind11)


#find_package(pybind11 REQUIRED)
#pybind11_add_module(wrapper wrapper.cpp)
#target_link_libraries(wrapper PRIVATE ${ISLLib})

# pybind11_add_module(core ${PROJECT_SOURCE_DIR}/lib/Polyhedral/core.cpp)
# pybind11_add_module(place ${PROJECT_SOURCE_DIR}/lib/Polyhedral/placeholer.cpp)
# pybind11_add_module(example2 ${PROJECT_SOURCE_DIR}/lib/Polyhedral/example2.cpp)


# add_library(core MODULE 
#            ${PROJECT_SOURCE_DIR}/lib/Polyhedral/core.cpp
#     )
# target_link_libraries(core 
#             pybind11::module
#   )

