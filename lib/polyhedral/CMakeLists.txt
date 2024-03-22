get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
# include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
# include_directories(/usr/include/python3.8)
aux_source_directory(. DIR_LIB_SRCS)
# 编译成静态库文件
# add_library(Functions  ${DIR_LIB_SRCS})
add_library(Functions  ${DIR_LIB_SRCS})
