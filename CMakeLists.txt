cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Deft)

# disable boost warning
add_definitions(-DBOOST_COROUTINES_NO_DEPRECATION_WARNING)

# Compiler options
set(CMAKE_CXX_STANDARD 20)
add_compile_options(-Wall -Wno-deprecated-declarations -Wsign-compare)

# 设置 libcuckoo 路径
set(LIBCUCKOO_DIR "/home/lyw/workspace/libcuckoo")  # 修改为您的实际安装路径

# 查找 gperftools 的 profiler 库
find_library(GPERFTOOLS_PROFILER profiler)
if(GPERFTOOLS_PROFILER)
    message(STATUS "Found gperftools profiler: ${GPERFTOOLS_PROFILER}")
    set(LINKS_FLAGS "-lcityhash -lboost_system -lboost_coroutine -lpthread -libverbs -lmemcached -lgflags -lprofiler")
else()
    message(WARNING "libprofiler not found - CPU profiling will be unavailable")
    set(LINKS_FLAGS "-lcityhash -lboost_system -lboost_coroutine -lpthread -libverbs -lmemcached -lgflags")
endif()

# 检查 libcuckoo 是否存在
if(EXISTS "${LIBCUCKOO_DIR}/include/libcuckoo/cuckoohash_map.hh")
    message(STATUS "Found libcuckoo: ${LIBCUCKOO_DIR}")
    # 不需要额外的链接库，因为 libcuckoo 是头文件库
else()
    message(FATAL_ERROR "libcuckoo not found at ${LIBCUCKOO_DIR} - required for concurrent hashmap support")
endif()

# Link Options
set(INCLUDE_BASE ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_BASE})

# 添加 parallel_hashmap 和 libcuckoo 头文件目录
include_directories(
    /home/lyw/parallel-hashmap
    ${LIBCUCKOO_DIR}/include
)

# Source file define
set(COMMON_SRC ${PROJECT_SOURCE_DIR}/src)

# Used by both server and clients
file(GLOB_RECURSE COMMON_FILE ${COMMON_SRC}/*.cpp)
add_library(deft STATIC ${COMMON_FILE})
link_libraries(deft)

file(GLOB TEST_SRC ${PROJECT_SOURCE_DIR}/test/*.cpp)
foreach (TEST ${TEST_SRC})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_executable(${TEST_NAME} ${TEST})
    target_link_libraries(${TEST_NAME} ${LINKS_FLAGS})
endforeach()