# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(Stone)

# Set CMake type
IF (CMAKE_BUILD_TYPE)
ELSE ()
    SET(CMAKE_BUILD_TYPE "Debug")
ENDIF ()

SET(CMAKE_BUILD_TYPE "Debug")

SET(CMAKE_CXX_FLAGS_DEBUG " $ENV{CXXFLAGS} -std=c++11 -DDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -g -w")
SET(CMAKE_CXX_FLAGS_RELEASE " $ENV{CXXFLAGS} -std=c++11 -DNDEBUG -DUSE_CUDA -DDLIB_NO_GUI_SUPPORT -O3 -w")

# Configure the build to print all details.
SET(CMAKE_VERBOSE_MAKEFILE OFF)

# Configure the build type. This can be passed as parameter too.
#set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

# Find CUDA, you may need to set CUDA_DIR variable
# to the absolute path to the directory containing CUDAConfig.cmake file
# via the command line or GUI
SET(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-10.0)
find_package(CUDA REQUIRED)
# /usr/share/cmake/Modules/FindCUDA.cmake
INCLUDE(FindCUDA)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the CUDAConfig.cmake file.
# Print some message showing some of them
message(STATUS "CUDA library status:")
message(STATUS "    toolkit root dir: ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    version: ${CUDA_VERSION}")
message(STATUS "    libraries: ${CUDA_LIBRARIES}")
message(STATUS "    include path: ${CUDA_INCLUDE_DIRS}")

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

#Bring the headers, *.h into the project
include_directories(include ${CUDA_INCLUDE_DIRS})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

add_subdirectory(src)
add_subdirectory(test)