cmake_minimum_required(VERSION 3.5.1)
project(RLR)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Try to find OpenCV
find_package(OpenCV 3.4.6 REQUIRED)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()

# Try to find Boost
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Could not locate Boost")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES src/*.cpp)

set(TENSORFLOW_ROOT_DIR "/home/do/tensorflow")
INCLUDE_DIRECTORIES( ${TENSORFLOW_ROOT_DIR} )
set(PREBUILT_DIR ${TENSORFLOW_ROOT_DIR}/tensorflow/lite/tools/make)
add_library(lib_tflite STATIC IMPORTED )
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION
#${PREBUILT_DIR}/gen/linux_aarch64/lib/libtensorflow-lite.a)
${PREBUILT_DIR}/gen/linux_x86_64/lib/libtensorflow-lite.a)

INCLUDE_DIRECTORIES(${TENSORFLOW_ROOT_DIR}
    ${PREBUILT_DIR}/downloads/flatbuffers/include)


add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${OpenCV_LIBS}
        ${Boost_LIBRARIES} lib_tflite -lpthread
)
