cmake_minimum_required(VERSION 3.22.1)

project("baby_cc")

find_library(log-lib log)

# Removed find_package as we use manually imported library

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES IMPORTED_LOCATION
        ${CMAKE_CURRENT_SOURCE_DIR}/Libs/${ANDROID_ABI}/libonnxruntime.so)

add_library(
        baby_cc
        SHARED
        NativeBridge.cpp
        RespiratoryAnalyzer.cpp
        FeatureExtractor.cpp
)

target_link_libraries(
        baby_cc
        ${log-lib}
        android
        onnxruntime
)

target_link_options(baby_cc PRIVATE "-Wl,-z,max-page-size=16384" "-Wl,-z,common-page-size=16384")
