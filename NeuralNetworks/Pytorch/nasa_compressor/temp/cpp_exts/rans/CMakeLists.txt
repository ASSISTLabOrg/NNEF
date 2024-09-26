cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(tiny_compressai_cpp)

list(APPEND CMAKE_PREFIX_PATH "/home/rafa/.local/lib/python3.11/site-packages/torch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Define our library target
add_library(tiny_compressai_rans SHARED rans_interface.cpp)
target_include_directories(tiny_compressai_rans PUBLIC ../../../third_party/ryg_rans )
# Link against LibTorch
target_link_libraries(tiny_compressai_rans "${TORCH_LIBRARIES}")

#$ mkdir build
#$ cd build
#$ cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" ..
#$ make -j