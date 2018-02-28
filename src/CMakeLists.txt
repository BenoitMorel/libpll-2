
include(CheckIncludeFile) 

set (LIBPLL_BASE_FLAGS "-Wall -Wsign-compare -D_GNU_SOURCE -O3 -fPIC ")

file(GLOB LIBPLL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/compress.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_derivatives.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_likelihood.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_partials.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_pmatrix.c
  ${CMAKE_CURRENT_SOURCE_DIR}/fasta.c
  ${CMAKE_CURRENT_SOURCE_DIR}/fast_parsimony.c
  ${CMAKE_CURRENT_SOURCE_DIR}/gamma.c
  ${CMAKE_CURRENT_SOURCE_DIR}/hardware.c
  ${CMAKE_CURRENT_SOURCE_DIR}/likelihood.c
  ${CMAKE_CURRENT_SOURCE_DIR}/list.c
  ${CMAKE_CURRENT_SOURCE_DIR}/maps.c
  ${CMAKE_CURRENT_SOURCE_DIR}/models.c
  ${CMAKE_CURRENT_SOURCE_DIR}/output.c
  ${CMAKE_CURRENT_SOURCE_DIR}/parsimony.c
  ${CMAKE_CURRENT_SOURCE_DIR}/partials.c
  ${CMAKE_CURRENT_SOURCE_DIR}/phylip.c
  ${CMAKE_CURRENT_SOURCE_DIR}/pll.c
  ${CMAKE_CURRENT_SOURCE_DIR}/random.c
  ${CMAKE_CURRENT_SOURCE_DIR}/repeats.c
  ${CMAKE_CURRENT_SOURCE_DIR}/rtree.c
  ${CMAKE_CURRENT_SOURCE_DIR}/stepwise.c
  ${CMAKE_CURRENT_SOURCE_DIR}/utree.c
  ${CMAKE_CURRENT_SOURCE_DIR}/utree_moves.c
  ${CMAKE_CURRENT_SOURCE_DIR}/utree_svg.c
  )

file(GLOB LIBPLL_SSE_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core_derivatives_sse.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_likelihood_sse.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_partials_sse.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_pmatrix_sse.c
  )

file(GLOB LIBPLL_AVX_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core_derivatives_avx.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_likelihood_avx.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_partials_avx.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_pmatrix_avx.c
  )

file(GLOB LIBPLL_AVX2_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/core_derivatives_avx2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_likelihood_avx2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_partials_avx2.c
  ${CMAKE_CURRENT_SOURCE_DIR}/core_pmatrix_avx2.c
  )

# check that user did not disable simd
if (NOT DEFINED ENABLE_SSE)
  SET(ENABLE_SSE "True")
endif ()
if (NOT DEFINED ENABLE_AVX)
  SET(ENABLE_AVX "True")
endif ()
if (NOT DEFINED ENABLE_AVX2)
  SET(ENABLE_AVX2 "True")
endif ()

# check simd installed 
# source: https://cmake.org/pipermail/cmake/2010-March/036115.html
if (ENABLE_SSE)
  set(CMAKE_REQUIRED_C_FLAGS_SAVE ${CMAKE_REQUIRED_C_FLAGS})
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS} "-msse3")
  CHECK_INCLUDE_FILE("pmmintrin.h" ENABLE_SSE)
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS_SAVE})
endif()
if (ENABLE_AVX)
  set(CMAKE_REQUIRED_C_FLAGS_SAVE ${CMAKE_REQUIRED_C_FLAGS})
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS} "-mavx")
  CHECK_INCLUDE_FILE("immintrin.h" ENABLE_AVX)
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS_SAVE})
endif()
if (ENABLE_AVX2)
  set(CMAKE_REQUIRED_C_FLAGS_SAVE ${CMAKE_REQUIRED_C_FLAGS})
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS} "-mfma -mavx2")
  CHECK_INCLUDE_FILE("zmmintrin.h" ENABLE_AVX2)
  set(CMAKE_REQUIRED_C_FLAGS ${CMAKE_REQUIRED_C_FLAGS_SAVE})
endif()


# set simd flags
if (${ENABLE_SSE})
  add_definitions(-DHAVE_SSE)
  set(SIMD_FLAGS "${SIMD_FLAGS} -msse3")
  message(STATUS "SSE enabled. To disable it, run cmake with -DENABLE_SSE=false")
  set(LIBPLL_SOURCES ${LIBPLL_SOURCES} ${LIBPLL_SSE_SOURCES})
endif ()
if (${ENABLE_AVX})
  add_definitions(-DHAVE_AVX)
  set(SIMD_FLAGS "${SIMD_FLAGS} -mavx")
  message(STATUS "AVX enabled. To disable it, run cmake with -DENABLE_AVX=false")
  set(LIBPLL_SOURCES ${LIBPLL_SOURCES} ${LIBPLL_AVX_SOURCES})
endif ()
if (${ENABLE_AVX2})
  add_definitions(-DHAVE_AVX2)
  set(SIMD_FLAGS "${SIMD_FLAGS} -mfma -mavx2")
  message(STATUS "AVX2 enabled. To disable it, run cmake with -DENABLE_AVX2=false")
  set(LIBPLL_SOURCES ${LIBPLL_SOURCES} ${LIBPLL_AVX2_SOURCES})
endif ()



add_definitions(-DHAVE_X86INTRIN_H)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${LIBPLL_BASE_FLAGS} ${SIMD_FLAGS}")

add_library(pll OBJECT ${LIBPLL_SOURCES})
set_property(TARGET pll PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library(pll_shared SHARED $<TARGET_OBJECTS:pll>)
add_library(pll_static STATIC $<TARGET_OBJECTS:pll>)
