
add_definitions(-DHAVE_SSE)
add_definitions(-DHAVE_AVX)
add_definitions(-DHAVE_AVX2)
add_definitions(-DHAVE_X86INTRIN_H)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wsign-compare -D_GNU_SOURCE -O3 -fPIC -mfma -mavx2 -mavx -msse3")

file(GLOB LIBPLL_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.c)
add_library(pll-static STATIC ${LIBPLL_SOURCES})
add_library(pll-shared SHARED ${LIBPLL_SOURCES})

add_library(pll OBJECT ${LIBPLL_SOURCES})
set_property(TARGET pll PROPERTY POSITION_INDEPENDENT_CODE 1)
add_library(pll_shared SHARED $<TARGET_OBJECTS:pll>)
add_library(pll_static STATIC $<TARGET_OBJECTS:pll>)
