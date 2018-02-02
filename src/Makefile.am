lib_LTLIBRARIES=libpll.la

AM_CPPFLAGS=-Wall -Wsign-compare -D_GNU_SOURCE -std=c++11 -O3 -fPIC -mfma -mavx2 -mavx -msse3 -fpermissive

LIBPLLHEADERS=\
pll.h

AM_YFLAGS = -d -p `${SED} -n 's/.*_\(.*\)/pll_\1_/p' <<<"$*"`
AM_LFLAGS = -P `${SED} -n 's/.*_\(.*\)/pll_\1_/p' <<<"$*"` -o lex.yy.c

libpll_la_SOURCES=\
fasta.cpp \
gamma.cpp \
likelihood.cpp \
list.cpp \
maps.cpp \
models.cpp \
pll.cpp \
output.cpp \
utree.cpp \
rtree.cpp \
derivatives.cpp \
partials.cpp \
compress.cpp \
utree_moves.cpp \
utree_svg.cpp \
parsimony.cpp \
core_derivatives.cpp \
core_partials.cpp \
core_pmatrix.cpp \
core_likelihood.cpp \
parse_utree.y \
parse_rtree.y \
lex_utree.l \
lex_rtree.l \
fast_parsimony.cpp \
stepwise.cpp \
random.cpp \
phylip.cpp \
hardware.cpp \
core_partials_avx2.cpp \
core_derivatives_avx2.cpp \
core_pmatrix_avx2.cpp \
core_likelihood_avx2.cpp \
fast_parsimony_avx2.cpp \
core_partials_avx.cpp \
core_derivatives_avx.cpp \
core_pmatrix_avx.cpp \
core_likelihood_avx.cpp \
fast_parsimony_avx.cpp\
core_partials_sse.cpp \
core_derivatives_sse.cpp \
core_likelihood_sse.cpp \
core_pmatrix_sse.cpp \
fast_parsimony_sse.cpp\
repeats.cpp 

libpll_la_CPPFLAGS = $(AM_CPPFLAGS)


# This is the version information for libtool and not the libpll version. For
#  more information see:
# https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
libpll_la_LDFLAGS = -version-info 0:0:0

pkgincludedir=$(includedir)/libpll
pkginclude_HEADERS = pll.h
