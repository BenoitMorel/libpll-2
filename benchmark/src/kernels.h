#ifndef _BENCH_KERNEL_H
#define _BENCH_KERNEL_H
#include "loader.h"

void update_all_partials(std::shared_ptr<Dataset> dataset);
double compute_likelihood(std::shared_ptr<Dataset> dataset);

#endif
