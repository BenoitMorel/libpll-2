#ifndef _LOADER_
#define _LOADER_

#include "pll.h"
#include "constants.h"

typedef struct {
  const char *name;
  pll_partition_t *partition;
  pll_utree_t *tree;
} Dataset;

Dataset *load_dataset(const char *newick_filename,
    const char *alignment_filename,
    unsigned int attribute,
    AlignmentFormat format,
    AlphabetType alphabet);

void destroy_dataset(Dataset *dataset);

#endif
