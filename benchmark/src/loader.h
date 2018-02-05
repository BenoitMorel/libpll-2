#ifndef _LOADER_
#define _LOADER_

#include "pll.h"
#include "constants.h"
#include <memory>
#include <string>


struct Dataset {
  std::string name;
  pll_partition_t *partition;
  pll_utree_t *tree;
  pll_operation_t *operations;
  unsigned int ops_count;
  pll_unode_t *root;
};

std::shared_ptr<Dataset> loadDataset(const std::string &newickFilename,
    const std::string &alignmentFilename,
    unsigned int attribute,
    AlignmentFormat format,
    AlphabetType alphabet);

void destroy_dataset(Dataset *dataset);

#endif
