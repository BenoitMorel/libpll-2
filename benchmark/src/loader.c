
#include "loader.h"
#include <stdlib.h>
#include <stdio.h>

struct pll_sequence {
  pll_sequence(char *label, char *seq, unsigned int len):
    label(label),
    seq(seq),
    len(len) {}
  char *label;
  char *seq;
  unsigned int len;
  ~pll_sequence() {
    free(label);
    free(seq);
  }
};
 
Dataset *load_dataset(const char *newick_filename,
    const char *alignment_filename,
    unsigned int attribute,
    AlignmentFormat format,
    AlphabetType alphabet)
{
  Dataset *result = (Dataset*)malloc(sizeof(Dataset));
  
  return result;
}

void destroy_dataset(Dataset *dataset)
{
  //pll_partition_destroy(dataset->partition);
  //pll_utree_graph_destroy(dataset->tree);
  free(dataset);
}

