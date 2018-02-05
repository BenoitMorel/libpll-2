#include "kernels.h"


void update_all_partials(std::shared_ptr<Dataset> dataset)
{
  pll_update_partials(dataset->partition, 
      dataset->operations, 
      dataset->ops_count);
}

double compute_likelihood(std::shared_ptr<Dataset> dataset)
{
  unsigned int params_indices[4] = {0,0,0,0};
  return pll_compute_edge_loglikelihood(dataset->partition,
      dataset->root->clv_index,
      dataset->root->scaler_index,
      dataset->root->back->clv_index,
      dataset->root->back->scaler_index,
      dataset->root->pmatrix_index,
      params_indices,
      NULL);
}
