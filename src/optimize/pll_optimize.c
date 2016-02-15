/*
 Copyright (C) 2015 Diego Darriba

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU Affero General Public License as
 published by the Free Software Foundation, either version 3 of the
 License, or (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Affero General Public License for more details.

 You should have received a copy of the GNU Affero General Public License
 along with this program.  If not, see <http://www.gnu.org/licenses/>.

 Contact: Diego Darriba <Diego.Darriba@h-its.org>,
 Exelixis Lab, Heidelberg Instutute for Theoretical Studies
 Schloss-Wolfsbrunnenweg 35, D-69118 Heidelberg, Germany
 */
#include "pll_optimize.h"
#include "lbfgsb/lbfgsb.h"

#define UPDATE_SCALERS 1
#define BL_OPT_METHOD PLL_BRANCH_OPT_NEWTON

static inline int is_nan(double v)
{
  return v!=v;
}

static int v_int_max (int * v, int n)
{
  int i, max = v[0];
  for (i = 1; i < n; i++)
    if (v[i] > max)
      max = v[i];
  return max;
}

static inline int d_equals(double a, double b)
{
  return (fabs(a-b) < 1e-10);
}

static int set_x_to_parameters(pll_optimize_options_t * params,
                               double *x)
{
  pll_partition_t * partition = params->lk_params.partition;
  pll_operation_t * operations = params->lk_params.operations;
  double * branch_lengths = params->lk_params.branch_lengths;
  unsigned int * matrix_indices = params->lk_params.matrix_indices;
  unsigned int params_index = params->params_index;
  unsigned int mixture_index = params->mixture_index;
  unsigned int n_branches, n_inner_nodes;
  double * xptr = x;

  if (params->lk_params.rooted)
  {
    n_branches = 2 * partition->tips - 2;
    n_inner_nodes = partition->tips - 1;
  }
  else
  {
    n_branches = 2 * partition->tips - 3;
    n_inner_nodes = partition->tips - 2;
  }

  /* update substitution rate parameters */
  if (params->which_parameters & PLL_PARAMETER_SUBST_RATES)
  {
    int * symm;
    int n_subst_rates;
    double * subst_rates;

    symm = params->subst_params_symmetries;
    n_subst_rates = partition->states * (partition->states - 1) / 2;
    subst_rates = (double *) malloc ((size_t) n_subst_rates * sizeof(double));

    assert(subst_rates);

    if (symm)
    {
      int i, j, k;
      int n_subst_free_params = 0;

      /* compute the number of free parameters */
      n_subst_free_params = v_int_max (symm, n_subst_rates);

      /* assign values to the substitution rates */
      k = 0;
      for (i = 0; i <= n_subst_free_params; i++)
      {
        double next_value = (i == symm[n_subst_rates - 1]) ? 1.0 : xptr[k++];
        for (j = 0; j < n_subst_rates; j++)
          if (symm[j] == i)
          {
            subst_rates[j] = next_value;
          }
      }
      xptr += n_subst_free_params;
    }
    else
    {
      memcpy (subst_rates, xptr, ((size_t)n_subst_rates - 1) * sizeof(double));
      subst_rates[n_subst_rates - 1] = 1.0;
      xptr += n_subst_rates-1;
    }

    pll_set_subst_params (partition,
                          params_index,
                          mixture_index,
                          subst_rates);
    free (subst_rates);
  }

  /* update stationary frequencies */
  if (params->which_parameters & PLL_PARAMETER_FREQUENCIES)
  {
    unsigned int i;
    unsigned int n_states = partition->states;
    unsigned int cur_index;
    double sum_ratios = 1.0;
    double *freqs = (double *) malloc ((size_t) n_states * sizeof(double));
    for (i = 0; i < (n_states - 1); ++i)
    {
      assert(!is_nan(xptr[i]));
      sum_ratios += xptr[i];
    }
    cur_index = 0;
    for (i = 0; i < (n_states); ++i)
    {
      if (i != params->highest_freq_state)
      {
        freqs[i] = xptr[cur_index] / sum_ratios;
        cur_index++;
      }
    }
    freqs[params->highest_freq_state] = 1.0 / sum_ratios;
    pll_set_frequencies (partition,
                         params_index,
                         mixture_index,
                         freqs);
    free (freqs);
    xptr += (n_states - 1);
  }
  /* update proportion of invariant sites */
  if (params->which_parameters & PLL_PARAMETER_PINV)
  {
    assert(!is_nan(xptr[0]));
    if (!pll_update_invariant_sites_proportion (partition,
                                                params_index,
                                                xptr[0]))
    {
      return PLL_FAILURE;
    }
    xptr++;
  }
  /* update gamma shape parameter */
  if (params->which_parameters & PLL_PARAMETER_ALPHA)
  {
    assert(!is_nan(xptr[0]));
    /* assign discrete rates */
    double * rate_cats;
    rate_cats = malloc((size_t)partition->rate_cats * sizeof(double));
    params->lk_params.alpha_value = xptr[0];
    if (!pll_compute_gamma_cats (xptr[0], partition->rate_cats, rate_cats))
    {
      return PLL_FAILURE;
    }
    pll_set_category_rates (partition, rate_cats);

    free(rate_cats);
    xptr++;
  }

  /* update free rates */
  if (params->which_parameters & PLL_PARAMETER_FREE_RATES)
  {
    pll_set_category_rates (partition, xptr);
    xptr += params->lk_params.partition->rate_cats;
  }

  /* update rate weights */
  if (params->which_parameters & PLL_PARAMETER_RATE_WEIGHTS)
  {
    unsigned int i;
        unsigned int n_rate_cats = params->lk_params.partition->rate_cats;
        unsigned int cur_index;
        double sum_ratios = 1.0;
        double *weights = (double *) malloc((size_t)n_rate_cats*sizeof(double));
        for (i = 0; i < (n_rate_cats - 1); ++i)
        {
          assert(!is_nan(xptr[i]));
          sum_ratios += xptr[i];
        }
        cur_index = 0;
        for (i = 0; i < (n_rate_cats); ++i)
        {
          if (i != params->highest_freq_state)
          {
            weights[i] = xptr[cur_index] / sum_ratios;
            cur_index++;
          }
        }
        weights[params->highest_freq_state] = 1.0 / sum_ratios;
        pll_set_category_weights(partition, weights);
        free (weights);
        xptr += (n_rate_cats - 1);
  }

  /* update all branch lengths */
  if (params->which_parameters & PLL_PARAMETER_BRANCHES_ALL)
  {
    /* assign branch lengths */
    memcpy (branch_lengths, xptr, (size_t)n_branches * sizeof(double));
    xptr += n_branches;
  }

  /* update single branch */
  if (params->which_parameters & PLL_PARAMETER_BRANCHES_SINGLE)
   {
    assert(!is_nan(xptr[0]));
     /* assign branch length */
     *branch_lengths = *xptr;
     pll_update_prob_matrices (partition,
                         0,
                         &params->lk_params.where.unrooted_t.edge_pmatrix_index,
                         xptr,
                         1);
     xptr += 1;
   }
  else
   {
       pll_update_prob_matrices (partition, 0, matrix_indices, branch_lengths,
                                 n_branches);

       pll_update_partials (partition, operations, n_inner_nodes);
   }
  return PLL_SUCCESS;
}

static double utree_derivative_func (void * parameters, double proposal,
                                     double *df, double *ddf)
{
  pll_newton_tree_params_t * params = (pll_newton_tree_params_t *) parameters;
  double score = pll_compute_likelihood_derivatives (
      params->partition,
      params->tree->scaler_index,
      params->tree->back->scaler_index,
      proposal,
      params->params_index, params->freqs_index,
      params->sumtable, df, ddf);
  return score;
}

static double compute_negative_lnl_unrooted (void * p, double *x)
{
  pll_optimize_options_t * params = (pll_optimize_options_t *) p;
  pll_partition_t * partition = params->lk_params.partition;
  double score;

  if (x && !set_x_to_parameters(params, x))
    return -INFINITY;

  if (params->lk_params.rooted)
  {
    score = -1
        * pll_compute_root_loglikelihood (
            partition, params->lk_params.where.rooted_t.root_clv_index,
            params->lk_params.where.rooted_t.scaler_index,
            params->lk_params.freqs_index);
  }
  else
  {
    score = -1
        * pll_compute_edge_loglikelihood (
            partition, params->lk_params.where.unrooted_t.parent_clv_index,
            params->lk_params.where.unrooted_t.parent_scaler_index,
            params->lk_params.where.unrooted_t.child_clv_index,
            params->lk_params.where.unrooted_t.child_scaler_index,
            params->lk_params.where.unrooted_t.edge_pmatrix_index,
            params->lk_params.freqs_index);
  }

  return score;
} /* compute_lnl_unrooted */

static unsigned int count_n_free_variables (pll_optimize_options_t * params)
{
  unsigned int num_variables = 0;
  pll_partition_t * partition = params->lk_params.partition;

  /* count number of variables for dynamic allocation */
  if (params->which_parameters & PLL_PARAMETER_SUBST_RATES)
  {
    int n_subst_rates = partition->states * (partition->states - 1) / 2;
    num_variables +=
        params->subst_params_symmetries ?
            (unsigned int) v_int_max (params->subst_params_symmetries, n_subst_rates) :
            (unsigned int) n_subst_rates - 1;
  }
  if (params->which_parameters & PLL_PARAMETER_FREQUENCIES)
    num_variables += partition->states - 1;
  num_variables += (params->which_parameters & PLL_PARAMETER_PINV) != 0;
  num_variables += (params->which_parameters & PLL_PARAMETER_ALPHA) != 0;
  if (params->which_parameters & PLL_PARAMETER_FREE_RATES)
    num_variables += partition->rate_cats;
  if (params->which_parameters & PLL_PARAMETER_RATE_WEIGHTS)
    num_variables += partition->rate_cats - 1;
  num_variables += (params->which_parameters & PLL_PARAMETER_BRANCHES_SINGLE)
      != 0;
  if (params->which_parameters & PLL_PARAMETER_BRANCHES_ALL)
  {
    unsigned int num_branch_lengths =
        params->lk_params.rooted ?
            (2 * partition->tips - 3) : (2 * partition->tips - 2);
    num_variables += num_branch_lengths;
  }
  return num_variables;
} /* count_n_free_variables */


/******************************************************************************/
/* BRENT'S OPTIMIZATION */
/******************************************************************************/

static double brent_target(void * p, double x)
{
  return compute_negative_lnl_unrooted(p, &x);
}

static void set_range(double abs_min,
                      double defaultv,
                      double abs_max,
                      double deltav,
                      double *guess,
                      double *minv,
                      double *maxv)
{
        *minv = abs_min;
        *maxv = abs_max;
        if (*guess < abs_min || *guess > abs_max)
        {
          *guess = defaultv;
        }
        else
        {
          *minv = max(abs_min, *guess - deltav);
          *maxv = min(abs_max, *guess + deltav);
        }
}

PLL_EXPORT double pll_optimize_parameters_brent(pll_optimize_options_t * params)
{
  double score = 0;

  /* Brent parameters */
  double xmin;
  double xguess;
  double xmax;
  double f2x;

  switch (params->which_parameters)
  {
    case PLL_PARAMETER_ALPHA:
      xguess = params->lk_params.alpha_value;
      set_range (PLL_OPT_MIN_ALPHA,
                 PLL_OPT_DEFAULT_ALPHA,
                 PLL_OPT_MAX_ALPHA,
                 PLL_OPT_ALPHA_OFFSET,
                 &xguess, &xmin, &xmax);
      break;
    case PLL_PARAMETER_PINV:
      xguess = params->lk_params.partition->prop_invar[params->params_index];
      set_range (PLL_OPT_MIN_PINV,
                 PLL_OPT_DEFAULT_PINV,
                 PLL_OPT_MAX_PINV,
                 PLL_OPT_PINV_OFFSET,
                 &xguess, &xmin, &xmax);
      break;
    case PLL_PARAMETER_BRANCHES_SINGLE:
      xguess = params->lk_params.branch_lengths[0];
      set_range (PLL_OPT_MIN_BRANCH_LEN,
                 PLL_OPT_DEFAULT_BRANCH_LEN,
                 PLL_OPT_MAX_BRANCH_LEN,
                 PLL_OPT_BRANCH_LEN_OFFSET,
                 &xguess, &xmin, &xmax);
      break;
    default:
      /* unavailable or multiple parameter */
      return -INFINITY;
  }

  double xres = pll_minimize_brent(xmin, xguess, xmax,
                                   params->pgtol,
                                   &score, &f2x,
                                   (void *) params,
                                   &brent_target);
  score = brent_target(params, xres);
  return score;
}

/******************************************************************************/
/* L-BFGS-B OPTIMIZATION */
/******************************************************************************/

PLL_EXPORT double pll_optimize_parameters_lbfgsb (
                                              pll_optimize_options_t * params)
{
  unsigned int i;
  pll_partition_t * partition = params->lk_params.partition;

  /* L-BFGS-B parameters */
  //  double initial_score;
  unsigned int num_variables;
  double score = 0;
  double *x, *lower_bounds, *upper_bounds;
  int *bound_type;

  /* ensure that the 2 branch optimization modes are not set together */
  assert(!((params->which_parameters & PLL_PARAMETER_BRANCHES_ALL)
      && (params->which_parameters & PLL_PARAMETER_BRANCHES_SINGLE)));

  num_variables = count_n_free_variables (params);

  x = (double *) calloc ((size_t) num_variables, sizeof(double));
  lower_bounds = (double *) calloc ((size_t) num_variables, sizeof(double));
  upper_bounds = (double *) calloc ((size_t) num_variables, sizeof(double));
  bound_type = (int *) calloc ((size_t) num_variables, sizeof(int));

  {
    int * nbd_ptr = bound_type;
    double * l_ptr = lower_bounds, *u_ptr = upper_bounds;
    unsigned int check_n = 0;

    /* substitution rate parameters */
    if (params->which_parameters & PLL_PARAMETER_SUBST_RATES)
    {
      unsigned int n_subst_rates;
      unsigned int n_subst_free_params;

      n_subst_rates = partition->states * (partition->states - 1) / 2;
      if (params->subst_params_symmetries)
      {
        n_subst_free_params =(unsigned int) v_int_max (
                                         params->subst_params_symmetries,
                                         (int) n_subst_rates);
      }
      else
      {
        n_subst_free_params = n_subst_rates;
      }

      int current_rate = 0;
      for (i = 0; i < n_subst_free_params; i++)
      {
        nbd_ptr[i] = PLL_LBFGSB_BOUND_BOTH;
        unsigned int j = i;
        if (params->subst_params_symmetries)
        {
          if (params->subst_params_symmetries[n_subst_rates-1] == current_rate)
            current_rate++;
          for (j=0; j<n_subst_rates; j++)
          {
            if (params->subst_params_symmetries[j] == current_rate)
              break;
          }
          current_rate++;
        }

        x[check_n + i] = partition->subst_params[params->params_index][j];
        set_range (PLL_OPT_MIN_SUBST_RATE,
                   PLL_OPT_DEFAULT_RATE_RATIO,
                   PLL_OPT_MAX_SUBST_RATE,
                   PLL_OPT_SUBST_RATE_OFFSET,
                   &x[check_n + i], &l_ptr[i], &u_ptr[i]);
      }
      nbd_ptr += n_subst_free_params;
      l_ptr += n_subst_free_params;
      u_ptr += n_subst_free_params;
      check_n += n_subst_free_params;
    }

    /* stationary frequency parameters */
    if (params->which_parameters & PLL_PARAMETER_FREQUENCIES)
    {
      unsigned int states = params->lk_params.partition->states;
      unsigned int n_freqs_free_params = states - 1;
      unsigned int cur_index;

      double * frequencies =
          params->lk_params.partition->frequencies[params->params_index];

      params->highest_freq_state = 3;
      for (i = 1; i < states; i++)
              if (frequencies[i] > frequencies[params->highest_freq_state])
                params->highest_freq_state = i;

      cur_index = 0;
      for (i = 0; i < states; i++)
      {
        if (i != params->highest_freq_state)
        {
          nbd_ptr[cur_index] = PLL_LBFGSB_BOUND_BOTH;
          x[check_n + cur_index] = frequencies[i]
              / frequencies[params->highest_freq_state];
          set_range (PLL_OPT_MIN_FREQ,
                     PLL_OPT_DEFAULT_FREQ_RATIO,
                     PLL_OPT_MAX_FREQ,
                     PLL_OPT_FREQ_OFFSET,
                     &x[check_n + cur_index],
                     &l_ptr[cur_index], &u_ptr[cur_index]);
          cur_index++;
        }
      }
      check_n += n_freqs_free_params;
      nbd_ptr += n_freqs_free_params;
      l_ptr += n_freqs_free_params;
      u_ptr += n_freqs_free_params;
    }

    /* proportion of invariant sites */
    if (params->which_parameters & PLL_PARAMETER_PINV)
    {
      *nbd_ptr = PLL_LBFGSB_BOUND_BOTH;
      x[check_n] = partition->prop_invar[params->params_index];
      set_range (PLL_OPT_MIN_PINV + PLL_LBFGSB_ERROR,
                 PLL_OPT_DEFAULT_PINV,
                 PLL_OPT_MAX_PINV,
                 PLL_OPT_PINV_OFFSET,
                 &x[check_n], l_ptr, u_ptr);
      check_n += 1;
      nbd_ptr++;
      l_ptr++;
      u_ptr++;
    }

    /* gamma shape parameter */
    if (params->which_parameters & PLL_PARAMETER_ALPHA)
    {
      *nbd_ptr = PLL_LBFGSB_BOUND_BOTH;
      x[check_n] = params->lk_params.alpha_value;
      set_range (PLL_OPT_MIN_ALPHA,
                 PLL_OPT_DEFAULT_ALPHA,
                 PLL_OPT_MAX_ALPHA,
                 PLL_OPT_ALPHA_OFFSET,
                 &x[check_n], l_ptr, u_ptr);
      check_n += 1;
      nbd_ptr++;
      l_ptr++;
      u_ptr++;
    }

    /* update free rates */
      if (params->which_parameters & PLL_PARAMETER_FREE_RATES)
      {
        int n_cats = params->lk_params.partition->rate_cats;
        int i;
        double *xptr = &x[check_n];
        memcpy(xptr, partition->rates, sizeof(double) * n_cats);
        for (i=0; i<n_cats; i++)
        {
          set_range (0,
                     1,
                     5,
                     PLL_OPT_ALPHA_OFFSET,
                     xptr, l_ptr, u_ptr);
          *nbd_ptr = PLL_LBFGSB_BOUND_BOTH;
          xptr++;
          nbd_ptr++;
          l_ptr++;
          u_ptr++;
        }
        check_n += n_cats;
      }

      if (params->which_parameters & PLL_PARAMETER_RATE_WEIGHTS)
      {
        unsigned int n_rates = params->lk_params.partition->rate_cats;
              unsigned int n_weights_free_params = n_rates - 1;
              unsigned int cur_index;

              double * rate_weights =
                  params->lk_params.partition->rate_weights;

              params->highest_freq_state = 3;
              for (i = 1; i < n_rates; i++)
                      if (rate_weights[i] > rate_weights[params->highest_weight_state])
                        params->highest_freq_state = i;

              cur_index = 0;
              for (i = 0; i < n_rates; i++)
              {
                if (i != params->highest_freq_state)
                {
                  nbd_ptr[cur_index] = PLL_LBFGSB_BOUND_BOTH;
                  x[check_n + cur_index] = rate_weights[i]
                      / rate_weights[params->highest_freq_state];
                  set_range (PLL_OPT_MIN_FREQ,
                             PLL_OPT_DEFAULT_FREQ_RATIO,
                             PLL_OPT_MAX_FREQ,
                             PLL_OPT_FREQ_OFFSET,
                             &x[check_n + cur_index],
                             &l_ptr[cur_index], &u_ptr[cur_index]);
                  cur_index++;
                }
              }
              check_n += n_weights_free_params;
              nbd_ptr += n_weights_free_params;
              l_ptr += n_weights_free_params;
              u_ptr += n_weights_free_params;
//
//        int n_cats = params->lk_params.partition->rate_cats;
//        int i;
//        double *xptr = &x[check_n];
//        memcpy(xptr, partition->rate_weights, sizeof(double) * n_cats);
//        for (i=0; i<n_cats; i++)
//        {
//          set_range (1e-4,
//                     0.25,
//                     1,
//                     1e-4,
//                     xptr, l_ptr, u_ptr);
//          *nbd_ptr = PLL_LBFGSB_BOUND_BOTH;
//          xptr++;
//          nbd_ptr++;
//          l_ptr++;
//          u_ptr++;
//        }
//        check_n += n_cats;
      }

    /* topology (UNIMPLEMENTED) */
    if (params->which_parameters & PLL_PARAMETER_TOPOLOGY)
    {
      return PLL_FAILURE;
    }

    /* single branch length */
    if (params->which_parameters & PLL_PARAMETER_BRANCHES_SINGLE)
    {
        *nbd_ptr = PLL_LBFGSB_BOUND_LOWER;
        x[check_n] = params->lk_params.branch_lengths[0];
        set_range (PLL_OPT_MIN_BRANCH_LEN + PLL_LBFGSB_ERROR,
                   PLL_OPT_DEFAULT_BRANCH_LEN,
                   PLL_OPT_MAX_BRANCH_LEN,
                   PLL_OPT_BRANCH_LEN_OFFSET,
                   &x[check_n], l_ptr, u_ptr);
        check_n += 1;
        nbd_ptr++;
        l_ptr++;
        u_ptr++;
    }

    /* all branches */
    if (params->which_parameters & PLL_PARAMETER_BRANCHES_ALL)
    {
      unsigned int num_branch_lengths =
          params->lk_params.rooted ?
              (2 * partition->tips - 3) : (2 * partition->tips - 2);
      for (i = 0; i < num_branch_lengths; i++)
      {
        nbd_ptr[i] = PLL_LBFGSB_BOUND_LOWER;
        x[check_n + i] = params->lk_params.branch_lengths[i];
        set_range (PLL_OPT_MIN_BRANCH_LEN + PLL_LBFGSB_ERROR,
                   PLL_OPT_DEFAULT_BRANCH_LEN,
                   PLL_OPT_MAX_BRANCH_LEN,
                   PLL_OPT_BRANCH_LEN_OFFSET,
                   &x[check_n + i], &l_ptr[i], &u_ptr[i]);
      }
      check_n += num_branch_lengths;
      nbd_ptr += num_branch_lengths;
      l_ptr += num_branch_lengths;
      u_ptr += num_branch_lengths;
    }
    assert(check_n == num_variables);
  }

  score = pll_minimize_lbfgsb(x, lower_bounds, upper_bounds, bound_type,
                              num_variables, params->factr, params->pgtol,
                              params, compute_negative_lnl_unrooted);

  free (x);
  free (lower_bounds);
  free (upper_bounds);
  free (bound_type);

  if (is_nan(score))
  {
    score = -INFINITY;
    if (!pll_errno)
    {
      pll_errno = PLL_ERROR_LBFGSB_UNKNOWN;
      snprintf(pll_errmsg, 200, "Unknown LBFGSB error");
    }
  }

  return score;
} /* pll_optimize_parameters_lbfgsb */




/******************************************************************************/
/* GENERIC */
/******************************************************************************/
static void update_op_scalers(pll_partition_t * partition,
                              pll_utree_t * parent,
                              pll_utree_t * right_child,
                              pll_utree_t * left_child)
{
  int i;
  pll_operation_t op;

  /* set CLV */
  op.parent_clv_index    = parent->clv_index;
  op.parent_scaler_index = parent->scaler_index;
  op.child1_clv_index    = right_child->back->clv_index;
  op.child1_matrix_index = right_child->back->pmatrix_index;
  op.child1_scaler_index = right_child->back->scaler_index;
  op.child2_clv_index    = left_child->back->clv_index;
  op.child2_matrix_index = left_child->back->pmatrix_index;
  op.child2_scaler_index = left_child->back->scaler_index;
#if(UPDATE_SCALERS)
  /* update scalers */
  if (parent->scaler_index != PLL_SCALE_BUFFER_NONE)
  {
    int n = partition->sites;
    for (i=0; i<n; i++)
    {
      partition->scale_buffer[parent->scaler_index][i] =
          partition->scale_buffer[parent->scaler_index][i]
              + ((right_child->back->scaler_index != PLL_SCALE_BUFFER_NONE) ?
                  partition->scale_buffer[right_child->back->scaler_index][i] :
                  0)
              - ((parent->back->scaler_index != PLL_SCALE_BUFFER_NONE) ?
                  partition->scale_buffer[parent->back->scaler_index][i] :
                  0);
    }
  }
#endif
  pll_update_partials (partition, &op, 1);
}

/* if keep_update, P-matrices are updated after each branch length opt */
static double recomp_iterative (pll_newton_tree_params_t * params,
                                double *best_lnl,
                                int radius,
                                int keep_update)
{
  double lnl = 0.0,
     new_lnl = 0.0;
  pll_utree_t *tr_p, *tr_q, *tr_z;

  DBG("Optimizing branch %3d - %3d (%.6f) [%.4f]\n",
      tree->clv_index, tree->back->clv_index, tree->length, prev_lnl);

  lnl = *best_lnl;
  tr_p = params->tree;
  tr_q = params->tree->next;
  tr_z = tr_q?tr_q->next:NULL;

#ifndef NDEBUG
  {
    /* evaluate first */
    double test_logl = pll_compute_edge_loglikelihood (
        params->partition,
        tr_p->clv_index, tr_p->scaler_index,
        tr_p->back->clv_index, tr_p->back->scaler_index,
        tr_p->pmatrix_index, 0);
    if (fabs(test_logl - lnl) > 1e-6)
    {
      printf("ERROR: %s-%s %f vs %f\n", tr_p->label, tr_p->back->label, test_logl, lnl);
    }
    assert(fabs(test_logl - lnl) < 1e-6);
  }
#endif

  /* set Branch Length */
  assert(d_equals(tr_p->length, tr_p->back->length));

#if(BL_OPT_METHOD == PLL_BRANCH_OPT_LBFGSB)
  new_lnl = -1 * pll_optimize_parameters_lbfgsb(params);
#elif(BL_OPT_METHOD == PLL_BRANCH_OPT_BRENT)
  new_lnl = -1 * pll_optimize_parameters_brent(params);
#else
  double xmin, xguess, xmax;

  pll_update_sumtable (params->partition,
                       tr_p->clv_index,
                       tr_p->back->clv_index,
                       params->params_index,
                       params->freqs_index,
                       params->sumtable);

  xmin = PLL_OPT_MIN_BRANCH_LEN + PLL_LBFGSB_ERROR;
  xmax = PLL_OPT_MAX_BRANCH_LEN;
  xguess = tr_p->length;
  if (xguess < xmin || xguess > xmax)
    xguess = PLL_OPT_DEFAULT_BRANCH_LEN;

  double xres = pll_minimize_newton (xmin, xguess, xmax,
                                     10, &new_lnl, params,
                                     utree_derivative_func);

  if (pll_errno)
  {
    printf ("ERROR: %s\n", pll_errmsg);
    return PLL_FAILURE;
  }

  if (new_lnl >= 0)
    new_lnl = *best_lnl;
#endif

  /* ensure that new_lnl is not NaN */
  assert (new_lnl == new_lnl);

  if (new_lnl > lnl)
  {
    /* consolidate */
    DBG("Consolidate branch %.14f  to %.14f [%f]\n", tr_p->length, params->lk_params.branch_lengths[0], new_lnl);
    tr_p->length = xres;
    tr_p->back->length = tr_p->length;
    if (keep_update)
    {
      pll_update_prob_matrices(params->partition,
                               params->params_index,
                               &(params->tree->pmatrix_index),
                               &(tr_p->length),1);
#ifndef NDEBUG
    {
      lnl = pll_compute_edge_loglikelihood (
                params->partition,
                tr_p->clv_index, tr_p->scaler_index,
                tr_p->back->clv_index, tr_p->back->scaler_index,
                tr_p->pmatrix_index, params->freqs_index);
      assert(fabs(lnl - new_lnl) < 1e-6);
      assert(lnl >= *best_lnl);
    }
#endif
      *best_lnl = new_lnl;
    }
  }
  else
  {
    /* revert */
    if (keep_update)
    {
       pll_update_prob_matrices(params->partition,
                                 params->params_index,
                                 &tr_p->pmatrix_index,
                                 &tr_p->length, 1);
    }
#ifndef NDEBUG
    {
    double test_logl = pll_compute_edge_loglikelihood (
        params->partition,
        tr_p->clv_index, tr_p->scaler_index,
        tr_p->back->clv_index, tr_p->back->scaler_index,
        tr_p->pmatrix_index, 0);
    assert(fabs(test_logl - *best_lnl) < 1e-6);
    }
#endif
  }

  DBG(" Optimized branch %3d - %3d (%.6f) [%.4f]\n",
      tr_p->clv_index, tr_p->back->clv_index, tr_p->length, lnl);

  /* update children */
  if (radius && tr_q && tr_z)
  {
    /* update children 'Q'
     * CLV at P is recomputed with children P->back and Z->back
     * Scaler is updated by subtracting Q->back and adding P->back
     */

    update_op_scalers(params->partition,
                      tr_q,
                      tr_p,
                      tr_z);

    /* eval */
    pll_newton_tree_params_t params_cpy;
    memcpy(&params_cpy, params, sizeof(pll_newton_tree_params_t));
    params_cpy.tree = tr_q->back;
    lnl = recomp_iterative (&params_cpy, best_lnl, radius-1, keep_update);

    /* update children 'Z'
     * CLV at P is recomputed with children P->back and Q->back
     * Scaler is updated by subtracting Z->back and adding Q->back
     */

    update_op_scalers(params->partition,
                      tr_z,
                      tr_q,
                      tr_p);

   /* eval */
    params_cpy.tree = tr_z->back;
    lnl = recomp_iterative (&params_cpy, best_lnl, radius-1, keep_update);

    /* reset to initial state
     * CLV at P is recomputed with children Q->back and Z->back
     * Scaler is updated by subtracting P->back and adding Z->back
     */

    update_op_scalers(params->partition,
                      tr_p,
                      tr_z,
                      tr_q);
  }

  return lnl;
} /* recomp_iterative */

PLL_EXPORT double pll_optimize_branch_lengths_local (
                                              pll_partition_t * partition,
                                              pll_utree_t * tree,
                                              unsigned int params_index,
                                              unsigned int freqs_index,
                                              double tolerance,
                                              int smoothings,
                                              int radius,
                                              int keep_update)
{
  int i;
  double lnl = 0.0;

  pll_newton_tree_params_t params;
  params.partition = partition;
  params.tree = tree;
  params.params_index = params_index;
  params.freqs_index = freqs_index;
  params.sumtable = 0;
      if ((params.sumtable = (double *) calloc (
          partition->sites
              * partition->rate_cats
              * partition->states,
          sizeof(double))) == NULL)
        return PLL_FAILURE;

  lnl = pll_compute_edge_loglikelihood (partition,
                                        tree->back->clv_index,
                                        tree->back->scaler_index,
                                        tree->clv_index,
                                        tree->scaler_index,
                                        tree->pmatrix_index,
                                        freqs_index);

  for (i = 0; i < smoothings; i++)
  {
    double new_lnl = lnl;
    recomp_iterative (&params, &new_lnl, radius, keep_update);
    params.tree = params.tree->back;
    recomp_iterative (&params, &new_lnl, radius-1, keep_update);
    assert(new_lnl >= lnl);
    if (fabs (new_lnl - lnl) < tolerance)
      i = smoothings;
    lnl = new_lnl;
  }

  free(params.sumtable);

  return -1*lnl;
} /* pll_optimize_branch_lengths_iterative */

PLL_EXPORT double pll_optimize_branch_lengths_iterative (
                                              pll_partition_t * partition,
                                              pll_utree_t * tree,
                                              unsigned int params_index,
                                              unsigned int freqs_index,
                                              double tolerance,
                                              int smoothings,
                                              int keep_update)
{
  double lnl;
  lnl = pll_optimize_branch_lengths_local (partition,
                                           tree,
                                           params_index,
                                           freqs_index,
                                           tolerance,
                                           smoothings,
                                           -1,
                                           keep_update);
  return lnl;
} /* pll_optimize_branch_lengths_iterative */

PLL_EXPORT double pll_derivative_func(void * parameters,
                                      double proposal,
                                      double *df, double *ddf)
{
  pll_optimize_options_t * params = (pll_optimize_options_t *) parameters;
  double score = pll_compute_likelihood_derivatives (
      params->lk_params.partition,
      params->lk_params.where.unrooted_t.parent_scaler_index,
      params->lk_params.where.unrooted_t.child_scaler_index,
      proposal,
      params->params_index,
      params->lk_params.freqs_index,
      params->sumtable,
      df, ddf);
  return score;
}
