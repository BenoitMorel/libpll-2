
class TemplateAVX {
  public:
   typedef __m256d reg;
   static const int vecsize = 4;

};

// <4, typename VEC::reg
template <int STATES, class VEC>
void pll_core_template_update_partial_ii(unsigned int sites,
                                               unsigned int rate_cats,
                                               double * parent_clv,
                                               unsigned int * parent_scaler,
                                               const double * left_clv,
                                               const double * right_clv,
                                               const double * left_matrix,
                                               const double * right_matrix,
                                               const unsigned int * left_scaler,
                                               const unsigned int * right_scaler,
                                               unsigned int attrib)
{
  unsigned int i,j,k,n;

  const double * lmat;
  const double * rmat;

  const unsigned int STATES_PADDED = (STATES+3) & 0xFFFFFFFC;
  unsigned int span_padded = STATES_PADDED * rate_cats;

  /* scaling-related stuff */
  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
  unsigned int scale_mask;
  unsigned int init_mask;
  typename VEC::reg v_scale_threshold = _mm256_set1_pd(PLL_SCALE_THRESHOLD);
  typename VEC::reg v_scale_factor = _mm256_set1_pd(PLL_SCALE_FACTOR);

  if (!parent_scaler)
  {
    /* scaling disabled / not required */
    scale_mode = init_mask = 0;
  }
  else
  {
    /* determine the scaling mode and init the vars accordingly */
    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
    init_mask = (scale_mode == 1) ? 0xF : 0;
    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
    /* add up the scale vector of the two children if available */
    pll_fill_parent_scaler(scaler_size, parent_scaler, left_scaler, right_scaler);
  }

  size_t displacement = (STATES_PADDED - STATES) * (STATES_PADDED);

  /* compute CLV */
  for (n = 0; n < sites; ++n)
  {
    lmat = left_matrix;
    rmat = right_matrix;
    scale_mask = init_mask;

    for (k = 0; k < rate_cats; ++k)
    {
      unsigned int rate_mask = 0xF;

      /* iterate over quadruples of rows */
      for (i = 0; i < STATES_PADDED; i += 4)
      {
        typename VEC::reg v_terma0 = _mm256_setzero_pd();
        typename VEC::reg v_termb0 = _mm256_setzero_pd();
        typename VEC::reg v_terma1 = _mm256_setzero_pd();
        typename VEC::reg v_termb1 = _mm256_setzero_pd();
        typename VEC::reg v_terma2 = _mm256_setzero_pd();
        typename VEC::reg v_termb2 = _mm256_setzero_pd();
        typename VEC::reg v_terma3 = _mm256_setzero_pd();
        typename VEC::reg v_termb3 = _mm256_setzero_pd();

        typename VEC::reg v_mat;
        typename VEC::reg v_lclv;
        typename VEC::reg v_rclv;

        /* point to the four rows of the left matrix */
        const double * lm0 = lmat;
        const double * lm1 = lm0 + STATES_PADDED;
        const double * lm2 = lm1 + STATES_PADDED;
        const double * lm3 = lm2 + STATES_PADDED;

        /* point to the four rows of the right matrix */
        const double * rm0 = rmat;
        const double * rm1 = rm0 + STATES_PADDED;
        const double * rm2 = rm1 + STATES_PADDED;
        const double * rm3 = rm2 + STATES_PADDED;

        /* iterate over quadruples of columns */
        for (j = 0; j < STATES_PADDED; j += 4)
        {
          v_lclv    = _mm256_load_pd(left_clv+j);
          v_rclv    = _mm256_load_pd(right_clv+j);

          /* row 0 */
          v_mat    = _mm256_load_pd(lm0);
          v_terma0 = _mm256_add_pd(v_terma0,
                                   _mm256_mul_pd(v_mat,v_lclv));
          v_mat    = _mm256_load_pd(rm0);
          v_termb0 = _mm256_add_pd(v_termb0,
                                   _mm256_mul_pd(v_mat,v_rclv));
          lm0 += 4;
          rm0 += 4;

          /* row 1 */
          v_mat    = _mm256_load_pd(lm1);
          v_terma1 = _mm256_add_pd(v_terma1,
                                   _mm256_mul_pd(v_mat,v_lclv));
          v_mat    = _mm256_load_pd(rm1);
          v_termb1 = _mm256_add_pd(v_termb1,
                                   _mm256_mul_pd(v_mat,v_rclv));
          lm1 += 4;
          rm1 += 4;

          /* row 2 */
          v_mat    = _mm256_load_pd(lm2);
          v_terma2 = _mm256_add_pd(v_terma2,
                                   _mm256_mul_pd(v_mat,v_lclv));
          v_mat    = _mm256_load_pd(rm2);
          v_termb2 = _mm256_add_pd(v_termb2,
                                   _mm256_mul_pd(v_mat,v_rclv));
          lm2 += 4;
          rm2 += 4;

          /* row 3 */
          v_mat    = _mm256_load_pd(lm3);
          v_terma3 = _mm256_add_pd(v_terma3,
                                   _mm256_mul_pd(v_mat,v_lclv));
          v_mat    = _mm256_load_pd(rm3);
          v_termb3 = _mm256_add_pd(v_termb3,
                                   _mm256_mul_pd(v_mat,v_rclv));
          lm3 += 4;
          rm3 += 4;
        }

        /* point pmatrix to the next four rows */ 
        lmat = lm3;
        rmat = rm3;

        typename VEC::reg xmm0 = _mm256_unpackhi_pd(v_terma0,v_terma1);
        typename VEC::reg xmm1 = _mm256_unpacklo_pd(v_terma0,v_terma1);

        typename VEC::reg xmm2 = _mm256_unpackhi_pd(v_terma2,v_terma3);
        typename VEC::reg xmm3 = _mm256_unpacklo_pd(v_terma2,v_terma3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        typename VEC::reg v_terma_sum = _mm256_add_pd(xmm2,xmm3);

        /* compute termb */

        xmm0 = _mm256_unpackhi_pd(v_termb0,v_termb1);
        xmm1 = _mm256_unpacklo_pd(v_termb0,v_termb1);

        xmm2 = _mm256_unpackhi_pd(v_termb2,v_termb3);
        xmm3 = _mm256_unpacklo_pd(v_termb2,v_termb3);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        typename VEC::reg v_termb_sum = _mm256_add_pd(xmm2,xmm3);

        typename VEC::reg v_prod = _mm256_mul_pd(v_terma_sum,v_termb_sum);

        /* check if scaling is needed for the current rate category */
        typename VEC::reg v_cmp = _mm256_cmp_pd(v_prod, v_scale_threshold, _CMP_LT_OS);
        rate_mask = rate_mask & _mm256_movemask_pd(v_cmp);

        _mm256_store_pd(parent_clv+i, v_prod);
      }

      if (scale_mode == 2)
      {
        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
         * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
        if (rate_mask == 0xF)
        {
          for (i = 0; i < STATES_PADDED; i += 4)
          {
            typename VEC::reg v_prod = _mm256_load_pd(parent_clv + i);
            v_prod = _mm256_mul_pd(v_prod, v_scale_factor);
            _mm256_store_pd(parent_clv + i, v_prod);
          }
          parent_scaler[n*rate_cats + k] += 1;
        }
      }
      else
        scale_mask = scale_mask & rate_mask;

      /* reset pointers to point to the start of the next p-matrix, as the
         vectorization assumes a square STATES_PADDED * STATES_PADDED matrix,
         even though the real matrix is STATES * STATES_PADDED */
      lmat -= displacement;
      rmat -= displacement;

      parent_clv += STATES_PADDED;
      left_clv   += STATES_PADDED;
      right_clv  += STATES_PADDED;
    }

    /* if *all* entries of the site CLV were below the threshold then scale
       (all) entries by PLL_SCALE_FACTOR */
    if (scale_mask == 0xF)
    {
      parent_clv -= span_padded;
      for (i = 0; i < span_padded; i += 4)
      {
        typename VEC::reg v_prod = _mm256_load_pd(parent_clv + i);
        v_prod = _mm256_mul_pd(v_prod,v_scale_factor);
        _mm256_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      parent_scaler[n] += 1;
    }
  }
}

