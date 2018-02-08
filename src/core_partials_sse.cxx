template <class VEC, unsigned int STATES, unsigned int p, bool OPT> 
void sse_inner_3(
      typename VEC::reg &xmm_mat,
      typename VEC::reg &xmm_clv,
      typename VEC::reg v_term[],
      const double *m[],
      unsigned int j)
{
    xmm_mat = VEC::load(m[p]);
    m[p] += 2;
    if (OPT) {
      if (j == 0 && STATES == 4) {
        v_term[p] = VEC::mult(xmm_mat,xmm_clv);
      } else {
        v_term[p] = VEC::add(v_term[p], VEC::mult(xmm_mat,xmm_clv));
      }
    } else {
        v_term[p] = VEC::add(v_term[p], VEC::mult(xmm_mat,xmm_clv));
    }
}

template <int STATES, class VEC>
PLL_EXPORT void pll_core_template_update_partial_ii_sse(unsigned int sites,
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
  const unsigned int STATES_PADDED = (STATES+(VEC::vecsize - 1)) & (0xFFFFFFFF - (VEC::vecsize - 1));
  const unsigned int span_padded = STATES_PADDED * rate_cats;

  /* scaling-related stuff */
  unsigned int scale_mode;  /* 0 = none, 1 = per-site, 2 = per-rate */
  unsigned int init_mask;
  if (!parent_scaler) {
    scale_mode = 0;
    init_mask = 0;
  } else {
    scale_mode = (attrib & PLL_ATTRIB_RATE_SCALERS) ? 2 : 1;
    init_mask = (scale_mode == 1) ? 0xF : 0;
    const size_t scaler_size = (scale_mode == 2) ? sites * rate_cats : sites;
    pll_fill_parent_scaler(scaler_size, parent_scaler, left_scaler, right_scaler);
  }
  
  typename VEC::reg v_scale_threshold = VEC::set(PLL_SCALE_THRESHOLD);
  typename VEC::reg v_scale_factor = VEC::set(PLL_SCALE_FACTOR);
  size_t displacement = (STATES_PADDED - STATES) * (STATES_PADDED);


  typename VEC::reg xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;

  for (unsigned int n = 0; n < sites; ++n)
  {
    const double * lmat = left_matrix;
    const double * rmat = right_matrix;
    unsigned int scale_mask = init_mask;

    for (unsigned int k = 0; k < rate_cats; ++k)
    {
      unsigned int rate_mask = 0x3;

      for (unsigned int i = 0; i < STATES_PADDED; i += VEC::vecsize)
      {
        typename VEC::reg v_terma[VEC::vecsize];
        typename VEC::reg v_termb[VEC::vecsize];
        typename VEC::reg v_lclv;
        typename VEC::reg v_rclv;
        
        const double * lm[VEC::vecsize];
        const double * rm[VEC::vecsize];
        /* For some reason, the compiler optimizes better with 3 seperate loops */
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          v_terma[j] = VEC::setzero();
          v_termb[j] = VEC::setzero();
        }
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          lm[j]= lmat + j * STATES_PADDED;
        }
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          rm[j]= rmat + j * STATES_PADDED;
        }

        FOR<VEC, true, true, VEC::vecsize>::inner_3(v_terma, lm, v_lclv, left_clv);
        FOR<VEC, true, true, VEC::vecsize>::inner_3(v_termb, rm, v_rclv, right_clv);
        FOR<VEC, false, true, VEC::vecsize>::inner_3(v_terma, lm, v_lclv, left_clv + 2);
        FOR<VEC, false, true, VEC::vecsize>::inner_3(v_termb, rm, v_rclv, right_clv + 2);

        for (unsigned int j = VEC::vecsize * 2; j < STATES_PADDED; j += VEC::vecsize * 2)
        {
          FOR<VEC, false, true, VEC::vecsize>::inner_3(v_terma, lm, v_lclv, left_clv + j);
          FOR<VEC, false, true, VEC::vecsize>::inner_3(v_termb, rm, v_rclv, right_clv + j);
          FOR<VEC, false, true, VEC::vecsize>::inner_3(v_terma, lm, v_lclv, left_clv + j + 2);
          FOR<VEC, false, true, VEC::vecsize>::inner_3(v_termb, rm, v_rclv, right_clv + j + 2);
        }
        
        lmat += 2 * STATES_PADDED;
        rmat += 2 * STATES_PADDED;
        
        xmm4 = _mm_hadd_pd(v_termb[0],v_termb[1]);
        xmm5 = _mm_hadd_pd(v_terma[0],v_terma[1]);
        xmm6 = _mm_mul_pd(xmm4,xmm5);

        /* check if scaling is needed for the current rate category */
        __m128d v_cmp = _mm_cmplt_pd(xmm6, v_scale_threshold);
        rate_mask = rate_mask & _mm_movemask_pd(v_cmp);

        _mm_store_pd(parent_clv+i,xmm6);
      }

      /* reset pointers to the start of the next p-matrix, as the vectorization
         assumes a square STATES_PADDED * STATES_PADDED matrix, even though the
         real matrix is STATES * STATES_PADDED */
      lmat -= displacement;
      rmat -= displacement;

      if (scale_mode == 2)
      {
        /* PER-RATE SCALING: if *all* entries of the *rate* CLV were below
         * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
        if (rate_mask == 0x3)
        {
          for (unsigned int i = 0; i < STATES_PADDED; i += 2)
          {
            __m128d v_prod = _mm_load_pd(parent_clv + i);
            v_prod = _mm_mul_pd(v_prod, v_scale_factor);
            _mm_store_pd(parent_clv + i, v_prod);
          }
          parent_scaler[n*rate_cats + k] += 1;
        }
      }
      else
        scale_mask = scale_mask & rate_mask;

      parent_clv += STATES_PADDED;
      left_clv   += STATES_PADDED;
      right_clv  += STATES_PADDED;
    }

    /* PER-SITE SCALING: if *all* entries of the *site* CLV were below
     * the threshold then scale (all) entries by PLL_SCALE_FACTOR */
    if (scale_mask == 0x3)
    {
      parent_clv -= span_padded;
      for (unsigned int i = 0; i < span_padded; i += 2)
      {
        __m128d v_prod = _mm_load_pd(parent_clv + i);
        v_prod = _mm_mul_pd(v_prod,v_scale_factor);
        _mm_store_pd(parent_clv + i, v_prod);
      }
      parent_clv += span_padded;
      parent_scaler[n] += 1;
    }
  }
}


