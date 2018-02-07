
class TemplateAVX {
public:
  typedef __m256d reg;
  static const int vecsize = 4;
  static constexpr auto load = _mm256_load_pd;
  static constexpr auto mult = _mm256_mul_pd;
  static constexpr auto add = _mm256_add_pd; 
  static constexpr auto set = _mm256_set1_pd; 
};

class TemplateSSE {
  public:
   typedef __m128d reg;
   static const int vecsize = 2;
  static constexpr auto load = _mm_load_pd;
  static constexpr auto mult = _mm_mul_pd;
  static constexpr auto add = _mm_add_pd; 
  static constexpr auto set = _mm_set1_pd; 

};

template <class VEC, unsigned int p>
class FOR {
public:
  static void plop(typename VEC::reg &v_mat, 
      typename VEC::reg v_terma[],
      typename VEC::reg v_termb[],
      const double * lm[],
      const double * rm[],
      typename VEC::reg &v_lclv,
      typename VEC::reg &v_rclv
      ) 
  {
    const int index = 4-p;
    v_mat    = VEC::load(lm[index]);
    v_terma[index] = VEC::add(v_terma[index], VEC::mult(v_mat,v_lclv));
    v_mat    = VEC::load(rm[index]);
    v_termb[index] = VEC::add(v_termb[index], VEC::mult(v_mat,v_rclv));
    lm[index] += VEC::vecsize;
    rm[index] += VEC::vecsize;
    FOR<VEC, p - 1>::plop(v_mat, v_terma, v_termb, lm, rm, v_lclv, v_rclv);
  }
};

template <class VEC>
class FOR<VEC, 0> {
public:
  static void plop(typename VEC::reg &v_mat, 
      typename VEC::reg v_terma[],
      typename VEC::reg v_termb[],
      const double * lm[],
      const double * rm[],
      typename  VEC::reg &v_lclv,
      typename VEC::reg &v_rclv
      )
  {
  }
};

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

  /* compute CLV */
  for (unsigned int n = 0; n < sites; ++n)
  {
    const double *lmat = left_matrix;
    const double *rmat = right_matrix;
    unsigned int scale_mask = init_mask;

    for (unsigned int k = 0; k < rate_cats; ++k)
    {
      unsigned int rate_mask = 0xF;

      /* iterate over quadruples of rows */
      for (unsigned int i = 0; i < STATES_PADDED; i += VEC::vecsize)
      {
        typename VEC::reg v_terma[VEC::vecsize];
        typename VEC::reg v_termb[VEC::vecsize];
        const double * lm[VEC::vecsize];
        const double * rm[VEC::vecsize];
        /* For some reason, the compiler optimizes better with 3 seperate loops */
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          v_terma[j] = _mm256_setzero_pd();
          v_termb[j] = _mm256_setzero_pd();
        }
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          lm[j]= lmat + j * STATES_PADDED;
        }
        for (unsigned int j = 0; j < VEC::vecsize; ++j) {
          rm[j]= rmat + j * STATES_PADDED;
        } 

        /* iterate over sets of columns */
        for (unsigned int j = 0; j < STATES_PADDED; j += VEC::vecsize)
        {
          typename VEC::reg v_mat;
          typename VEC::reg v_lclv    = VEC::load(left_clv + j);
          typename VEC::reg v_rclv    = VEC::load(right_clv + j);
          FOR<VEC, VEC::vecsize>::plop(v_mat, v_terma, v_termb, lm, rm, v_lclv, v_rclv);
        }

        /* point pmatrix to the next four rows */ 
        lmat = lm[3];
        rmat = rm[3];

        typename VEC::reg xmm0 = _mm256_unpackhi_pd(v_terma[0],v_terma[1]);
        typename VEC::reg xmm1 = _mm256_unpacklo_pd(v_terma[0],v_terma[1]);

        typename VEC::reg xmm2 = _mm256_unpackhi_pd(v_terma[2],v_terma[3]);
        typename VEC::reg xmm3 = _mm256_unpacklo_pd(v_terma[2],v_terma[3]);

        xmm0 = _mm256_add_pd(xmm0,xmm1);
        xmm1 = _mm256_add_pd(xmm2,xmm3);

        xmm2 = _mm256_permute2f128_pd(xmm0,xmm1, _MM_SHUFFLE(0,2,0,1));

        xmm3 = _mm256_blend_pd(xmm0,xmm1,12);

        typename VEC::reg v_terma_sum = _mm256_add_pd(xmm2,xmm3);

        /* compute term[b] */

        xmm0 = _mm256_unpackhi_pd(v_termb[0],v_termb[1]);
        xmm1 = _mm256_unpacklo_pd(v_termb[0],v_termb[1]);

        xmm2 = _mm256_unpackhi_pd(v_termb[2],v_termb[3]);
        xmm3 = _mm256_unpacklo_pd(v_termb[2],v_termb[3]);

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
          for (unsigned int i = 0; i < STATES_PADDED; i += 4)
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
      for (unsigned int i = 0; i < span_padded; i += 4)
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


template <int STATES, class VEC>
PLL_EXPORT void pll_core_template_update_partial_ii_sse(
                                               unsigned int sites,
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
  unsigned int i,j,k,n;

  const double * lmat;
  const double * rmat;

  __m128d xmm0,xmm1,xmm2,xmm3,xmm4,xmm5,xmm6;

  for (n = 0; n < sites; ++n)
  {
    lmat = left_matrix;
    rmat = right_matrix;
    unsigned int scale_mask = init_mask;

    for (k = 0; k < rate_cats; ++k)
    {
      unsigned int rate_mask = 0x3;

      for (i = 0; i < STATES_PADDED; i += 2)
      {
        __m128d v_terma0;
        __m128d v_terma1;
        __m128d v_termb0;
        __m128d v_termb1;

        if (STATES != 4) {
          v_terma0 = _mm_setzero_pd();
          v_terma1 = _mm_setzero_pd();
          v_termb0 = _mm_setzero_pd();
          v_termb1 = _mm_setzero_pd();
        }
        const double * lm0 = lmat;
        const double * lm1 = lm0 + STATES_PADDED;

        const double * rm0 = rmat;
        const double * rm1 = rm0 + STATES_PADDED;
        
      


        for (j = 0; j < STATES_PADDED; j += 4)
        {
          /* load left and right clvs */
          //if (STATES != 4 || i == 0) {
            xmm0 = _mm_load_pd(left_clv+j);
            xmm1 = _mm_load_pd(right_clv+j); 
          //}

          xmm2 = _mm_load_pd(lm0);
          lm0 += 2;
          if (j == 0 && STATES == 4) {
            v_terma0 = _mm_mul_pd(xmm2,xmm0);
          } else {
            v_terma0 = _mm_add_pd(v_terma0, _mm_mul_pd(xmm2,xmm0));
          }
          
          xmm2 = _mm_load_pd(lm1);
          lm1 += 2;
          if (j == 0 && STATES == 4) {
            v_terma1 = _mm_mul_pd(xmm2,xmm0);
          } else {
            v_terma1 = _mm_add_pd(v_terma1, _mm_mul_pd(xmm2,xmm0));
          }


          xmm2 = _mm_load_pd(rm0);
          rm0 += 2;
          if (j == 0 && STATES == 4) {
            v_termb0 = _mm_mul_pd(xmm2,xmm1);
          } else {
            v_termb0 = _mm_add_pd(v_termb0,_mm_mul_pd(xmm2,xmm1));
          }


          xmm2 = _mm_load_pd(rm1);
          rm1 += 2;
          if (j == 0 && STATES == 4) {
            v_termb1 = _mm_mul_pd(xmm2,xmm1);
          } else {
            v_termb1 = _mm_add_pd(v_termb1,_mm_mul_pd(xmm2,xmm1));
          }
        
          
          //if (STATES != 4 || i == 0) {
            xmm3 = _mm_load_pd(left_clv+j + 2);
            xmm4 = _mm_load_pd(right_clv+j + 2); 
          //}

          xmm2 = _mm_load_pd(lm0);
          lm0 += 2;
            v_terma0 = _mm_add_pd(v_terma0, _mm_mul_pd(xmm2,xmm3));
          
          xmm2 = _mm_load_pd(lm1);
          lm1 += 2;
            v_terma1 = _mm_add_pd(v_terma1, _mm_mul_pd(xmm2,xmm3));


          xmm2 = _mm_load_pd(rm0);
          rm0 += 2;
            v_termb0 = _mm_add_pd(v_termb0,_mm_mul_pd(xmm2,xmm4));


          xmm2 = _mm_load_pd(rm1);
          rm1 += 2;
            v_termb1 = _mm_add_pd(v_termb1,_mm_mul_pd(xmm2,xmm4));
        
        
        
        
        }
        
        lmat = lm1;
        rmat = rm1;

        xmm4 = _mm_hadd_pd(v_terma0,v_terma1);
        xmm5 = _mm_hadd_pd(v_termb0,v_termb1);
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
          for (i = 0; i < STATES_PADDED; i += 2)
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
      for (i = 0; i < span_padded; i += 2)
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


