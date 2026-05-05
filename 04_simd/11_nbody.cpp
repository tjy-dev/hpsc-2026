#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  for(int i=0; i<N; i++) {
    __m512 xi = _mm512_set1_ps(x[i]);
    __m512 yi = _mm512_set1_ps(y[i]);
    __mmask16 mask = 0xffff ^ (1 << i);
    __m512 zero = _mm512_setzero_ps();

    __m512  rx = _mm512_sub_ps(xi, xvec);
    __m512  ry = _mm512_sub_ps(yi, yvec);
    __m512 rx2 = _mm512_mul_ps(rx, rx);
    __m512 ry2 = _mm512_mul_ps(ry, ry);
    __m512  r2 = _mm512_add_ps(rx2, ry2);
    __m512   r = _mm512_sqrt_ps(r2);
    __m512  r3 = _mm512_mul_ps(r2, r);
    __m512 one = _mm512_set1_ps(1.0f);
    __m512 r3_safe = _mm512_mask_blend_ps(mask, one, r3);

    __m512 numerator_x = _mm512_mul_ps(rx, mvec);
    __m512 contrib_x   = _mm512_div_ps(numerator_x, r3_safe);
    contrib_x = _mm512_mask_blend_ps(mask, zero, contrib_x);
    fx[i] -= _mm512_reduce_add_ps(contrib_x);

    __m512 numerator_y = _mm512_mul_ps(ry, mvec);
    __m512 contrib_y   = _mm512_div_ps(numerator_y, r3_safe);
    contrib_y = _mm512_mask_blend_ps(mask, zero, contrib_y);
    fy[i] -= _mm512_reduce_add_ps(contrib_y);
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
