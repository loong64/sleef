// Helper-level lane-layout tests for the LoongArch LSX/LASX backends.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CONFIG 1

#if defined(TEST_LSX)
#include "helperlsx.h"
#elif defined(TEST_LASX)
#include "helperlasx.h"
#else
#error Select TEST_LSX or TEST_LASX
#endif

static int check_int32(const char *name, const int32_t *actual,
                       const int32_t *expected, int count) {
  for (int i = 0; i < count; i++) {
    if (actual[i] != expected[i]) {
      fprintf(stderr, "%s: lane %d: got 0x%08x, expected 0x%08x\n",
              name, i, (uint32_t)actual[i], (uint32_t)expected[i]);
      return 1;
    }
  }
  return 0;
}

static int check_double(const char *name, const double *actual,
                        const double *expected, int count) {
  for (int i = 0; i < count; i++) {
    if (actual[i] != expected[i]) {
      fprintf(stderr, "%s: lane %d: got %.17g, expected %.17g\n",
              name, i, actual[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

static int test_mask_conversion_layout(void) {
#if defined(TEST_LSX)
  /* vo64/vo32 are stored in the raw vopmask word layout. */
  const uint32_t vo32_source_words[VECTLENSP] = { UINT32_MAX, 0, 0, 0 };
  const uint32_t vo64_source_words[VECTLENSP] = { UINT32_MAX, UINT32_MAX, 0, 0 };
  const uint32_t vo64_expected_words[VECTLENSP] = { UINT32_MAX, UINT32_MAX, 0, 0 };
  const uint32_t vo32_expected_words[VECTLENDP] = { UINT32_MAX, 0 };
#else
  const uint32_t vo32_source_words[VECTLENSP] = {
    UINT32_MAX, 0, UINT32_MAX, 0, 0, 0, 0, 0
  };
  const uint32_t vo64_source_words[VECTLENSP] = {
    UINT32_MAX, UINT32_MAX, 0, 0, UINT32_MAX, UINT32_MAX, 0, 0
  };
  const uint32_t vo64_expected_words[VECTLENSP] = {
    UINT32_MAX, UINT32_MAX, 0, 0, UINT32_MAX, UINT32_MAX, 0, 0
  };
  const uint32_t vo32_expected_words[VECTLENDP] = { UINT32_MAX, 0, UINT32_MAX, 0 };
#endif
  uint32_t actual32[VECTLENSP];
  vopmask mask;

  memcpy(&mask, vo32_source_words, sizeof(mask));
  mask = vcast_vo64_vo32(mask);
  memcpy(actual32, &mask, sizeof(actual32));
  if (check_int32("vcast_vo64_vo32", (const int32_t *)actual32,
                  (const int32_t *)vo64_expected_words, VECTLENSP)) return 1;

  memcpy(&mask, vo64_source_words, sizeof(mask));
  mask = vcast_vo32_vo64(mask);
  memcpy(actual32, &mask, sizeof(actual32));
  return check_int32("vcast_vo32_vo64", (const int32_t *)actual32,
                     (const int32_t *)vo32_expected_words, VECTLENDP);
}

static int check_float(const char *name, const float *actual,
                       const float *expected, int count) {
  for (int i = 0; i < count; i++) {
    if (actual[i] != expected[i]) {
      fprintf(stderr, "%s: lane %d: got %.9g, expected %.9g\n",
              name, i, actual[i], expected[i]);
      return 1;
    }
  }
  return 0;
}

static int test_mask_select_layout(void) {
  uint64_t mask_bits[VECTLENDP];
#if defined(TEST_LSX)
  const double x_values[VECTLENDP] = { 10.0, 20.0 };
  const double y_values[VECTLENDP] = { 1.0, 2.0 };
  const double expected[VECTLENDP] = { 1.0, 20.0 };
#else
  const double x_values[VECTLENDP] = {
    10.0, 20.0, 30.0, 40.0
  };
  const double y_values[VECTLENDP] = {
    1.0, 2.0, 3.0, 4.0
  };
  const double expected[VECTLENDP] = {
    1.0, 20.0, 30.0, 4.0
  };
#endif
  double actual[VECTLENDP];
  vopmask mask;
  vdouble x, y, result;

  for (int i = 0; i < VECTLENDP; i++) mask_bits[i] = (i == 1 || i == 2) ? UINT64_MAX : 0;
  memcpy(&mask, mask_bits, sizeof(mask));
  memcpy(&x, x_values, sizeof(x));
  memcpy(&y, y_values, sizeof(y));
  result = vsel_vd_vo_vd_vd(mask, x, y);
  memcpy(actual, &result, sizeof(actual));
  return check_double("vsel_vd_vo_vd_vd", actual, expected, VECTLENDP);
}

static int test_integer_layout(void) {
#if defined(TEST_LSX)
  int32_t input[VECTLENDP] = { 0x11, 0x22 };
  int32_t expected[VECTLENSP] = { 0, 0x11, 0, 0x22 };
#else
  int32_t input[VECTLENDP] = { 0x11, 0x22, 0x33, 0x44 };
  int32_t expected[VECTLENSP] = { 0, 0x11, 0, 0x22, 0, 0x33, 0, 0x44 };
#endif
  int32_t actual[VECTLENSP];
  vint vi = { 0 };
  memcpy(&vi, input, sizeof(input));

  vint2 vi2 = vcastu_vi2_vi(vi);
  memcpy(actual, &vi2, sizeof(actual));
  if (check_int32("vcastu_vi2_vi", actual, expected, VECTLENSP)) return 1;

  vint roundtrip = vcastu_vi_vi2(vi2);
  memcpy(actual, &roundtrip, sizeof(input));
  if (check_int32("vcastu_vi_vi2", actual, input, VECTLENDP)) return 1;

  vint2 first_half = vreinterpretFirstHalf_vi2_vi(vi);
  memset(expected, 0, sizeof(expected));
  memcpy(expected, input, sizeof(input));
  memcpy(actual, &first_half, sizeof(actual));
  if (check_int32("vreinterpretFirstHalf_vi2_vi", actual, expected, VECTLENSP)) return 1;

  vint first = vreinterpretFirstHalf_vi_vi2(first_half);
  memcpy(actual, &first, sizeof(input));
  return check_int32("vreinterpretFirstHalf_vi_vi2", actual, input, VECTLENDP);
}

static int test_numeric_layout(void) {
#if defined(TEST_LSX)
  const double doubles[VECTLENDP] = { 1.0, 2.0 };
  const double expected_rev21_doubles[VECTLENDP] = { 2.0, 1.0 };
  const double expected_reva2_doubles[VECTLENDP] = { 1.0, 2.0 };
  const float floats[VECTLENSP] = { 1.0f, 2.0f, 3.0f, 4.0f };
  const float expected_rev21_floats[VECTLENSP] = { 2.0f, 1.0f, 4.0f, 3.0f };
  const float expected_reva2_floats[VECTLENSP] = { 3.0f, 4.0f, 1.0f, 2.0f };
#else
  const double doubles[VECTLENDP] = { 1.0, 2.0, 3.0, 4.0 };
  const double expected_rev21_doubles[VECTLENDP] = { 2.0, 1.0, 4.0, 3.0 };
  const double expected_reva2_doubles[VECTLENDP] = { 3.0, 4.0, 1.0, 2.0 };
  const float floats[VECTLENSP] = { 1.0f, 2.0f, 3.0f, 4.0f,
                                    5.0f, 6.0f, 7.0f, 8.0f };
  const float expected_rev21_floats[VECTLENSP] = { 2.0f, 1.0f, 4.0f, 3.0f,
                                                   6.0f, 5.0f, 8.0f, 7.0f };
  const float expected_reva2_floats[VECTLENSP] = { 7.0f, 8.0f, 5.0f, 6.0f,
                                                   3.0f, 4.0f, 1.0f, 2.0f };
#endif
  double actual_doubles[VECTLENDP];
  float actual_floats[VECTLENSP];
  vdouble vd;
  vfloat vf;
  vdouble result_doubles;
  vfloat result_floats;

  memcpy(&vd, doubles, sizeof(doubles));
  result_doubles = vrev21_vd_vd(vd);
  memcpy(actual_doubles, &result_doubles, sizeof(actual_doubles));
  if (check_double("vrev21_vd_vd", actual_doubles, expected_rev21_doubles, VECTLENDP)) return 1;

  result_doubles = vreva2_vd_vd(vd);
  memcpy(actual_doubles, &result_doubles, sizeof(actual_doubles));
  if (check_double("vreva2_vd_vd", actual_doubles, expected_reva2_doubles, VECTLENDP)) return 1;

  memcpy(&vf, floats, sizeof(floats));
  result_floats = vrev21_vf_vf(vf);
  memcpy(actual_floats, &result_floats, sizeof(actual_floats));
  if (check_float("vrev21_vf_vf", actual_floats, expected_rev21_floats, VECTLENSP)) return 1;

  result_floats = vreva2_vf_vf(vf);
  memcpy(actual_floats, &result_floats, sizeof(actual_floats));
  return check_float("vreva2_vf_vf", actual_floats, expected_reva2_floats, VECTLENSP);
}

static int test_rounding_layout(void) {
#if defined(TEST_LSX)
  const double input[VECTLENDP] = { 1.75, -2.5 };
  const int32_t expected[VECTLENDP] = { 1, -2 };
  const int32_t expected_round[VECTLENDP] = { 2, -2 };
#else
  const double input[VECTLENDP] = { 1.75, -2.5, 3.75, -4.5 };
  const int32_t expected[VECTLENDP] = { 1, -2, 3, -4 };
  const int32_t expected_round[VECTLENDP] = { 2, -2, 4, -4 };
#endif
  int32_t actual[VECTLENSP];
  vdouble vd;
  memcpy(&vd, input, sizeof(input));
  vint vi = vtruncate_vi_vd(vd);
  memcpy(actual, &vi, sizeof(int32_t) * VECTLENDP);
  if (check_int32("vtruncate_vi_vd", actual, expected, VECTLENDP)) return 1;

  vi = vrint_vi_vd(vd);
  memcpy(actual, &vi, sizeof(int32_t) * VECTLENDP);
  return check_int32("vrint_vi_vd", actual, expected_round, VECTLENDP);
}

static int test_lasx_vint_layout(void) {
#if !defined(TEST_LASX)
  return 0;
#else
  const int32_t input[VECTLENDP] = {
    INT32_MIN, 0x12345678, -1, INT32_MAX
  };
  const double expected_double[VECTLENDP] = {
    (double)INT32_MIN, 0x12345678, -1.0, (double)INT32_MAX
  };
  int32_t actual[VECTLENDP];
  double actual_double[VECTLENDP];
  vint vi;
  memcpy(&vi, input, sizeof(input));

  vdouble vd = vcast_vd_vi(vi);
  memcpy(actual_double, &vd, sizeof(actual_double));
  if (check_double("vcast_vd_vi", actual_double, expected_double, VECTLENDP)) return 1;

  vmask signed_mask = vcast_vm_vi(vi);
  vint signed_roundtrip = vcast_vi_vm(signed_mask);
  memcpy(actual, &signed_roundtrip, sizeof(actual));
  if (check_int32("vcast_vm_vi/vcast_vi_vm", actual, input, VECTLENDP)) return 1;

  vmask unsigned_mask = vcastu_vm_vi(vi);
  vint unsigned_roundtrip = vcastu_vi_vm(unsigned_mask);
  memcpy(actual, &unsigned_roundtrip, sizeof(actual));
  if (check_int32("vcastu_vm_vi/vcastu_vi_vm", actual, input, VECTLENDP)) return 1;

  const int32_t x_values[VECTLENDP] = { 11, 22, 33, 44 };
  const int32_t y_values[VECTLENDP] = { -1, -2, -3, -4 };
  const int32_t expected_select[VECTLENDP] = { -1, 22, 33, -4 };
  const uint32_t mask_words[VECTLENSP] = {
    0, UINT32_MAX, UINT32_MAX, 0, 0, 0, 0, 0
  };
  vopmask mask;
  vint x, y;
  memcpy(&mask, mask_words, sizeof(mask));
  memcpy(&x, x_values, sizeof(x));
  memcpy(&y, y_values, sizeof(y));
  vint selected = vsel_vi_vo_vi_vi(mask, x, y);
  memcpy(actual, &selected, sizeof(actual));
  return check_int32("vsel_vi_vo_vi_vi", actual, expected_select, VECTLENDP);
#endif
}

static int test_fma_signed_zero(void) {
  uint64_t actual_dp[VECTLENDP];
  uint32_t actual_sp[VECTLENSP];
  vdouble x_dp = vcast_vd_d(1.0);
  vdouble y_dp = vcast_vd_d(1.0);
  vdouble z_dp = vcast_vd_d(1.0);
  vfloat x_sp = vcast_vf_f(1.0f);
  vfloat y_sp = vcast_vf_f(1.0f);
  vfloat z_sp = vcast_vf_f(1.0f);

  vdouble result_dp = vfmanp_vd_vd_vd_vd(x_dp, y_dp, z_dp);
  memcpy(actual_dp, &result_dp, sizeof(actual_dp));
  for (int i = 0; i < VECTLENDP; i++) {
    if (actual_dp[i] != UINT64_C(0)) {
      fprintf(stderr, "vfmanp double: lane %d: got 0x%016llx, expected +0\n",
              i, (unsigned long long)actual_dp[i]);
      return 1;
    }
  }

  result_dp = vmlanp_vd_vd_vd_vd(x_dp, y_dp, z_dp);
  memcpy(actual_dp, &result_dp, sizeof(actual_dp));
  for (int i = 0; i < VECTLENDP; i++) {
    if (actual_dp[i] != UINT64_C(0)) {
      fprintf(stderr, "vmlanp double: lane %d: got 0x%016llx, expected +0\n",
              i, (unsigned long long)actual_dp[i]);
      return 1;
    }
  }

  vfloat result_sp = vfmanp_vf_vf_vf_vf(x_sp, y_sp, z_sp);
  memcpy(actual_sp, &result_sp, sizeof(actual_sp));
  for (int i = 0; i < VECTLENSP; i++) {
    if (actual_sp[i] != UINT32_C(0)) {
      fprintf(stderr, "vfmanp float: lane %d: got 0x%08x, expected +0\n",
              i, actual_sp[i]);
      return 1;
    }
  }

  result_sp = vmlanp_vf_vf_vf_vf(x_sp, y_sp, z_sp);
  memcpy(actual_sp, &result_sp, sizeof(actual_sp));
  for (int i = 0; i < VECTLENSP; i++) {
    if (actual_sp[i] != UINT32_C(0)) {
      fprintf(stderr, "vmlanp float: lane %d: got 0x%08x, expected +0\n",
              i, actual_sp[i]);
      return 1;
    }
  }
  return 0;
}

int check_feature(double d, float f) {
  (void)f;
  vdouble value = vcast_vd_d(d);
#if defined(TEST_LSX)
  value = __lsx_vfadd_d(value, value);
#else
  value = __lasx_xvfadd_d(value, value);
#endif
  volatile double result[VECTLENDP];
  memcpy((void *)result, &value, sizeof(result));
  return result[0] == d + d;
}

int main2(int argc, char **argv) {
  (void)argc;
  (void)argv;
  if (test_mask_conversion_layout()) return 1;
  if (test_mask_select_layout()) return 1;
  if (test_integer_layout()) return 1;
  if (test_numeric_layout()) return 1;
  if (test_rounding_layout()) return 1;
  if (test_lasx_vint_layout()) return 1;
  if (test_fma_signed_zero()) return 1;
  return 0;
}
