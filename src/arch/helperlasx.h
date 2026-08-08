/* LoongArch LASX ISA backend. */
#if !defined(__loongarch_asx) && !defined(SLEEF_GENHEADER)
#error Please specify -mlasx.
#endif

#if !defined(SLEEF_GENHEADER)
#include <lsxintrin.h>
#include <lasxintrin.h>
#include <stdint.h>
#include "misc.h"
#endif

#ifndef CONFIG
#error CONFIG macro not defined
#endif

#define ENABLE_DP
//@#define ENABLE_DP
#define ENABLE_SP
//@#define ENABLE_SP
#define LOG2VECTLENDP 2
//@#define LOG2VECTLENDP 2
#define VECTLENDP (1 << LOG2VECTLENDP)
//@#define VECTLENDP (1 << LOG2VECTLENDP)
#define LOG2VECTLENSP (LOG2VECTLENDP + 1)
//@#define LOG2VECTLENSP (LOG2VECTLENDP + 1)
#define VECTLENSP (1 << LOG2VECTLENSP)
//@#define VECTLENSP (1 << LOG2VECTLENSP)
#define DFTPRIORITY LOG2VECTLENDP
#define FULL_FP_ROUNDING
//@#define FULL_FP_ROUNDING
#define ACCURATE_SQRT
//@#define ACCURATE_SQRT
#define ISANAME "LoongArch LASX"
#if CONFIG == 1
#define ENABLE_FMA_DP
//@#define ENABLE_FMA_DP
#define ENABLE_FMA_SP
//@#define ENABLE_FMA_SP
#endif

typedef v8i32 vmask;
typedef v8i32 vopmask;
typedef __m256 vfloat;
typedef v8i32 vint2;
typedef __m256d vdouble;
typedef v4i32 vint;
typedef v4i64 vint64;
typedef v4u64 vuint64;

typedef struct { vmask x, y; } vquad;
typedef vquad vargquad;

#if VECTLENDP == 2
static INLINE vopmask vcast_vo32_vo64(vopmask m) { return (vopmask){ m[1], m[3], 0, 0 }; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return (vopmask){ m[0], m[0], m[1], m[1] }; }

static INLINE vint vcast_vi_i(int i) { return (vint) { i, i }; }
static INLINE vint2 vcast_vi2_i(int i) { return (vint2) { i, i, i, i }; }
static INLINE vfloat vcast_vf_f(float f) { return (vfloat) { f, f, f, f }; }
static INLINE vdouble vcast_vd_d(double d) { return (vdouble) { d, d }; }
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vcast_vl_l(long double d) { return (vlongdouble) { d, d }; }
#endif
static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], 0, 0 }; }

static INLINE vdouble vrev21_vd_vd(vdouble vd) { return (vdouble) { vd[1], vd[0] }; }
static INLINE vdouble vreva2_vd_vd(vdouble vd) { return vd; }
static INLINE vfloat vrev21_vf_vf(vfloat vd) { return (vfloat) { vd[1], vd[0], vd[3], vd[2] }; }
static INLINE vfloat vreva2_vf_vf(vfloat vd) { return (vfloat) { vd[2], vd[3], vd[0], vd[1] }; }
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vrev21_vl_vl(vlongdouble vd) { return (vlongdouble) { vd[1], vd[0] }; }
static INLINE vlongdouble vreva2_vl_vl(vlongdouble vd) { return vd; }
static INLINE vlongdouble vposneg_vl_vl(vlongdouble vd) { return (vlongdouble) { +vd[0], -vd[1] }; }
static INLINE vlongdouble vnegpos_vl_vl(vlongdouble vd) { return (vlongdouble) { -vd[0], +vd[1] }; }
#endif

#define PNMASK ((vdouble) { +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0 })
static INLINE vdouble vposneg_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)PNMASK); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)NPMASK); }

#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f })
static INLINE vfloat vposneg_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)PNMASKf); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)NPMASKf); }
#elif VECTLENDP == 4
static INLINE vopmask vcast_vo32_vo64(vopmask m) {
  __m256i t = __lasx_xvpickod_w((__m256i)m, (__m256i)m);
  return (vopmask)__lasx_xvpermi_d(t, 0x58);
}
static INLINE vopmask vcast_vo64_vo32(vopmask m) {
  __m256i t = __lasx_xvpermi_d((__m256i)m, 0x50);
  return (vopmask)__lasx_xvilvl_w(t, t);
}

static INLINE vint vcast_vi_i(int i) { return (vint)__lsx_vreplgr2vr_w(i); }
static INLINE vint2 vcast_vi2_i(int i) { return (vint2)__lasx_xvreplgr2vr_w(i); }
static INLINE vfloat vcast_vf_f(float f) { union { float f; int32_t i; } u = { f }; return (vfloat)__lasx_xvreplgr2vr_w(u.i); }
static INLINE vdouble vcast_vd_d(double d) { union { double d; int64_t i; } u = { d }; return (vdouble)__lasx_xvreplgr2vr_d(u.i); }
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vcast_vl_l(long double d) { return (vlongdouble) { d, d, d, d }; }
#endif

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h, l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) {
  return (vint2) { 0, vi[0], 0, vi[1], 0, vi[2], 0, vi[3] };
}
static INLINE vint vcastu_vi_vi2(vint2 vi2) {
  return (vint) { vi2[1], vi2[3], vi2[5], vi2[7] };
}

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) {
  return (vint) { vi2[0], vi2[1], vi2[2], vi2[3] };
}
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) {
  return (vint2) { vi[0], vi[1], vi[2], vi[3], 0, 0, 0, 0 };
}

#define PNMASK ((vdouble) { +0.0, -0.0, +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0, -0.0, +0.0 })
static INLINE vdouble vposneg_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)PNMASK); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)NPMASK); }

#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f })
static INLINE vfloat vposneg_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)PNMASKf); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)NPMASKf); }

static INLINE vdouble vrev21_vd_vd(vdouble vd) { return (vdouble)__lasx_xvshuf4i_w((__m256i)vd, 0x4e); }
static INLINE vdouble vreva2_vd_vd(vdouble vd) { return (vdouble)__lasx_xvpermi_d((__m256i)vd, 0x4e); }
static INLINE vfloat vrev21_vf_vf(vfloat vd) { return (vfloat)__lasx_xvshuf4i_w((__m256i)vd, 0xb1); }
static INLINE vfloat vreva2_vf_vf(vfloat vd) { return (vfloat)__lasx_xvpermi_d((__m256i)vd, 0x1b); }
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vrev21_vl_vl(vlongdouble vd) { return (vlongdouble) { vd[1], vd[0], vd[3], vd[2] }; }
static INLINE vlongdouble vreva2_vl_vl(vlongdouble vd) { return (vlongdouble) { vd[2], vd[3], vd[0], vd[1] }; }
static INLINE vlongdouble vposneg_vl_vl(vlongdouble vd) { return (vlongdouble) { +vd[0], -vd[1], +vd[2], -vd[3] }; }
static INLINE vlongdouble vnegpos_vl_vl(vlongdouble vd) { return (vlongdouble) { -vd[0], +vd[1], -vd[2], +vd[3] }; }
#endif
#elif VECTLENDP == 8
static INLINE vopmask vcast_vo32_vo64(vopmask m) { return (vopmask){ m[1], m[3], m[5], m[7], m[9], m[11], m[13], m[15], 0, 0, 0, 0, 0, 0, 0, 0 }; }
static INLINE vopmask vcast_vo64_vo32(vopmask m) { return (vopmask){ m[0], m[0], m[1], m[1], m[2], m[2], m[3], m[3], m[4], m[4], m[5], m[5], m[6], m[6], m[7], m[7] }; }

static INLINE vint vcast_vi_i(int i) { return (vint) { i, i, i, i, i, i, i, i }; }
static INLINE vint2 vcast_vi2_i(int i) { return (vint2) { i, i, i, i, i, i, i, i, i, i, i, i, i, i, i, i }; }
static INLINE vfloat vcast_vf_f(float f) { return (vfloat) { f, f, f, f, f, f, f, f, f, f, f, f, f, f, f, f }; }
static INLINE vdouble vcast_vd_d(double d) { return (vdouble) { d, d, d, d, d, d, d, d }; }
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vcast_vl_l(long double d) { return (vlongdouble) { d, d, d, d, d, d, d, d }; }
#endif

static INLINE vmask vcast_vm_i_i(int h, int l) { return (vmask){ l, h, l, h, l, h, l, h, l, h, l, h, l, h, l, h }; }
static INLINE vint2 vcastu_vi2_vi(vint vi) { return (vint2){ 0, vi[0], 0, vi[1], 0, vi[2], 0, vi[3], 0, vi[4], 0, vi[5], 0, vi[6], 0, vi[7] }; }
static INLINE vint vcastu_vi_vi2(vint2 vi2) { return (vint){ vi2[1], vi2[3], vi2[5], vi2[7], vi2[9], vi2[11], vi2[13], vi2[15] }; }

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) { return (vint){ vi2[0], vi2[1], vi2[2], vi2[3], vi2[4], vi2[5], vi2[6], vi2[7] }; }
static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) { return (vint2){ vi[0], vi[1], vi[2], vi[3], vi[4], vi[5], vi[6], vi[7], 0, 0, 0, 0, 0, 0, 0, 0 }; }

#define PNMASK ((vdouble) { +0.0, -0.0, +0.0, -0.0, +0.0, -0.0, +0.0, -0.0 })
#define NPMASK ((vdouble) { -0.0, +0.0, -0.0, +0.0, -0.0, +0.0, -0.0, +0.0 })
static INLINE vdouble vposneg_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)PNMASK); }
static INLINE vdouble vnegpos_vd_vd(vdouble d) { return (vdouble)((vmask)d ^ (vmask)NPMASK); }

#define PNMASKf ((vfloat) { +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f })
#define NPMASKf ((vfloat) { -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f })
static INLINE vfloat vposneg_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)PNMASKf); }
static INLINE vfloat vnegpos_vf_vf(vfloat d) { return (vfloat)((vmask)d ^ (vmask)NPMASKf); }

static INLINE vdouble vrev21_vd_vd(vdouble vd) { return (vdouble) { vd[1], vd[0], vd[3], vd[2], vd[5], vd[4], vd[7], vd[6] }; }
static INLINE vdouble vreva2_vd_vd(vdouble vd) { return (vdouble) { vd[6], vd[7], vd[4], vd[5], vd[2], vd[3], vd[0], vd[1] }; }
static INLINE vfloat vrev21_vf_vf(vfloat vd) {
  return (vfloat) {
    vd[1], vd[0], vd[3], vd[2], vd[5], vd[4], vd[7], vd[6],
      vd[9], vd[8], vd[11], vd[10], vd[13], vd[12], vd[15], vd[14] };
}
static INLINE vfloat vreva2_vf_vf(vfloat vd) {
  return (vfloat) {
    vd[14], vd[15], vd[12], vd[13], vd[10], vd[11], vd[8], vd[9],
      vd[6], vd[7], vd[4], vd[5], vd[2], vd[3], vd[0], vd[1]};
}
#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vrev21_vl_vl(vlongdouble vd) { return (vlongdouble) { vd[1], vd[0], vd[3], vd[2], vd[5], vd[4], vd[7], vd[6] }; }
static INLINE vlongdouble vreva2_vl_vl(vlongdouble vd) { return (vlongdouble) { vd[6], vd[7], vd[4], vd[5], vd[2], vd[3], vd[0], vd[1] }; }
static INLINE vlongdouble vposneg_vl_vl(vlongdouble vd) { return (vlongdouble) { +vd[0], -vd[1], +vd[2], -vd[3], +vd[4], -vd[5], +vd[6], -vd[7] }; }
static INLINE vlongdouble vnegpos_vl_vl(vlongdouble vd) { return (vlongdouble) { -vd[0], +vd[1], -vd[2], +vd[3], -vd[4], +vd[5], -vd[6], +vd[7] }; }
#endif
#else
static INLINE vint vcast_vi_i(int k) {
  vint ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = k;
  return ret;
}

static INLINE vint2 vcast_vi2_i(int k) {
  vint2 ret;
  for(int i=0;i<VECTLENSP;i++) ret[i] = k;
  return ret;
}

static INLINE vdouble vcast_vd_d(double d) {
  vdouble ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = d;
  return ret;
}

static INLINE vfloat vcast_vf_f(float f) {
  vfloat ret;
  for(int i=0;i<VECTLENSP;i++) ret[i] = f;
  return ret;
}

#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vcast_vl_l(long double d) {
  vlongdouble ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = d;
  return ret;
}
#endif

static INLINE vopmask vcast_vo32_vo64(vopmask m) {
  vopmask ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = m[i*2+1];
  for(int i=VECTLENDP;i<VECTLENDP*2;i++) ret[i] = 0;
  return ret;
}

static INLINE vopmask vcast_vo64_vo32(vopmask m) {
  vopmask ret;
  for(int i=0;i<VECTLENDP;i++) ret[i*2] = ret[i*2+1] = m[i];
  return ret;
}

static INLINE vmask vcast_vm_i_i(int h, int l) {
  vmask ret;
  for(int i=0;i<VECTLENDP;i++) {
    ret[i*2+0] = l;
    ret[i*2+1] = h;
  }
  return ret;
}

static INLINE vint2 vcastu_vi2_vi(vint vi) {
  vint2 ret;
  for(int i=0;i<VECTLENDP;i++) {
    ret[i*2+0] = 0;
    ret[i*2+1] = vi[i];
  }
  return ret;
}

static INLINE vint vcastu_vi_vi2(vint2 vi2) {
  vint ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = vi2[i*2+1];
  return ret;
}

static INLINE vint vreinterpretFirstHalf_vi_vi2(vint2 vi2) {
  vint ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = vi2[i];
  return ret;
}

static INLINE vint2 vreinterpretFirstHalf_vi2_vi(vint vi) {
  vint2 ret;
  for(int i=0;i<VECTLENDP;i++) ret[i] = vi[i];
  for(int i=VECTLENDP;i<VECTLENDP*2;i++) ret[i] = 0;
  return ret;
}

static INLINE vdouble vrev21_vd_vd(vdouble d0) {
  vdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = d0[i*2+1];
    r[i*2+1] = d0[i*2+0];
  }
  return r;
}

static INLINE vdouble vreva2_vd_vd(vdouble d0) {
  vdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = d0[(VECTLENDP/2-1-i)*2+0];
    r[i*2+1] = d0[(VECTLENDP/2-1-i)*2+1];
  }
  return r;
}

static INLINE vfloat vrev21_vf_vf(vfloat d0) {
  vfloat r;
  for(int i=0;i<VECTLENSP/2;i++) {
    r[i*2+0] = d0[i*2+1];
    r[i*2+1] = d0[i*2+0];
  }
  return r;
}

static INLINE vfloat vreva2_vf_vf(vfloat d0) {
  vfloat r;
  for(int i=0;i<VECTLENSP/2;i++) {
    r[i*2+0] = d0[(VECTLENSP/2-1-i)*2+0];
    r[i*2+1] = d0[(VECTLENSP/2-1-i)*2+1];
  }
  return r;
}

#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vrev21_vl_vl(vlongdouble d0) {
  vlongdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = d0[i*2+1];
    r[i*2+1] = d0[i*2+0];
  }
  return r;
}

static INLINE vlongdouble vreva2_vl_vl(vlongdouble d0) {
  vlongdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = d0[(VECTLENDP/2-1-i)*2+0];
    r[i*2+1] = d0[(VECTLENDP/2-1-i)*2+1];
  }
  return r;
}
#endif

static INLINE vdouble vposneg_vd_vd(vdouble d0) {
  vdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = +d0[i*2+0];
    r[i*2+1] = -d0[i*2+1];
  }
  return r;
}

static INLINE vdouble vnegpos_vd_vd(vdouble d0) {
  vdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = -d0[i*2+0];
    r[i*2+1] = +d0[i*2+1];
  }
  return r;
}

static INLINE vfloat vposneg_vf_vf(vfloat d0) {
  vfloat r;
  for(int i=0;i<VECTLENSP/2;i++) {
    r[i*2+0] = +d0[i*2+0];
    r[i*2+1] = -d0[i*2+1];
  }
  return r;
}

static INLINE vfloat vnegpos_vf_vf(vfloat d0) {
  vfloat r;
  for(int i=0;i<VECTLENSP/2;i++) {
    r[i*2+0] = -d0[i*2+0];
    r[i*2+1] = +d0[i*2+1];
  }
  return r;
}

#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vposneg_vl_vl(vlongdouble d0) {
  vlongdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = +d0[i*2+0];
    r[i*2+1] = -d0[i*2+1];
  }
  return r;
}

static INLINE vlongdouble vnegpos_vl_vl(vlongdouble d0) {
  vlongdouble r;
  for(int i=0;i<VECTLENDP/2;i++) {
    r[i*2+0] = -d0[i*2+0];
    r[i*2+1] = +d0[i*2+1];
  }
  return r;
}
#endif
#endif

//

static INLINE int vavailability_i(int name) { return -1; }
static INLINE void vprefetch_v_p(const void *ptr) { }

static INLINE int vtestallones_i_vo64(vopmask g) {
  return __lasx_xbnz_d((__m256i)g);
}

static INLINE int vtestallones_i_vo32(vopmask g) {
  return __lasx_xbnz_w((__m256i)g);
}

//

static vint2 vloadu_vi2_p(int32_t *p) {
  return (vint2)__lasx_xvld(p, 0);
}

static void vstoreu_v_p_vi2(int32_t *p, vint2 v) {
  __lasx_xvst((__m256i)v, p, 0);
}

static vint vloadu_vi_p(int32_t *p) {
  return (vint)__lsx_vld(p, 0);
}

static void vstoreu_v_p_vi(int32_t *p, vint v) {
  __lsx_vst((__m128i)v, p, 0);
}

//

static INLINE vmask vand_vm_vm_vm(vmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vm_vm(vmask x, vmask y) {
  return (vmask)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}
static INLINE vmask vor_vm_vm_vm(vmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vm_vm(vmask x, vmask y) { return x ^ y; }

static INLINE vopmask vand_vo_vo_vo(vopmask x, vopmask y) { return x & y; }
static INLINE vopmask vandnot_vo_vo_vo(vopmask x, vopmask y) {
  return (vopmask)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}
static INLINE vopmask vor_vo_vo_vo(vopmask x, vopmask y) { return x | y; }
static INLINE vopmask vxor_vo_vo_vo(vopmask x, vopmask y) { return x ^ y; }

static INLINE vmask vand_vm_vo64_vm(vopmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vo64_vm(vopmask x, vmask y) {
  return (vmask)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}
static INLINE vmask vor_vm_vo64_vm(vopmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vo64_vm(vopmask x, vmask y) { return x ^ y; }

static INLINE vmask vand_vm_vo32_vm(vopmask x, vmask y) { return x & y; }
static INLINE vmask vandnot_vm_vo32_vm(vopmask x, vmask y) {
  return (vmask)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}
static INLINE vmask vor_vm_vo32_vm(vopmask x, vmask y) { return x | y; }
static INLINE vmask vxor_vm_vo32_vm(vopmask x, vmask y) { return x ^ y; }

//

static INLINE vdouble vsel_vd_vo_vd_vd(vopmask o, vdouble x, vdouble y) {
  return (vdouble)__lasx_xvbitsel_v((__m256i)y, (__m256i)x, (__m256i)o);
}
static INLINE vint2 vsel_vi2_vo_vi2_vi2(vopmask o, vint2 x, vint2 y) {
  return (vint2)__lasx_xvbitsel_v((__m256i)y, (__m256i)x, (__m256i)o);
}

static INLINE CONST vdouble vsel_vd_vo_d_d(vopmask o, double v1, double v0) {
  return vsel_vd_vo_vd_vd(o, vcast_vd_d(v1), vcast_vd_d(v0));
}

static INLINE vdouble vsel_vd_vo_vo_d_d_d(vopmask o0, vopmask o1, double d0, double d1, double d2) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_d_d(o1, d1, d2));
}

static INLINE vdouble vsel_vd_vo_vo_vo_d_d_d_d(vopmask o0, vopmask o1, vopmask o2, double d0, double d1, double d2, double d3) {
  return vsel_vd_vo_vd_vd(o0, vcast_vd_d(d0), vsel_vd_vo_vd_vd(o1, vcast_vd_d(d1), vsel_vd_vo_d_d(o2, d2, d3)));
}

static INLINE vdouble vcast_vd_vi(vint vi) {
  vint2 t = (vint2) { vi[0], vi[1], 0, 0, vi[2], vi[3], 0, 0 };
  return __lasx_xvffintl_d_w((__m256i)t);
}
static INLINE vint vtruncate_vi_vd(vdouble vd) {
  vdouble high = (vdouble)__lasx_xvpermi_d((__m256i)vd, 0x0e);
  return vreinterpretFirstHalf_vi_vi2((vint2)__lasx_xvftintrz_w_d(high, vd));
}
static INLINE vint vrint_vi_vd(vdouble vd) {
  vdouble high = (vdouble)__lasx_xvpermi_d((__m256i)vd, 0x0e);
  return vreinterpretFirstHalf_vi_vi2((vint2)__lasx_xvftintrne_w_d(high, vd));
}
static INLINE vdouble vtruncate_vd_vd(vdouble vd) { return __lasx_xvfrintrz_d(vd); }
static INLINE vdouble vrint_vd_vd(vdouble vd) { return __lasx_xvfrintrne_d(vd); }

static INLINE vopmask veq64_vo_vm_vm(vmask x, vmask y) {
  return (vopmask)__lasx_xvseq_d((__m256i)x, (__m256i)y);
}

static INLINE vmask vadd64_vm_vm_vm(vmask x, vmask y) { return (vmask)__lasx_xvadd_d((__m256i)x, (__m256i)y); }

static INLINE vmask vcast_vm_i64(int64_t i) { return (vmask)__lasx_xvreplgr2vr_d(i); }
static INLINE vmask vcast_vm_u64(uint64_t i) { return (vmask)__lasx_xvreplgr2vr_d((long long)i); }
static INLINE vmask vcastu_vm_vi(vint vi) {
  return (vmask)vcastu_vi2_vi(vi);
}
static INLINE vint vcastu_vi_vm(vmask vm) {
  return (vint) { vm[1], vm[3], vm[5], vm[7] };
}
static INLINE vmask vcast_vm_vi(vint vi) {
  return (vmask)__lasx_vext2xv_d_w((__m256i)vreinterpretFirstHalf_vi2_vi(vi));
}
static INLINE vint vcast_vi_vm(vmask vm) {
  return (vint) { vm[0], vm[2], vm[4], vm[6] };
}
static INLINE vmask vsel_vm_vo64_vm_vm(vopmask m, vmask x, vmask y) {
  return (vmask)__lasx_xvbitsel_v((__m256i)y, (__m256i)x, (__m256i)m);
}
static INLINE vmask vsub64_vm_vm_vm(vmask x, vmask y) { return (vmask)__lasx_xvsub_d((__m256i)x, (__m256i)y); }
static INLINE vmask vneg64_vm_vm(vmask x) { return (vmask)__lasx_xvneg_d((__m256i)x); }
static INLINE vopmask vgt64_vo_vm_vm(vmask x, vmask y) { return (vopmask)__lasx_xvslt_d((__m256i)y, (__m256i)x); }
#define vsll64_vm_vm_i(x, c) ((vmask)__lasx_xvslli_d((__m256i)(x), c))
//@#define vsll64_vm_vm_i(x, c) ((vmask)__lasx_xvslli_d((__m256i)(x), c))
#define vsrl64_vm_vm_i(x, c) ((vmask)__lasx_xvsrli_d((__m256i)(x), c))
//@#define vsrl64_vm_vm_i(x, c) ((vmask)__lasx_xvsrli_d((__m256i)(x), c))

//

static INLINE vmask vreinterpret_vm_vd(vdouble vd) { return (vmask)vd; }
static INLINE vmask vreinterpret_vm_vi64(vint64 vi) { return (vmask)vi; }
static INLINE vint2 vreinterpret_vi2_vd(vdouble vd) { return (vint2)vd; }
static INLINE vdouble vreinterpret_vd_vi2(vint2 vi) { return (vdouble)vi; }
static INLINE vdouble vreinterpret_vd_vm(vmask vm) { return (vdouble)vm; }

static INLINE vdouble vadd_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfadd_d(x, y); }
static INLINE vdouble vsub_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfsub_d(x, y); }
static INLINE vdouble vmul_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfmul_d(x, y); }
static INLINE vdouble vdiv_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfdiv_d(x, y); }
static INLINE vdouble vrec_vd_vd(vdouble x) { return __lasx_xvfrecip_d(x); }

static INLINE vdouble vabs_vd_vd(vdouble d) { return (vdouble)__lasx_xvbitclri_d((__m256i)d, 63); }
static INLINE vdouble vneg_vd_vd(vdouble d) { return (vdouble)__lasx_xvbitrevi_d((__m256i)d, 63); }
#if CONFIG == 1
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmadd_d(x, y, z); }
// Use FMA with a sign-bit negation: FNMSUB has different signed-zero semantics for z - x*y.
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmadd_d(vneg_vd_vd(x), y, z); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmsub_d(x, y, z); }
#else
static INLINE vdouble vmla_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
static INLINE vdouble vmlanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsub_vd_vd_vd(z, vmul_vd_vd_vd(x, y)); }
static INLINE vdouble vmlapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsub_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }
#endif
static INLINE vdouble vfma_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmadd_d(x, y, z); }
static INLINE vdouble vfmanp_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmadd_d(vneg_vd_vd(x), y, z); }
static INLINE vdouble vfmapn_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return __lasx_xvfmsub_d(x, y, z); }
static INLINE vdouble vmax_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfmax_d(x, y); }
static INLINE vdouble vmin_vd_vd_vd(vdouble x, vdouble y) { return __lasx_xvfmin_d(x, y); }

static INLINE vdouble vsubadd_vd_vd_vd(vdouble x, vdouble y) { return vadd_vd_vd_vd(x, vnegpos_vd_vd(y)); }
static INLINE vdouble vmlsubadd_vd_vd_vd_vd(vdouble x, vdouble y, vdouble z) { return vsubadd_vd_vd_vd(vmul_vd_vd_vd(x, y), z); }

static INLINE vopmask veq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_ceq_d(x, y); }
static INLINE vopmask vneq_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_cune_d(x, y); }
static INLINE vopmask vlt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_clt_d(x, y); }
static INLINE vopmask vle_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_cle_d(x, y); }
static INLINE vopmask vgt_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_clt_d(y, x); }
static INLINE vopmask vge_vo_vd_vd(vdouble x, vdouble y) { return (vopmask)__lasx_xvfcmp_cle_d(y, x); }

static INLINE vint vadd_vi_vi_vi(vint x, vint y) { return x + y; }
static INLINE vint vsub_vi_vi_vi(vint x, vint y) { return x - y; }
static INLINE vint vneg_vi_vi(vint e) { return -e; }

static INLINE vint vand_vi_vi_vi(vint x, vint y) { return x & y; }
static INLINE vint vandnot_vi_vi_vi(vint x, vint y) {
  return (vint)__lsx_vandn_v((__m128i)x, (__m128i)y);
}
static INLINE vint vor_vi_vi_vi(vint x, vint y) { return x | y; }
static INLINE vint vxor_vi_vi_vi(vint x, vint y) { return x ^ y; }

static INLINE vint vand_vi_vo_vi(vopmask x, vint y) { return vreinterpretFirstHalf_vi_vi2((vint2)x) & y; }
static INLINE vint vandnot_vi_vo_vi(vopmask x, vint y) {
  return (vint)__lsx_vandn_v((__m128i)vreinterpretFirstHalf_vi_vi2((vint2)x), (__m128i)y);
}

#define vsll_vi_vi_i(x, c) ((vint)__lsx_vslli_w((__m128i)(x), (c)))
//@#define vsll_vi_vi_i(x, c) ((vint)__lasx_xvslli_w((__m256i)(x), (c)))
#define vsrl_vi_vi_i(x, c) ((vint)__lsx_vsrli_w((__m128i)(x), (c)))
//@#define vsrl_vi_vi_i(x, c) ((vint)__lasx_xvsrli_w((__m256i)(x), (c)))
#define vsra_vi_vi_i(x, c) ((vint)__lsx_vsrai_w((__m128i)(x), (c)))
//@#define vsra_vi_vi_i(x, c) ((vint)__lasx_xvsrai_w((__m256i)(x), (c)))

static INLINE vint veq_vi_vi_vi(vint x, vint y) { return x == y; }
static INLINE vint vgt_vi_vi_vi(vint x, vint y) { return (vint)__lsx_vslt_w((__m128i)y, (__m128i)x); }

static INLINE vopmask veq_vo_vi_vi(vint x, vint y) {
  return (vopmask)vreinterpretFirstHalf_vi2_vi((vint)__lsx_vseq_w((__m128i)x, (__m128i)y));
}
static INLINE vopmask vgt_vo_vi_vi(vint x, vint y) {
  return (vopmask)vreinterpretFirstHalf_vi2_vi((vint)__lsx_vslt_w((__m128i)y, (__m128i)x));
}

static INLINE vint vsel_vi_vo_vi_vi(vopmask m, vint x, vint y) {
  return (vint)__lsx_vbitsel_v((__m128i)y, (__m128i)x,
                               (__m128i)vreinterpretFirstHalf_vi_vi2((vint2)m));
}

static INLINE vopmask visinf_vo_vd(vdouble d) { return veq_vo_vd_vd(vabs_vd_vd(d), vcast_vd_d(SLEEF_INFINITY)); }
static INLINE vopmask vispinf_vo_vd(vdouble d) { return veq_vo_vd_vd(d, vcast_vd_d(SLEEF_INFINITY)); }
static INLINE vopmask visminf_vo_vd(vdouble d) { return veq_vo_vd_vd(d, vcast_vd_d(-SLEEF_INFINITY)); }
static INLINE vopmask visnan_vo_vd(vdouble d) { return (vopmask)__lasx_xvfcmp_cune_d(d, d); }

static INLINE vdouble vsqrt_vd_vd(vdouble d) {
#if !defined(SLEEF_ENABLE_ALTSQRT)
  return __lasx_xvfsqrt_d(d);
#else
#if defined(__clang__)
  typedef int64_t vi64 __attribute__((ext_vector_type(VECTLENDP)));
#else
  typedef int64_t vi64 __attribute__((vector_size(sizeof(int64_t)*VECTLENDP)));
#endif

  vdouble q = vcast_vd_d(1);

  vopmask o = (vopmask)(d < 8.636168555094445E-78);
  d = vsel_vd_vo_vd_vd(o, d * 1.157920892373162E77, d);

  q = vsel_vd_vo_vd_vd(o, vcast_vd_d(2.9387358770557188E-39), vcast_vd_d(1));

  q = (vdouble)vor_vm_vm_vm(vlt_vo_vd_vd(d, vcast_vd_d(0)), (vmask)q);

  vdouble x = (vdouble)(0x5fe6ec85e7de30daLL - ((vi64)(d + 1e-320) >> 1));
  x = x * (  3 - d * x * x);
  x = x * ( 12 - d * x * x);
  x = x * (768 - d * x * x);
  x *= 1.0 / (1 << 13);
  x = (d - (d * x) * (d * x)) * (x * 0.5) + d * x;

  return x * q;
#endif
}

static INLINE double vcast_d_vd(vdouble v) { return v[0]; }
static INLINE float vcast_f_vf(vfloat v) { return v[0]; }

static INLINE vdouble vload_vd_p(const double *ptr) { return (vdouble)__lasx_xvld(ptr, 0); }
static INLINE vdouble vloadu_vd_p(const double *ptr) {
  return (vdouble)__lasx_xvld(ptr, 0);
}

static INLINE vdouble vgather_vd_p_vi(const double *ptr, vint vi) {
  vdouble vd;
  for(int i=0;i<VECTLENDP;i++) vd[i] = ptr[vi[i]];
  return vd;
}

static INLINE void vstore_v_p_vd(double *ptr, vdouble v) { __lasx_xvst((__m256i)v, ptr, 0); }
static INLINE void vstoreu_v_p_vd(double *ptr, vdouble v) {
  __lasx_xvst((__m256i)v, ptr, 0);
}
static INLINE void vstream_v_p_vd(double *ptr, vdouble v) { *(vdouble *)ptr = v; }

static INLINE void vscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) {
  for(int i=0;i<VECTLENDP/2;i++) {
    *(ptr+(offset + step * i)*2 + 0) = v[i*2+0];
    *(ptr+(offset + step * i)*2 + 1) = v[i*2+1];
  }
}

static INLINE void vsscatter2_v_p_i_i_vd(double *ptr, int offset, int step, vdouble v) { vscatter2_v_p_i_i_vd(ptr, offset, step, v); }

//

static INLINE vfloat vsel_vf_vo_vf_vf(vopmask o, vfloat x, vfloat y) {
  return (vfloat)__lasx_xvbitsel_v((__m256i)y, (__m256i)x, (__m256i)o);
}

static INLINE CONST vfloat vsel_vf_vo_f_f(vopmask o, float v1, float v0) {
  return vsel_vf_vo_vf_vf(o, vcast_vf_f(v1), vcast_vf_f(v0));
}

static INLINE vfloat vsel_vf_vo_vo_f_f_f(vopmask o0, vopmask o1, float d0, float d1, float d2) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_f_f(o1, d1, d2));
}

static INLINE vfloat vsel_vf_vo_vo_vo_f_f_f_f(vopmask o0, vopmask o1, vopmask o2, float d0, float d1, float d2, float d3) {
  return vsel_vf_vo_vf_vf(o0, vcast_vf_f(d0), vsel_vf_vo_vf_vf(o1, vcast_vf_f(d1), vsel_vf_vo_f_f(o2, d2, d3)));
}

static INLINE vint2 vcast_vi2_vm(vmask vm) { return (vint2)vm; }
static INLINE vmask vcast_vm_vi2(vint2 vi) { return (vmask)vi; }

static INLINE vfloat vcast_vf_vi2(vint2 vi) { return __lasx_xvffint_s_w((__m256i)vi); }

static INLINE vint2 vtruncate_vi2_vf(vfloat vf) { return (vint2)__lasx_xvftintrz_w_s(vf); }

static INLINE vint2 vrint_vi2_vf(vfloat vf) { return (vint2)__lasx_xvftintrne_w_s(vf); }
static INLINE vfloat vtruncate_vf_vf(vfloat vd) { return __lasx_xvfrintrz_s(vd); }
static INLINE vfloat vrint_vf_vf(vfloat vd) { return __lasx_xvfrintrne_s(vd); }

static INLINE vmask vreinterpret_vm_vf(vfloat vf) { return (vmask)vf; }
static INLINE vfloat vreinterpret_vf_vm(vmask vm) { return (vfloat)vm; }
static INLINE vfloat vreinterpret_vf_vi2(vint2 vi) { return (vfloat)vi; }
static INLINE vint2 vreinterpret_vi2_vf(vfloat vf) { return (vint2)vf; }

static INLINE vfloat vadd_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfadd_s(x, y); }
static INLINE vfloat vsub_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfsub_s(x, y); }
static INLINE vfloat vmul_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfmul_s(x, y); }
static INLINE vfloat vdiv_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfdiv_s(x, y); }
static INLINE vfloat vrec_vf_vf(vfloat x) { return __lasx_xvfrecip_s(x); }

static INLINE vfloat vabs_vf_vf(vfloat f) { return (vfloat)__lasx_xvbitclri_w((__m256i)f, 31); }
static INLINE vfloat vneg_vf_vf(vfloat d) { return (vfloat)__lasx_xvbitrevi_w((__m256i)d, 31); }
#if CONFIG == 1
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmadd_s(x, y, z); }
// FNMSUB computes -(x*y-z), which has different signed-zero semantics
// from the required fma(-x, y, z).
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmadd_s(vneg_vf_vf(x), y, z); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmsub_s(x, y, z); }
#else
static INLINE vfloat vmla_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
static INLINE vfloat vmlanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(z, vmul_vf_vf_vf(x, y)); }
static INLINE vfloat vmlapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsub_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }
#endif
static INLINE vfloat vfma_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmadd_s(x, y, z); }
static INLINE vfloat vfmanp_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmadd_s(vneg_vf_vf(x), y, z); }
static INLINE vfloat vfmapn_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return __lasx_xvfmsub_s(x, y, z); }
static INLINE vfloat vmax_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfmax_s(x, y); }
static INLINE vfloat vmin_vf_vf_vf(vfloat x, vfloat y) { return __lasx_xvfmin_s(x, y); }

static INLINE vfloat vsubadd_vf_vf_vf(vfloat x, vfloat y) { return vadd_vf_vf_vf(x, vnegpos_vf_vf(y)); }
static INLINE vfloat vmlsubadd_vf_vf_vf_vf(vfloat x, vfloat y, vfloat z) { return vsubadd_vf_vf_vf(vmul_vf_vf_vf(x, y), z); }

static INLINE vopmask veq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_ceq_s(x, y); }
static INLINE vopmask vneq_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_cune_s(x, y); }
static INLINE vopmask vlt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_clt_s(x, y); }
static INLINE vopmask vle_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_cle_s(x, y); }
static INLINE vopmask vgt_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_clt_s(y, x); }
static INLINE vopmask vge_vo_vf_vf(vfloat x, vfloat y) { return (vopmask)__lasx_xvfcmp_cle_s(y, x); }

static INLINE vint2 vadd_vi2_vi2_vi2(vint2 x, vint2 y) { return x + y; }
static INLINE vint2 vsub_vi2_vi2_vi2(vint2 x, vint2 y) { return x - y; }
static INLINE vint2 vneg_vi2_vi2(vint2 e) { return -e; }

static INLINE vint2 vand_vi2_vi2_vi2(vint2 x, vint2 y) { return x & y; }
static INLINE vint2 vandnot_vi2_vi2_vi2(vint2 x, vint2 y) {
  return (vint2)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}
static INLINE vint2 vor_vi2_vi2_vi2(vint2 x, vint2 y) { return x | y; }
static INLINE vint2 vxor_vi2_vi2_vi2(vint2 x, vint2 y) { return x ^ y; }

static INLINE vint2 vand_vi2_vo_vi2(vopmask x, vint2 y) { return (vint2)x & y; }
static INLINE vint2 vandnot_vi2_vo_vi2(vopmask x, vint2 y) {
  return (vint2)__lasx_xvandn_v((__m256i)x, (__m256i)y);
}

#define vsll_vi2_vi2_i(x, c) ((vint2)__lasx_xvslli_w((__m256i)(x), (c)))
//@#define vsll_vi2_vi2_i(x, c) ((vint2)__lasx_xvslli_w((__m256i)(x), (c)))
#define vsrl_vi2_vi2_i(x, c) ((vint2)__lasx_xvsrli_w((__m256i)(x), (c)))
//@#define vsrl_vi2_vi2_i(x, c) ((vint2)__lasx_xvsrli_w((__m256i)(x), (c)))
#define vsra_vi2_vi2_i(x, c) ((vint2)__lasx_xvsrai_w((__m256i)(x), (c)))
//@#define vsra_vi2_vi2_i(x, c) ((vint2)__lasx_xvsrai_w((__m256i)(x), (c)))

static INLINE vopmask veq_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)(x == y); }
static INLINE vopmask vgt_vo_vi2_vi2(vint2 x, vint2 y) { return (vopmask)__lasx_xvslt_w((__m256i)y, (__m256i)x); }
static INLINE vint2 veq_vi2_vi2_vi2(vint2 x, vint2 y) { return x == y; }
static INLINE vint2 vgt_vi2_vi2_vi2(vint2 x, vint2 y) { return (vint2)__lasx_xvslt_w((__m256i)y, (__m256i)x); }

static INLINE vopmask visinf_vo_vf(vfloat d) { return veq_vo_vf_vf(vabs_vf_vf(d), vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask vispinf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(SLEEF_INFINITYf)); }
static INLINE vopmask visminf_vo_vf(vfloat d) { return veq_vo_vf_vf(d, vcast_vf_f(-SLEEF_INFINITYf)); }
static INLINE vopmask visnan_vo_vf(vfloat d) { return (vopmask)__lasx_xvfcmp_cune_s(d, d); }

static INLINE vfloat vsqrt_vf_vf(vfloat d) {
#if !defined(SLEEF_ENABLE_ALTSQRT)
  return __lasx_xvfsqrt_s(d);
#else
  vfloat q = vcast_vf_f(1);

  vopmask o = (vopmask)(d < 5.4210108624275221700372640043497e-20f); // 2^-64
  d = vsel_vf_vo_vf_vf(o, d * vcast_vf_f(18446744073709551616.0f), d); // 2^64
  q = vsel_vf_vo_vf_vf(o, vcast_vf_f(0.00000000023283064365386962890625f), vcast_vf_f(1)); // 2^-32
  q = (vfloat)vor_vm_vm_vm(vlt_vo_vf_vf(d, vcast_vf_f(0)), (vmask)q);

  vfloat x = (vfloat)(0x5f330de2 - (((vint2)d) >> 1));
  x = x * ( 3.0f - d * x * x);
  x = x * (12.0f - d * x * x);
  x *= 0.0625f;
  x = (d - (d * x) * (d * x)) * (x * 0.5) + d * x;

  return x * q;
#endif
}

static INLINE vfloat vload_vf_p(const float *ptr) { return (vfloat)__lasx_xvld(ptr, 0); }
static INLINE vfloat vloadu_vf_p(const float *ptr) {
  return (vfloat)__lasx_xvld(ptr, 0);
}

static INLINE vfloat vgather_vf_p_vi2(const float *ptr, vint2 vi2) {
  vfloat vf;
  for(int i=0;i<VECTLENSP;i++) vf[i] = ptr[vi2[i]];
  return vf;
}

static INLINE void vstore_v_p_vf(float *ptr, vfloat v) { __lasx_xvst((__m256i)v, ptr, 0); }
static INLINE void vstoreu_v_p_vf(float *ptr, vfloat v) {
  __lasx_xvst((__m256i)v, ptr, 0);
}
static INLINE void vstream_v_p_vf(float *ptr, vfloat v) { *(vfloat *)ptr = v; }

static INLINE void vscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) {
  for(int i=0;i<VECTLENSP/2;i++) {
    *(ptr+(offset + step * i)*2 + 0) = v[i*2+0];
    *(ptr+(offset + step * i)*2 + 1) = v[i*2+1];
  }
}

static INLINE void vsscatter2_v_p_i_i_vf(float *ptr, int offset, int step, vfloat v) { vscatter2_v_p_i_i_vf(ptr, offset, step, v); }

//

#ifdef ENABLE_LONGDOUBLE
static INLINE vlongdouble vadd_vl_vl_vl(vlongdouble x, vlongdouble y) { return x + y; }
static INLINE vlongdouble vsub_vl_vl_vl(vlongdouble x, vlongdouble y) { return x - y; }
static INLINE vlongdouble vmul_vl_vl_vl(vlongdouble x, vlongdouble y) { return x * y; }

static INLINE vlongdouble vneg_vl_vl(vlongdouble d) { return -d; }
static INLINE vlongdouble vsubadd_vl_vl_vl(vlongdouble x, vlongdouble y) { return vadd_vl_vl_vl(x, vnegpos_vl_vl(y)); }
static INLINE vlongdouble vmlsubadd_vl_vl_vl_vl(vlongdouble x, vlongdouble y, vlongdouble z) { return vsubadd_vl_vl_vl(vmul_vl_vl_vl(x, y), z); }

static INLINE vlongdouble vload_vl_p(const long double *ptr) { return *(vlongdouble *)ptr; }
static INLINE vlongdouble vloadu_vl_p(const long double *ptr) {
  vlongdouble vd;
  for(int i=0;i<VECTLENDP;i++) vd[i] = ptr[i];
  return vd;
}

static INLINE void vstore_v_p_vl(long double *ptr, vlongdouble v) { *(vlongdouble *)ptr = v; }
static INLINE void vstoreu_v_p_vl(long double *ptr, vlongdouble v) {
  for(int i=0;i<VECTLENDP;i++) ptr[i] = v[i];
}
static INLINE void vstream_v_p_vl(long double *ptr, vlongdouble v) { *(vlongdouble *)ptr = v; }

static INLINE void vscatter2_v_p_i_i_vl(long double *ptr, int offset, int step, vlongdouble v) {
  for(int i=0;i<VECTLENDP/2;i++) {
    *(ptr+(offset + step * i)*2 + 0) = v[i*2+0];
    *(ptr+(offset + step * i)*2 + 1) = v[i*2+1];
  }
}

static INLINE void vsscatter2_v_p_i_i_vl(long double *ptr, int offset, int step, vlongdouble v) { vscatter2_v_p_i_i_vl(ptr, offset, step, v); }
#endif

#if defined(Sleef_quad2_DEFINED) && defined(ENABLEFLOAT128)
/* vquad is an opaque pair of vector-sized bit containers. */
static vquad loadu_vq_p(void *p) {
  vquad vd;
  memcpy(&vd, p, VECTLENDP * 16);
  return vd;
}

static INLINE vquad cast_vq_aq(vargquad aq) {
  vquad vq;
  memcpy(&vq, &aq, VECTLENDP * 16);
  return vq;
}

static INLINE vargquad cast_aq_vq(vquad vq) {
  vargquad aq;
  memcpy(&aq, &vq, VECTLENDP * 16);
  return aq;
}
#endif
