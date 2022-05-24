import ctypes


class Interface:
    def __init__(self, so_path: str = './put.so'):
        self.so = ctypes.cdll.LoadLibrary(so_path)
        self.so.py_acos.restype = ctypes.c_double
        self.so.py_asin.restype = ctypes.c_double
        self.so.py_atan.restype = ctypes.c_double
        self.so.py_atan2.restype = ctypes.c_double
        self.so.py_cos.restype = ctypes.c_double
        self.so.py_sin.restype = ctypes.c_double
        self.so.py_tan.restype = ctypes.c_double
        self.so.py_cosh.restype = ctypes.c_double
        self.so.py_sinh.restype = ctypes.c_double
        self.so.py_tanh.restype = ctypes.c_double
        self.so.py_exp.restype = ctypes.c_double
        self.so.py_frexp.restype = ctypes.c_double
        self.so.py_ldexp.restype = ctypes.c_double
        self.so.py_log.restype = ctypes.c_double
        self.so.py_log10.restype = ctypes.c_double
        self.so.py_modf.restype = ctypes.c_double
        self.so.py_pow.restype = ctypes.c_double
        self.so.py_sqrt.restype = ctypes.c_double
        self.so.py_ceil.restype = ctypes.c_double
        self.so.py_fabs.restype = ctypes.c_double
        self.so.py_floor.restype = ctypes.c_double
        self.so.py_fmod.restype = ctypes.c_double
        self.so.py_erf.restype = ctypes.c_double
        self.so.py_erfc.restype = ctypes.c_double
        self.so.py_gamma.restype = ctypes.c_double
        self.so.py_hypot.restype = ctypes.c_double
        self.so.py_isnan.restype = ctypes.c_int
        self.so.py_finite.restype = ctypes.c_int
        self.so.py_j0.restype = ctypes.c_double
        self.so.py_j1.restype = ctypes.c_double
        self.so.py_jn.restype = ctypes.c_double
        self.so.py_lgamma.restype = ctypes.c_double
        self.so.py_y0.restype = ctypes.c_double
        self.so.py_y1.restype = ctypes.c_double
        self.so.py_yn.restype = ctypes.c_double
        self.so.py_acosh.restype = ctypes.c_double
        self.so.py_asinh.restype = ctypes.c_double
        self.so.py_atanh.restype = ctypes.c_double
        self.so.py_cbrt.restype = ctypes.c_double
        self.so.py_logb.restype = ctypes.c_double
        self.so.py_nextafter.restype = ctypes.c_double
        self.so.py_remainder.restype = ctypes.c_double
        self.so.py_scalb.restype = ctypes.c_double
        self.so.py_scalb.restype = ctypes.c_double
        self.so.py_significand.restype = ctypes.c_double
        self.so.py_copysign.restype = ctypes.c_double
        self.so.py_ilogb.restype = ctypes.c_int
        self.so.py_rint.restype = ctypes.c_double
        self.so.py_scalbn.restype = ctypes.c_double
        self.so.py_expm1.restype = ctypes.c_double
        self.so.py_log1p.restype = ctypes.c_double
        self.so.py_gamma_r.restype = ctypes.c_double
        self.so.py_lgamma_r.restype = ctypes.c_double
        self.so.py___ieee754_sqrt.restype = ctypes.c_double
        self.so.py___ieee754_acos.restype = ctypes.c_double
        self.so.py___ieee754_acosh.restype = ctypes.c_double
        self.so.py___ieee754_log.restype = ctypes.c_double
        self.so.py___ieee754_atanh.restype = ctypes.c_double
        self.so.py___ieee754_asin.restype = ctypes.c_double
        self.so.py___ieee754_atan2.restype = ctypes.c_double
        self.so.py___ieee754_exp.restype = ctypes.c_double
        self.so.py___ieee754_cosh.restype = ctypes.c_double
        self.so.py___ieee754_fmod.restype = ctypes.c_double
        self.so.py___ieee754_pow.restype = ctypes.c_double
        self.so.py___ieee754_lgamma_r.restype = ctypes.c_double
        self.so.py___ieee754_gamma_r.restype = ctypes.c_double
        self.so.py___ieee754_lgamma.restype = ctypes.c_double
        self.so.py___ieee754_gamma.restype = ctypes.c_double
        self.so.py___ieee754_log10.restype = ctypes.c_double
        self.so.py___ieee754_sinh.restype = ctypes.c_double
        self.so.py___ieee754_hypot.restype = ctypes.c_double
        self.so.py___ieee754_j0.restype = ctypes.c_double
        self.so.py___ieee754_j1.restype = ctypes.c_double
        self.so.py___ieee754_y0.restype = ctypes.c_double
        self.so.py___ieee754_y1.restype = ctypes.c_double
        self.so.py___ieee754_jn.restype = ctypes.c_double
        self.so.py___ieee754_yn.restype = ctypes.c_double
        self.so.py___ieee754_remainder.restype = ctypes.c_double
        self.so.py___ieee754_rem_pio2.restype = ctypes.c_int
        self.so.py___ieee754_scalb.restype = ctypes.c_double
        self.so.py___ieee754_scalb.restype = ctypes.c_double
        self.so.py___kernel_standard.restype = ctypes.c_double
        self.so.py___kernel_sin.restype = ctypes.c_double
        self.so.py___kernel_cos.restype = ctypes.c_double
        self.so.py___kernel_tan.restype = ctypes.c_double
        self.so.py___kernel_rem_pio2.restype = ctypes.c_int

    def acos(self, x0):
        return self.so.py_acos(ctypes.c_double(x0))

    def asin(self, x0):
        return self.so.py_asin(ctypes.c_double(x0))

    def atan(self, x0):
        return self.so.py_atan(ctypes.c_double(x0))

    def atan2(self, x0, x1):
        return self.so.py_atan2(ctypes.c_double(x0), ctypes.c_double(x1))

    def cos(self, x0):
        return self.so.py_cos(ctypes.c_double(x0))

    def sin(self, x0):
        return self.so.py_sin(ctypes.c_double(x0))

    def tan(self, x0):
        return self.so.py_tan(ctypes.c_double(x0))

    def cosh(self, x0):
        return self.so.py_cosh(ctypes.c_double(x0))

    def sinh(self, x0):
        return self.so.py_sinh(ctypes.c_double(x0))

    def tanh(self, x0):
        return self.so.py_tanh(ctypes.c_double(x0))

    def exp(self, x0):
        return self.so.py_exp(ctypes.c_double(x0))

    def frexp(self, x0, x1):
        return self.so.py_frexp(ctypes.c_double(x0), ctypes.byref(ctypes.c_int(x1)))

    def ldexp(self, x0, x1):
        return self.so.py_ldexp(ctypes.c_double(x0), ctypes.c_int(x1))

    def log(self, x0):
        return self.so.py_log(ctypes.c_double(x0))

    def log10(self, x0):
        return self.so.py_log10(ctypes.c_double(x0))

    def modf(self, x0, x1):
        return self.so.py_modf(ctypes.c_double(x0), ctypes.byref(ctypes.c_double(x1)))

    def pow(self, x0, x1):
        return self.so.py_pow(ctypes.c_double(x0), ctypes.c_double(x1))

    def sqrt(self, x0):
        return self.so.py_sqrt(ctypes.c_double(x0))

    def ceil(self, x0):
        return self.so.py_ceil(ctypes.c_double(x0))

    def fabs(self, x0):
        return self.so.py_fabs(ctypes.c_double(x0))

    def floor(self, x0):
        return self.so.py_floor(ctypes.c_double(x0))

    def fmod(self, x0, x1):
        return self.so.py_fmod(ctypes.c_double(x0), ctypes.c_double(x1))

    def erf(self, x0):
        return self.so.py_erf(ctypes.c_double(x0))

    def erfc(self, x0):
        return self.so.py_erfc(ctypes.c_double(x0))

    def gamma(self, x0):
        return self.so.py_gamma(ctypes.c_double(x0))

    def hypot(self, x0, x1):
        return self.so.py_hypot(ctypes.c_double(x0), ctypes.c_double(x1))

    def isnan(self, x0):
        return self.so.py_isnan(ctypes.c_double(x0))

    def finite(self, x0):
        return self.so.py_finite(ctypes.c_double(x0))

    def j0(self, x0):
        return self.so.py_j0(ctypes.c_double(x0))

    def j1(self, x0):
        return self.so.py_j1(ctypes.c_double(x0))

    def jn(self, x0, x1):
        return self.so.py_jn(ctypes.c_int(x0), ctypes.c_double(x1))

    def lgamma(self, x0):
        return self.so.py_lgamma(ctypes.c_double(x0))

    def y0(self, x0):
        return self.so.py_y0(ctypes.c_double(x0))

    def y1(self, x0):
        return self.so.py_y1(ctypes.c_double(x0))

    def yn(self, x0, x1):
        return self.so.py_yn(ctypes.c_int(x0), ctypes.c_double(x1))

    def acosh(self, x0):
        return self.so.py_acosh(ctypes.c_double(x0))

    def asinh(self, x0):
        return self.so.py_asinh(ctypes.c_double(x0))

    def atanh(self, x0):
        return self.so.py_atanh(ctypes.c_double(x0))

    def cbrt(self, x0):
        return self.so.py_cbrt(ctypes.c_double(x0))

    def logb(self, x0):
        return self.so.py_logb(ctypes.c_double(x0))

    def nextafter(self, x0, x1):
        return self.so.py_nextafter(ctypes.c_double(x0), ctypes.c_double(x1))

    def remainder(self, x0, x1):
        return self.so.py_remainder(ctypes.c_double(x0), ctypes.c_double(x1))

    def scalb(self, x0, x1):
        return self.so.py_scalb(ctypes.c_double(x0), ctypes.c_int(x1))

    def scalb(self, x0, x1):
        return self.so.py_scalb(ctypes.c_double(x0), ctypes.c_double(x1))

    def significand(self, x0):
        return self.so.py_significand(ctypes.c_double(x0))

    def copysign(self, x0, x1):
        return self.so.py_copysign(ctypes.c_double(x0), ctypes.c_double(x1))

    def ilogb(self, x0):
        return self.so.py_ilogb(ctypes.c_double(x0))

    def rint(self, x0):
        return self.so.py_rint(ctypes.c_double(x0))

    def scalbn(self, x0, x1):
        return self.so.py_scalbn(ctypes.c_double(x0), ctypes.c_int(x1))

    def expm1(self, x0):
        return self.so.py_expm1(ctypes.c_double(x0))

    def log1p(self, x0):
        return self.so.py_log1p(ctypes.c_double(x0))

    def gamma_r(self, x0, x1):
        return self.so.py_gamma_r(ctypes.c_double(x0), ctypes.byref(ctypes.c_int(x1)))

    def lgamma_r(self, x0, x1):
        return self.so.py_lgamma_r(ctypes.c_double(x0), ctypes.byref(ctypes.c_int(x1)))

    def __ieee754_sqrt(self, x0):
        return self.so.py___ieee754_sqrt(ctypes.c_double(x0))

    def __ieee754_acos(self, x0):
        return self.so.py___ieee754_acos(ctypes.c_double(x0))

    def __ieee754_acosh(self, x0):
        return self.so.py___ieee754_acosh(ctypes.c_double(x0))

    def __ieee754_log(self, x0):
        return self.so.py___ieee754_log(ctypes.c_double(x0))

    def __ieee754_atanh(self, x0):
        return self.so.py___ieee754_atanh(ctypes.c_double(x0))

    def __ieee754_asin(self, x0):
        return self.so.py___ieee754_asin(ctypes.c_double(x0))

    def __ieee754_atan2(self, x0, x1):
        return self.so.py___ieee754_atan2(ctypes.c_double(x0), ctypes.c_double(x1))

    def __ieee754_exp(self, x0):
        return self.so.py___ieee754_exp(ctypes.c_double(x0))

    def __ieee754_cosh(self, x0):
        return self.so.py___ieee754_cosh(ctypes.c_double(x0))

    def __ieee754_fmod(self, x0, x1):
        return self.so.py___ieee754_fmod(ctypes.c_double(x0), ctypes.c_double(x1))

    def __ieee754_pow(self, x0, x1):
        return self.so.py___ieee754_pow(ctypes.c_double(x0), ctypes.c_double(x1))

    def __ieee754_lgamma_r(self, x0, x1):
        return self.so.py___ieee754_lgamma_r(ctypes.c_double(x0), ctypes.byref(ctypes.c_int(x1)))

    def __ieee754_gamma_r(self, x0, x1):
        return self.so.py___ieee754_gamma_r(ctypes.c_double(x0), ctypes.byref(ctypes.c_int(x1)))

    def __ieee754_lgamma(self, x0):
        return self.so.py___ieee754_lgamma(ctypes.c_double(x0))

    def __ieee754_gamma(self, x0):
        return self.so.py___ieee754_gamma(ctypes.c_double(x0))

    def __ieee754_log10(self, x0):
        return self.so.py___ieee754_log10(ctypes.c_double(x0))

    def __ieee754_sinh(self, x0):
        return self.so.py___ieee754_sinh(ctypes.c_double(x0))

    def __ieee754_hypot(self, x0, x1):
        return self.so.py___ieee754_hypot(ctypes.c_double(x0), ctypes.c_double(x1))

    def __ieee754_j0(self, x0):
        return self.so.py___ieee754_j0(ctypes.c_double(x0))

    def __ieee754_j1(self, x0):
        return self.so.py___ieee754_j1(ctypes.c_double(x0))

    def __ieee754_y0(self, x0):
        return self.so.py___ieee754_y0(ctypes.c_double(x0))

    def __ieee754_y1(self, x0):
        return self.so.py___ieee754_y1(ctypes.c_double(x0))

    def __ieee754_jn(self, x0, x1):
        return self.so.py___ieee754_jn(ctypes.c_int(x0), ctypes.c_double(x1))

    def __ieee754_yn(self, x0, x1):
        return self.so.py___ieee754_yn(ctypes.c_int(x0), ctypes.c_double(x1))

    def __ieee754_remainder(self, x0, x1):
        return self.so.py___ieee754_remainder(ctypes.c_double(x0), ctypes.c_double(x1))

    def __ieee754_rem_pio2(self, x0, x1):
        return self.so.py___ieee754_rem_pio2(ctypes.c_double(x0), ctypes.byref(ctypes.c_double(x1)))

    def __ieee754_scalb(self, x0, x1):
        return self.so.py___ieee754_scalb(ctypes.c_double(x0), ctypes.c_int(x1))

    def __ieee754_scalb(self, x0, x1):
        return self.so.py___ieee754_scalb(ctypes.c_double(x0), ctypes.c_double(x1))

    def __kernel_standard(self, x0, x1, x2):
        return self.so.py___kernel_standard(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_int(x2))

    def __kernel_sin(self, x0, x1, x2):
        return self.so.py___kernel_sin(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_int(x2))

    def __kernel_cos(self, x0, x1):
        return self.so.py___kernel_cos(ctypes.c_double(x0), ctypes.c_double(x1))

    def __kernel_tan(self, x0, x1, x2):
        return self.so.py___kernel_tan(ctypes.c_double(x0), ctypes.c_double(x1), ctypes.c_int(x2))

    def __kernel_rem_pio2(self, x0, x1, x2, x3, x4, x5):
        return self.so.py___kernel_rem_pio2(ctypes.byref(ctypes.c_double(x0)), ctypes.byref(ctypes.c_double(x1)),
                                            ctypes.c_int(x2), ctypes.c_int(x3), ctypes.c_int(x4),
                                            ctypes.byref(ctypes.c_int(x5)))
