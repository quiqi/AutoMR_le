import ctypes


class Interface:
    def __init__(self, so_path):
        self.cmath = ctypes.cdll.LoadLibrary(so_path)
        self.cmath.py_asin.restype = ctypes.c_double
        self.cmath.py_atan.restype = ctypes.c_double
        self.cmath.py_atan2.restype = ctypes.c_double
        self.cmath.py_cos.restype = ctypes.c_double
        self.cmath.py_sin.restype = ctypes.c_double
        self.cmath.py_tan.restype = ctypes.c_double
        self.cmath.py_cosh.restype = ctypes.c_double
        self.cmath.py_sinh.restype = ctypes.c_double
        self.cmath.py_tanh.restype = ctypes.c_double
        self.cmath.py_exp.restype = ctypes.c_double
        self.cmath.py_log.restype = ctypes.c_double
        self.cmath.py_log10.restype = ctypes.c_double
        self.cmath.py_pow.restype = ctypes.c_double
        self.cmath.py_sqrt.restype = ctypes.c_double
        self.cmath.py_ceil.restype = ctypes.c_double
        self.cmath.py_fabs.restype = ctypes.c_double
        self.cmath.py_floor.restype = ctypes.c_double
        self.cmath.py_fmod.restype = ctypes.c_double
        self.cmath.py_erf.restype = ctypes.c_double
        self.cmath.py_erfc.restype = ctypes.c_double
        self.cmath.py_gamma.restype = ctypes.c_double
        self.cmath.py_hypot.restype = ctypes.c_double
        self.cmath.py_isnan.restype = ctypes.c_int
        self.cmath.py_finite.restype = ctypes.c_int
        self.cmath.py_j0.restype = ctypes.c_double
        self.cmath.py_j1.restype = ctypes.c_double
        self.cmath.py_jn.restype = ctypes.c_double
        self.cmath.py_lgamma.restype = ctypes.c_double
        self.cmath.py_y0.restype = ctypes.c_double
        self.cmath.py_y1.restype = ctypes.c_double
        self.cmath.py_yn.restype = ctypes.c_double
        self.cmath.py_acosh.restype = ctypes.c_double
        self.cmath.py_asinh.restype = ctypes.c_double
        self.cmath.py_atanh.restype = ctypes.c_double
        self.cmath.py_cbrt.restype = ctypes.c_double
        self.cmath.py_logb.restype = ctypes.c_double
        self.cmath.py_nextafter.restype = ctypes.c_double
        self.cmath.py_remainder.restype = ctypes.c_double
        self.cmath.py_scalb_di.restype = ctypes.c_double
        self.cmath.py_scalb_dd.restype = ctypes.c_double
        self.cmath.py_significand.restype = ctypes.c_double
        self.cmath.py_copysign.restype = ctypes.c_double
        self.cmath.py_ilogb.restype = ctypes.c_int
        self.cmath.py_rint.restype = ctypes.c_double
        self.cmath.py_scalbn.restype = ctypes.c_double
        self.cmath.py_expm1.restype = ctypes.c_double
        self.cmath.py_log1p.restype = ctypes.c_double

    def asin(self, x):
        return self.cmath.py_asin(ctypes.c_double(x))

    def atan(self, x):
        return self.cmath.py_atan(ctypes.c_double(x))

    def atan2(self, x):
        return self.cmath.py_atan2(ctypes.c_double(x))

    def cos(self, x):
        return self.cmath.py_cos(ctypes.c_double(x))

    def sin(self, x):
        return self.cmath.py_sin(ctypes.c_double(x))

    def tan(self, x):
        return self.cmath.py_tan(ctypes.c_double(x))

    def cosh(self, x):
        return self.cmath.py_cosh(ctypes.c_double(x))

    def sinh(self, x):
        return self.cmath.py_sinh(ctypes.c_double(x))

    def tanh(self, x):
        return self.cmath.py_tanh(ctypes.c_double(x))

    def exp(self, x):
        return self.cmath.py_exp(ctypes.c_double(x))

    def log(self, x):
        return self.cmath.py_log(ctypes.c_double(x))

    def log10(self, x):
        return self.cmath.py_log10(ctypes.c_double(x))

    def pow(self, x):
        return self.cmath.py_pow(ctypes.c_double(x))

    def sqrt(self, x):
        return self.cmath.py_sqrt(ctypes.c_double(x))

    def ceil(self, x):
        return self.cmath.py_ceil(ctypes.c_double(x))

    def fabs(self, x):
        return self.cmath.py_fabs(ctypes.c_double(x))

    def floor(self, x):
        return self.cmath.py_floor(ctypes.c_double(x))

    def fmod(self, x):
        return self.cmath.py_fmod(ctypes.c_double(x))

    def erf(self, x):
        return self.cmath.py_erf(ctypes.c_double(x))

    def erfc(self, x):
        return self.cmath.py_erfc(ctypes.c_double(x))

    def gamma(self, x):
        return self.cmath.py_gamma(ctypes.c_double(x))

    def hypot(self, x1, x2):
        return self.cmath.py_hypot(ctypes.c_double(x1), ctypes.c_double(x2))

    def isnan(self, x):
        return self.cmath.py_isnan(ctypes.c_double(x))

    def finite(self, x):
        return self.cmath.py_finite(ctypes.c_double(x))

    def j0(self, x):
        return self.cmath.py_j0(ctypes.c_double(x))

    def j1(self, x):
        return self.cmath.py_j1(ctypes.c_double(x))

    def jn(self, x1, x2):
        return self.cmath.py_jn(ctypes.c_int(x1), ctypes.c_double(x2))

    def lgamma(self, x):
        return self.cmath.py_lgamma(ctypes.c_double(x))

    def y0(self, x):
        return self.cmath.py_y0(ctypes.c_double(x))

    def y1(self, x):
        return self.cmath.py_y1(ctypes.c_double(x))

    def yn(self, x1, x2):
        return self.cmath.py_yn(ctypes.c_int(x1), ctypes.c_double(x2))

    def acosh(self, x):
        return self.cmath.py_acosh(ctypes.c_double(x))

    def asinh(self, x):
        return self.cmath.py_asinh(ctypes.c_double(x))

    def atanh(self, x):
        return self.cmath.py_atanh(ctypes.c_double(x))

    def cbrt(self, x):
        return self.cmath.py_cbrt(ctypes.c_double(x))

    def logb(self, x):
        return self.cmath.py_logb(ctypes.c_double(x))

    def nextafter(self, x):
        return self.cmath.py_nextafter(ctypes.c_double(x))

    def remainder(self, x):
        return self.cmath.py_remainder(ctypes.c_double(x))

    def scalb_di(self, x1, x2):
        return self.cmath.py_scalb_di(ctypes.c_double(x1), ctypes.c_int(x2))

    def scalb_dd(self, x1, x2):
        return self.cmath.py_scalb_dd(ctypes.c_double(x1), ctypes.c_double(x2))

    def significand(self, x):
        return self.cmath.py_significand(ctypes.c_double(x))

    def copysign(self, x):
        return self.cmath.py_copysign(ctypes.c_double(x))

    def ilogb(self, x):
        return self.cmath.py_ilogb(ctypes.c_double(x))

    def rint(self, x):
        return self.cmath.py_rint(ctypes.c_double(x))

    def scalbn(self, x1, x2):
        return self.cmath.py_scalbn(ctypes.c_double(x1), ctypes.c_int(x2))

    def expm1(self, x):
        return self.cmath.py_expm1(ctypes.c_double(x))

    def log1p(self, x):
        return self.cmath.py_log1p(ctypes.c_double(x))


if __name__ == '__main__':
    cmath = Interface('./doc/_fdlibm.so')
    a = cmath.exp(1)
    print(a)
