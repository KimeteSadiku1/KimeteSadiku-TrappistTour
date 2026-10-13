"""
Mekanika orbitale e nevojshme per problemin (zevendeson librarine pykep 2.x,
e cila nuk instalohet ne Kaggle).

  lambert()    - si arrihet nga pika r1 te pika r2 per kohen tof
  propagate()  - ku ndodhet anija pas kohes t ne orbite
  flyby()      - si ndryshon shpejtesia kur anija kalon prane nje planeti
  elements()   - gjysmeboshti a, ekscentriciteti e dhe anomalia E e nje orbite
  Planet       - pozita dhe shpejtesia e nje planeti ne nje date te caktuar

Vektoret jane tuple (x, y, z). Njesite: metra, sekonda.
"""
import math

AU = 149597870691.0          # njesia astronomike [m]
DAY = 86400.0                # nje dite [s]
EARTH_RADIUS = 6378137.0     # [m]


# ---------- veprime me vektore ----------
def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def norm(a):
    return math.sqrt(dot(a, a))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def comb(a, ka, b, kb):
    """ka*a + kb*b"""
    return (ka * a[0] + kb * b[0], ka * a[1] + kb * b[1], ka * a[2] + kb * b[2])


NAN3 = (float("nan"),) * 3


# ---------- elementet e orbites ----------
def elements(r, v, mu):
    """Kthen (a, e, E) si ne pykep: a gjithmone pozitiv, E = anomalia e ekscentrike
    (ose ekuivalenti per hiperbole). NaN per orbita te degjeneruara."""
    try:
        h = cross(r, v)
        p = dot(h, h) / mu
        R = norm(r)
        vh = cross(v, h)
        ev = (vh[0] / mu - r[0] / R, vh[1] / mu - r[1] / R, vh[2] / mu - r[2] / R)
        e = norm(ev)
        a = abs(p / (1 - e * e))
        c = dot(ev, r) / e / R
        nu = math.acos(c) if -1 <= c <= 1 else float("nan")   # anomalia e vertete
        if dot(r, v) < 0:
            nu = 2 * math.pi - nu
        k = math.sqrt(abs((1 - e) / (1 + e)))
        E = 2 * math.atan(k * math.tan(nu / 2))
        return a, e, E
    except (ZeroDivisionError, ValueError):
        return float("nan"), float("nan"), float("nan")


# ---------- propagimi ne orbite (ekuacioni i Keplerit) ----------
def propagate(r0, v0, t, mu):
    """Pozita dhe shpejtesia pas t sekondash. Ngre RuntimeError nese deshton."""
    try:
        R = norm(r0)
        V = norm(v0)
        a = -mu / (V * V - 2 * mu / R)          # gjysmeboshti (negativ per hiperbole)
        s0 = dot(r0, v0) / math.sqrt(mu)
        if a > 0:   # elipse: zgjidhim per ndryshimin e anomalise x
            n = math.sqrt(mu / a ** 3) * t
            f = lambda x: x - n + s0 / math.sqrt(a) * (1 - math.cos(x)) - (1 - R / a) * math.sin(x)
            x = bisect(f, n - 4, n + 4)
            c, s = math.cos(x), math.sin(x)
            G = a * s0 / math.sqrt(mu) * (1 - c) + R * math.sqrt(a / mu) * s
            r = a + (R - a) * c + s0 * math.sqrt(a) * s
            Ft = -math.sqrt(mu * a) / (r * R) * s
        else:       # hiperbole
            n = math.sqrt(-mu / a ** 3) * t
            f = lambda x: -x - n + s0 / math.sqrt(-a) * (math.cosh(x) - 1) + (1 - R / a) * math.sinh(x)
            lo, hi = -1.0, 1.0
            while f(lo) > 0:
                lo *= 2
            while f(hi) < 0:
                hi *= 2
            x = bisect(f, lo, hi)
            c, s = math.cosh(x), math.sinh(x)
            G = a * s0 / math.sqrt(mu) * (1 - c) + R * math.sqrt(-a / mu) * s
            r = a + (R - a) * c + s0 * math.sqrt(-a) * s
            Ft = -math.sqrt(-mu * a) / (r * R) * s
        F = 1 - a / R * (1 - c)
        Gt = 1 - a / r * (1 - c)
        return comb(r0, F, v0, G), comb(r0, Ft, v0, Gt)
    except (ZeroDivisionError, OverflowError, ValueError) as err:
        raise RuntimeError(err)


def bisect(f, lo, hi, iters=100):
    """Rrenja e nje funksioni rrites f ne [lo, hi] me metoden e pergjysmimit."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ---------- problemi i Lambert-it (algoritmi i Izzo, 2015) ----------
def lambert(r1, r2, tof, mu):
    """Shpejtesia ne nisje dhe ne mberritje per te shkuar nga r1 te r2 per tof sekonda
    (0 rrotullime, drejtimi prograd). Kthen NaN nese nuk ka zgjidhje."""
    try:
        c = norm(sub(r2, r1))
        n1, n2 = norm(r1), norm(r2)
        s = (n1 + n2 + c) / 2
        i1, i2 = mul(r1, 1 / n1), mul(r2, 1 / n2)
        ih = cross(i1, i2)
        ih = mul(ih, 1 / norm(ih))
        lam = math.sqrt(max(0.0, 1 - c / s))
        if ih[2] < 0:                        # kendi i transferit > 180 grade
            lam = -lam
            t1, t2 = cross(i1, ih), cross(i2, ih)
        else:
            t1, t2 = cross(ih, i1), cross(ih, i2)
        T = math.sqrt(2 * mu / s ** 3) * tof
        x = _solve_x(lam, T)
        y = math.sqrt(1 - lam * lam * (1 - x * x))
        g = math.sqrt(mu * s / 2)
        rho = (n1 - n2) / c
        sig = math.sqrt(1 - rho * rho)
        vr1 = g * ((lam * y - x) - rho * (lam * y + x)) / n1
        vr2 = -g * ((lam * y - x) + rho * (lam * y + x)) / n2
        vt = g * sig * (y + lam * x)
        return comb(i1, vr1, t1, vt / n1), comb(i2, vr2, t2, vt / n2)
    except (ZeroDivisionError, OverflowError, ValueError, RuntimeError):
        return NAN3, NAN3


def _tof(x, lam):
    """Koha (pa dimension) e transferit si funksion i variablit x."""
    y = math.sqrt(1 - lam * lam * (1 - x * x))
    if 0.77 < x < 1.18:                      # afer parabolës: seri hipergjeometrike
        eta = y - lam * x
        z = (1 - lam - x * eta) / 2
        q, term, k = 1.0, 1.0, 0
        while True:
            term *= (3 + k) * (1 + k) / (2.5 + k) * z / (k + 1)
            if q + term == q:
                break
            q += term
            k += 1
        return (eta ** 3 * 4 / 3 * q + 4 * lam * eta) / 2
    if x < 1:
        psi = math.acos(max(-1.0, min(1.0, x * y + lam * (1 - x * x))))
    else:
        psi = math.asinh((y - x * lam) * math.sqrt(x * x - 1))
    return (psi / math.sqrt(abs(1 - x * x)) - x + lam * y) / (1 - x * x)


def _solve_x(lam, T):
    """Gjen x qe _tof(x) = T me metoden e Newton-it (derivati numerik)."""
    T00 = math.acos(lam) + lam * math.sqrt(1 - lam * lam)
    T1 = 2 / 3 * (1 - lam ** 3)
    if T >= T00:
        x = -(T - T00) / (T - T00 + 4)
    elif T <= T1:
        x = T1 * (T1 - T) / (0.4 * (1 - lam ** 5) * T) + 1
    else:
        x = (T / T00) ** (math.log(2) / math.log(T1 / T00)) - 1
    for _ in range(60):
        h = 1e-7
        f = _tof(x, lam) - T
        d = (_tof(x + h, lam) - _tof(x - h, lam)) / (2 * h)
        x_new = x - f / d
        if x_new <= -1:
            x_new = (x - 1) / 2
        if abs(x_new - x) < 1e-12:
            return x_new
        x = x_new
    raise RuntimeError("Lambert nuk konvergjoi")


# ---------- fly-by ----------
def flyby(v, v_pla, rp, beta, mu):
    """Shpejtesia pas kalimit prane planetit (rp = distanca minimale, beta = kendi i planit)."""
    vr = sub(v, v_pla)
    vn = norm(vr)
    delta = 2 * math.asin(1 / (1 + rp * vn * vn / mu))      # kendi i kthimit
    i = mul(vr, 1 / vn)
    j = cross(i, v_pla)
    j = mul(j, 1 / norm(j))
    k = cross(i, j)
    cd, sd = math.cos(delta), math.sin(delta)
    cb, sb = math.cos(beta), math.sin(beta)
    return tuple(v_pla[m] + vn * (cd * i[m] + cb * sd * j[m] + sb * sd * k[m]) for m in range(3))


# ---------- planetet ----------
class Planet:
    def __init__(self, t_ref, a, e, i, W, w, M0, mu_star, mu_self):
        self.t_ref, self.a, self.e, self.i, self.W, self.w, self.M0 = t_ref, a, e, i, W, w, M0
        self.mu_star, self.mu_self = mu_star, mu_self
        self.radius = EARTH_RADIUS
        self.n = math.sqrt(mu_star / a ** 3)            # levizja mesatare [rad/s]

    def eph(self, t_days):
        """Pozita dhe shpejtesia ne daten t (dite nga 2000-01-01)."""
        M = (self.M0 + self.n * (t_days - self.t_ref) * DAY) % (2 * math.pi)
        E = M
        for _ in range(30):                              # ekuacioni i Keplerit M = E - e sin E
            E -= (E - self.e * math.sin(E) - M) / (1 - self.e * math.cos(E))
        a, e = self.a, self.e
        b = a * math.sqrt(1 - e * e)
        x, y = a * (math.cos(E) - e), b * math.sin(E)
        k = self.n / (1 - e * math.cos(E))
        vx, vy = -a * k * math.sin(E), b * k * math.cos(E)
        cW, sW, cw, sw = math.cos(self.W), math.sin(self.W), math.cos(self.w), math.sin(self.w)
        ci, si = math.cos(self.i), math.sin(self.i)
        R = ((cW * cw - sW * sw * ci, -cW * sw - sW * cw * ci),
             (sW * cw + cW * sw * ci, -sW * sw + cW * cw * ci),
             (sw * si, cw * si))
        return (tuple(R[m][0] * x + R[m][1] * y for m in range(3)),
                tuple(R[m][0] * vx + R[m][1] * vy for m in range(3)))
