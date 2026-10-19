"""
Problemi Trappist Tour (ESA SpOC 2022): te dhenat e planeteve dhe funksioni fitness.

Vektori x (34 vlera):
  x[0:3]          u, v, T0                hyrja ne sistem (drejtimi + kohezgjatja)
  x[3:27]         6 x [beta, rp, eta, T]  per cdo hap planet -> planet
  x[27:34]        s0..s6                  renditja e planeteve (0..6 = b..h)

fitness(x) -> (DV, T, eq, ineq) njesoj si kodi zyrtar i ESA-s.

Funksioni eshte ndare ne HAPA (legs): kostoja e hapit k varet vetem nga
gjendja e anijes kur arrin te planeti k dhe nga 4 parametrat e hapit.
Kete e shfrytezon algoritmi (solver.py).
"""
import math
from physics import AU, DAY, Planet, lambert, propagate, flyby, elements, norm, sub

# ---------- konstantet (nga kodi zyrtar) ----------
G = 6.67430e-11
MU = G * 8.98266512e-2 * 1.989e30          # parametri gravitacional i yllit TRAPPIST-1
SAFE_RADIUS = 10 * 83484000.0              # distanca minimale nga ylli [m]
T_START = 7000.0                           # 2019-03-02 ne dite nga 2000-01-01
R_START = 10 * AU                          # anija nis ne sferen 10 AU
V_START = 1e4                              # me shpejtesi 10 km/s
NAMES = "bcdefgh"
INF = float("inf")

_mass = [1.36004499, 1.29688971, 0.38588417, 0.688627613, 1.03510927, 1.31679654, 0.320988718]
_a = [0.01154035, 0.0158073, 0.02226718, 0.02926548, 0.0385054, 0.04683828, 0.06191385]
_e = [0.00455247, 0.00107698, 0.00624362, 0.00577129, 0.00861154, 0.00400202, 0.00378361]
_i = [1.1, 1.3, 0.5, -0.3, 0.01, 1.2, 2.3]
_W = [238.35940145, 221.02196847, 90.26523049, 164.16001523, 171.42668515, 166.63349296, 282.808809]
_w = [126.22225948, 80.78036351, 147.44525146, 315.09376191, 179.82855007, 25.31242216, 176.19682451]
_M = [33.40998384, -89.79506054, -28.38482427, 45.00185452, -145.16404128, 12.59883066, -12.12072197]
D2R = math.pi / 180
PLANETS = [Planet(T_START, _a[k] * AU, _e[k], _i[k] * D2R, _W[k] * D2R, _w[k] * D2R, _M[k] * D2R,
                  MU, G * _mass[k] * 5.9722e24) for k in range(7)]

# kufijte e variablave
LEG_LB = [0.0, 1.1, 0.001, 5.0]                 # beta, rp, eta, T
LEG_UB = [2 * math.pi, 100.0, 0.999, 2000.0]
LB = [0.0, 0.25, 5.0] + LEG_LB * 6 + [0] * 7
UB = [1.0, 0.75, 2000.0] + LEG_UB * 6 + [6] * 7


def distance_violation(r, v, t_days):
    """A i afrohet anija yllit me shume se SAFE_RADIUS gjate t_days?
    Kthen > 0 nese po (shkelje), perndryshe nje numer negativ."""
    a, e, E0 = elements(r, v, MU)
    if not (math.isfinite(a) and math.isfinite(e)):
        return 1e17
    rp = a * abs(1 - e)                           # distanca me e vogel e orbites (periapsis)
    try:
        r2, v2 = propagate(r, v, t_days * DAY, MU)
    except RuntimeError:
        return 1e17
    period = 2 * math.pi * math.sqrt(a ** 3 / MU) / DAY
    E = elements(r2, v2, MU)[2]
    # a e kaloi anija periapsis-in gjate kesaj kohe? (njesoj si kodi zyrtar)
    if t_days > period:
        passed = True
    elif E0 > 0:
        passed = 0 < E < E0
    else:
        passed = E > 0 or t_days > period / 2
    return SAFE_RADIUS - rp if passed else -1e17


class State:
    """Gjendja e anijes kur arrin te nje planet."""
    def __init__(self, planet, t, r, v_planet, v_in):
        self.planet, self.t, self.r, self.v_planet, self.v_in = planet, t, r, v_planet, v_in


def entry_leg(u, v, T0, planet):
    """Hapi hyres: nga sfera 10 AU te planeti i pare. Kthen (DV, shkelja, State)."""
    th, ph = 2 * math.pi * u, math.acos(2 * v - 1) - math.pi / 2
    r0 = (R_START * math.cos(ph) * math.cos(th), R_START * math.cos(ph) * math.sin(th), R_START * math.sin(ph))
    t = T_START + T0
    rP, vP = PLANETS[planet].eph(t)
    v1, v2 = lambert(r0, rP, T0 * DAY, MU)
    if math.isnan(v1[0]):
        return INF, INF, None
    dv = abs(norm(v1) - V_START)           # anija tashme leviz me 10 km/s
    return dv, distance_violation(r0, v1, T0), State(planet, t, rP, vP, v2)


def next_leg(st, beta, rp, eta, T, planet):
    """Fly-by te planeti aktual, udhetim i lire eta*T dite, manover (DSM),
    pastaj harku Lambert deri te planeti tjeter. Kthen (DV, shkelja, State)."""
    p = PLANETS[st.planet]
    v_out = flyby(st.v_in, st.v_planet, rp * p.radius, beta, p.mu_self)
    viol = distance_violation(st.r, v_out, eta * T)
    try:
        r, v = propagate(st.r, v_out, eta * T * DAY, MU)
    except RuntimeError:
        return INF, INF, None
    t = st.t + T
    rP, vP = PLANETS[planet].eph(t)
    v1, v2 = lambert(r, rP, (1 - eta) * T * DAY, MU)
    if math.isnan(v1[0]):
        return INF, INF, None
    viol = max(viol, distance_violation(r, v1, (1 - eta) * T))
    return norm(sub(v1, v)), viol, State(planet, t, rP, vP, v2)


def fitness(x):
    """(DV totale [m/s], koha totale [dite], eq, ineq) – eq == 0 dhe ineq <= 0 => e pranueshme."""
    seq = [int(s) for s in x[27:34]]
    eq = len(set(seq)) - 7
    T = sum(x[2:27:4])
    dv, viol, st = entry_leg(x[0], x[1], x[2], seq[0])
    for k in range(6):
        if st is None:
            return 1e10, T, eq, 1e17
        b = 3 + 4 * k
        d, vk, st = next_leg(st, x[b], x[b + 1], x[b + 2], x[b + 3], seq[k + 1])
        dv += d
        viol = max(viol, vk)
    if st is None:
        return 1e10, T, eq, 1e17
    return dv, T, eq, viol


def example():
    """Zgjidhja shembull nga ESA (DV = 2379 m/s, T = 3470 dite)."""
    return [0.829814333836995, 0.49439092971262544, 1501.6919241751443,
            1.5873030906083387, 1.512021386175137, 0.23024558713584858, 255.87994897371408,
            4.161192563429589, 8.278275659200725, 0.10265621329161562, 318.6140876998619,
            4.546432641418882, 4.078585717496654, 0.33605586368271645, 374.6287841767241,
            0.9461693914845313, 19.13985248269904, 0.4834964750829982, 343.8204904698013,
            2.1935168149524964, 32.48727508444479, 0.40957309270588427, 325.1537638293067,
            4.743610880353654, 49.95062395827792, 0.3668039400426466, 350.0377039343523,
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def describe(x):
    """Shtyp nje permbledhje te tour-it."""
    seq = [int(s) for s in x[27:34]]
    dv, T, eq, ineq = fitness(x)
    print("Renditja:", " -> ".join(NAMES[s] for s in seq))
    print(f"DV totale = {dv:.1f} m/s, koha totale = {T:.1f} dite, e pranueshme = {eq == 0 and ineq <= 0}")
