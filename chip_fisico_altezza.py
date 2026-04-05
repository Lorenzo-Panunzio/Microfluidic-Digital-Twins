"""
================================================================================
CHIP MICROFLUIDICO – OTTIMIZZATORE PARAMETRI FISICI REALI (ROMBI)
================================================================================

OBIETTIVO:
  Esplorazione libera dei parametri geometrici. 
  FUNZIONE OBIETTIVO AGGIORNATA: Minimizza sia l'errore dal 90% (Delta) 
  sia la disomogeneità (CV).

VINCOLI FISICI IMPLEMENTATI:
  - Le aperture non possono fisicamente eccedere la lunghezza della pre-camera.
  - Gap di tolleranza di 30µm per evitare "muri" fluidodinamici.
  - Libertà di "chiusura sfalsata" (end_staggered) per rompere le fasce laminari.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import warnings
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from numba import njit, prange

warnings.filterwarnings('ignore')

# ================================================================================
# FUNZIONI DI INPUT INTERATTIVO
# ================================================================================

def input_with_default(prompt, default, type_func=float):
    try:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == "": return default
        return type_func(val)
    except:
        return default

def input_list(prompt, default_list, type_func=float):
    try:
        val = input(f"{prompt} {default_list}: ").strip()
        if val == "": return default_list
        return [type_func(x.strip()) for x in val.split(",")]
    except:
        return default_list

def geom_signature(g: 'ParamGeom') -> tuple:
    return (
        g.n_col_rombi, g.n_rombi_per_col, 
        round(g.W_rombo, 1), round(g.H_rombo, 1), round(g.overlap_rombi, 1),
        g.n_ap_top, g.n_ap_bottom, 
        round(g.W_apertura, 1), round(g.H_apertura, 1),
        round(g.L_distribuzione, 1), round(g.W_distribuzione, 1),
        g.end_staggered
    )

# ================================================================================
# PARAMETRI BASE DEL CHIP
# ================================================================================

@dataclass
class ParamBase:
    # --- Fisica ---
    D_farmaco: float = 5e-10       
    mu: float        = 1.2e-3      
    rho: float       = 1025.0      
    C_in: float      = 100.0       
    Q_in: float      = 1.0         
    depth: float     = 200e-6      

    # --- Parametri Flusso Pulsatile ---
    battito_freq: float  = 1.0     
    battito_amp: float   = 0.5     

    # --- Dimensioni fisse ---
    L_precam: float  = 1800.0      
    W_precam: float  = 400.0       
    L_canale_in: float = 500.0     
    W_canale_in: float = 150.0     
    L_camera: float  = 2000.0      
    W_camera: float  = 400.0       
    L_outlet: float  = 500.0       
    W_outlet: float  = 150.0       
    W_filtro: float  = 200.0       

    # --- Griglia ---
    Nx: int          = 400
    Ny: int          = 224         

    # --- Targeting ---
    target_removal: float = 0.90
    tolleranza: float     = 0.008  
    CFL: float            = 0.3

    def __post_init__(self):
        self.Q_in_m3s = self.Q_in * 1e-9 / 60.0
        A_in          = self.W_canale_in * 1e-6 * self.depth
        self.u_in     = self.Q_in_m3s / A_in

@dataclass
class ParamGeom:
    n_col_rombi: int     = 2        
    n_rombi_per_col: int = 1        
    W_rombo: float       = 200.0    
    H_rombo: float       = 160.0    
    overlap_rombi: float = 10.0     

    n_ap_top: int       = 8
    n_ap_bottom: int    = 8
    W_apertura: float   = 80.0     
    H_apertura: float   = 80.0     

    L_distribuzione: float = 600.0 
    W_distribuzione: float = 400.0 
    
    end_staggered: bool  = False  # NUOVO GRADO DI LIBERTA'

    def label(self) -> str:
        stag_mark = "+C" if self.end_staggered else ""
        return (f"nRombi={self.n_col_rombi}x{self.n_rombi_per_col}{stag_mark} "
                f"Wr={self.W_rombo:.0f} Hr={self.H_rombo:.0f} "
                f"na={self.n_ap_top+self.n_ap_bottom} "
                f"wa={self.W_apertura:.0f}μm")

# ================================================================================
# GEOMETRIA
# ================================================================================

class Geometria:
    def __init__(self, base: ParamBase, geom: ParamGeom):
        self.b  = base
        self.g  = geom
        Nx, Ny  = base.Nx, base.Ny

        self.fluido        = np.zeros((Ny, Nx), dtype=np.int32)
        self.precam        = np.zeros((Ny, Nx), dtype=np.int32)
        self.camera        = np.zeros((Ny, Nx), dtype=np.int32)
        self.filtro        = np.zeros((Ny, Nx), dtype=np.int32)
        self.ap_top        = np.zeros((Ny, Nx), dtype=np.int32)
        self.ap_bot        = np.zeros((Ny, Nx), dtype=np.int32)
        self.pillar        = np.zeros((Ny, Nx), dtype=np.int32)
        self.distribuzione = np.zeros((Ny, Nx), dtype=np.int32)
        self.outlet        = np.zeros((Ny, Nx), dtype=np.int32)
        self.canale_in     = np.zeros((Ny, Nx), dtype=np.int32)

        self.valid      = False
        self._build()

    def _build(self):
        b, g   = self.b, self.g
        Nx, Ny = b.Nx, b.Ny
        jc     = Ny // 2

        L_tot  = (b.L_canale_in + b.L_precam + g.L_distribuzione + b.L_camera + b.L_outlet)
        W_tot  = b.W_precam + 2 * b.W_filtro + 100.0
        self.L_tot = L_tot
        self.W_tot = W_tot

        dx_um  = L_tot / Nx
        dy_um  = W_tot / Ny
        self.dx = dx_um * 1e-6
        self.dy = dy_um * 1e-6

        self.i0_precam = int(b.L_canale_in   / dx_um)
        self.i1_precam = self.i0_precam + int(b.L_precam        / dx_um)
        self.i1_dist   = self.i1_precam + int(g.L_distribuzione / dx_um)
        self.i1_cam    = self.i1_dist   + int(b.L_camera        / dx_um)

        hp = int((b.W_precam        / dy_um) / 2)
        hc = int((b.W_camera        / dy_um) / 2)
        hd = int((g.W_distribuzione / dy_um) / 2)
        ho = int((b.W_outlet        / dy_um) / 2)
        hi = max(2, int((b.W_canale_in / dy_um) / 2))

        for i in range(self.i0_precam):
            prog   = i / max(1, self.i0_precam)
            j_half = int(hi + (hp - hi) * prog)
            self.canale_in[max(0, jc-j_half):min(Ny, jc+j_half), i] = 1
        self.fluido |= self.canale_in

        j_pb, j_pt = jc - hp, jc + hp
        self.j_pb, self.j_pt = j_pb, j_pt
        self.precam[j_pb:j_pt, self.i0_precam:self.i1_precam] = 1
        self.fluido |= self.precam

        if g.n_col_rombi > 0 and g.n_rombi_per_col > 0:
            sx  = (self.i1_precam - self.i0_precam) // (g.n_col_rombi + 1)
            
            # 1. Colonne Principali
            for col in range(g.n_col_rombi):
                ic = self.i0_precam + sx * (col + 1)
                total_h = g.n_rombi_per_col * g.H_rombo - (g.n_rombi_per_col - 1) * g.overlap_rombi
                start_y = (b.W_precam - total_h) / 2.0 + g.H_rombo / 2.0
                
                for row in range(g.n_rombi_per_col):
                    if g.n_rombi_per_col == 1: 
                        cy_um = b.W_precam / 2.0
                    else: 
                        cy_um = start_y + row * (g.H_rombo - g.overlap_rombi)
                        
                    offset_y_um = cy_um - (b.W_precam / 2.0)
                    cy_idx = jc + int(offset_y_um / dy_um)
                    
                    W_px = g.W_rombo / dx_um
                    H_px = g.H_rombo / dy_um
                    
                    for j in range(max(0, cy_idx - int(H_px)), min(Ny, cy_idx + int(H_px) + 1)):
                        for i in range(max(0, ic - int(W_px)), min(Nx, ic + int(W_px) + 1)):
                            dx_norm = abs(i - ic) / max(1e-5, (W_px / 2.0))
                            dy_norm = abs(j - cy_idx) / max(1e-5, (H_px / 2.0))
                            if dx_norm + dy_norm <= 1.0:
                                if j_pb < j < j_pt and self.i0_precam < i < self.i1_precam:
                                    self.pillar[j, i] = 1

            # 2. Rombi Sfalsati Centrali (Decisi dall'algoritmo)
            # Se end_staggered è True, l'algoritmo aggiunge un rombo in più alla fine
            n_stag = g.n_col_rombi if g.end_staggered else (g.n_col_rombi - 1)
            
            for col in range(n_stag):
                ic_staggered = self.i0_precam + int(sx * (col + 1.5))
                if ic_staggered >= self.i1_precam: continue # Anti-sbordamento
                
                cy_idx = jc 
                W_px = g.W_rombo / dx_um
                H_px = g.H_rombo / dy_um
                
                for j in range(max(0, cy_idx - int(H_px)), min(Ny, cy_idx + int(H_px) + 1)):
                    for i in range(max(0, ic_staggered - int(W_px)), min(Nx, ic_staggered + int(W_px) + 1)):
                        dx_norm = abs(i - ic_staggered) / max(1e-5, (W_px / 2.0))
                        dy_norm = abs(j - cy_idx) / max(1e-5, (H_px / 2.0))
                        if dx_norm + dy_norm <= 1.0:
                            if j_pb < j < j_pt and self.i0_precam < i < self.i1_precam:
                                self.pillar[j, i] = 1

        self.fluido[self.pillar == 1] = 0
        self.precam[self.pillar == 1] = 0

        w_valvola = int((g.W_distribuzione / 3.0) / dy_um)  
        spessore_v = max(2, int(20.0 / dx_um))
        for i in range(self.i1_precam, self.i1_precam + spessore_v):
            for j in range(0, Ny):
                if abs(j - jc) > w_valvola:
                    self.fluido[j, i] = 0
                    self.distribuzione[j, i] = 0
                    self.precam[j, i] = 0

        aw = max(2, int(g.W_apertura / dx_um))
        ah = max(2, int(g.H_apertura / dy_um))
        for label, sign, sp_n in [('top', +1, g.n_ap_top), ('bot', -1, g.n_ap_bottom)]:
            sp = (self.i1_precam - self.i0_precam) // (sp_n + 1)
            for a in range(sp_n):
                ic = self.i0_precam + (a + 1) * sp
                x0 = max(self.i0_precam, ic - aw // 2)
                x1 = min(self.i1_precam, ic + aw // 2)
                if sign == +1:
                    y0, y1 = j_pt, min(Ny - 1, j_pt + ah)
                    if y1 > y0 and x1 > x0:
                        self.ap_top[y0:y1, x0:x1] = 1
                        self.fluido[y0:y1, x0:x1] = 1
                else:
                    y1, y0 = j_pb, max(1, j_pb - ah)
                    if y1 > y0 and x1 > x0:
                        self.ap_bot[y0:y1, x0:x1] = 1
                        self.fluido[y0:y1, x0:x1] = 1

        fh = int(b.W_filtro / dy_um)
        for i in range(self.i0_precam, self.i1_precam):
            self.filtro[j_pt:min(Ny-1, j_pt+fh), i] = 1
            self.filtro[max(1, j_pb-fh):j_pb, i]     = 1

        self.distribuzione[jc-hd:jc+hd, self.i1_precam:self.i1_dist] = 1
        self.fluido |= self.distribuzione
        self.camera[jc-hc:jc+hc, self.i1_dist:self.i1_cam] = 1
        self.fluido |= self.camera
        self.outlet[jc-ho:jc+ho, self.i1_cam:Nx] = 1
        self.fluido |= self.outlet

        labeled, n = ndimage.label(self.fluido)
        if n > 1:
            sizes   = ndimage.sum(self.fluido, labeled, range(1, n+1))
            largest = np.argmax(sizes) + 1
            mask    = (labeled == largest).astype(np.int32)
            for arr in [self.precam, self.camera, self.distribuzione,
                        self.outlet, self.canale_in, self.ap_top, self.ap_bot]:
                arr &= mask
            self.fluido = mask

        self.n_ap_cells_top = int(np.sum(self.ap_top))
        self.n_ap_cells_bot = int(np.sum(self.ap_bot))
        self.valid = (self.n_ap_cells_top > 0 or self.n_ap_cells_bot > 0)

# ================================================================================
# CAMPO DI VELOCITÀ E SOLVER (Invariati)
# ================================================================================

@njit(fastmath=True, cache=True)
def solve_laplace_velocity(fluido, ap_top, ap_bot, Nx, Ny, dx, dy, u_in, Q_snk_m3s, depth, max_iter=15000):
    phi = np.zeros((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        for i in range(Nx):
            if fluido[j, i]: phi[j, i] = u_in * (Nx - 1 - i) * dx
                
    N_ap = 0
    for j in range(Ny):
        for i in range(Nx):
            if ap_top[j, i] == 1 or ap_bot[j, i] == 1: N_ap += 1
                
    S_ap = Q_snk_m3s / (N_ap * dx * dy * depth) if N_ap > 0 else 0.0
    w, dx2, dy2 = 1.90, dx*dx, dy*dy
    den = 2.0/dx2 + 2.0/dy2
    
    for it in range(max_iter):
        max_err = 0.0
        for j in range(Ny):
            for i in range(Nx):
                if fluido[j, i] == 0 or i == Nx - 1: continue 
                
                phi_old = phi[j, i]
                phi_W = phi[j, i-1] if i > 0 and fluido[j, i-1] else (phi[j, i] + u_in * dx if i == 0 else phi[j, i])
                phi_E = phi[j, i+1] if i < Nx - 1 and fluido[j, i+1] else phi[j, i]
                phi_S = phi[j-1, i] if j > 0 and fluido[j-1, i] else phi[j, i]
                phi_N = phi[j+1, i] if j < Ny - 1 and fluido[j+1, i] else phi[j, i]
                
                S = S_ap if (ap_top[j, i] == 1 or ap_bot[j, i] == 1) else 0.0
                phi_new = ( (phi_E + phi_W)/dx2 + (phi_N + phi_S)/dy2 - S ) / den
                phi_sor = phi_old + w * (phi_new - phi_old)
                phi[j, i] = phi_sor
                
                err = abs(phi_sor - phi_old)
                if err > max_err: max_err = err
        
        if max_err < 1e-13: break
            
    u, v = np.zeros((Ny, Nx), dtype=np.float64), np.zeros((Ny, Nx), dtype=np.float64)
    for j in range(Ny):
        for i in range(Nx):
            if fluido[j, i]:
                if i == 0: u[j, i] = u_in
                elif i == Nx - 1: u[j, i] = -(phi[j, i] - phi[j, i-1]) / dx
                else:
                    if fluido[j, i+1] and fluido[j, i-1]: u[j, i] = -(phi[j, i+1] - phi[j, i-1]) / (2*dx)
                    elif fluido[j, i+1]: u[j, i] = -(phi[j, i+1] - phi[j, i]) / dx
                    elif fluido[j, i-1]: u[j, i] = -(phi[j, i] - phi[j, i-1]) / dx
                        
                if j > 0 and j < Ny - 1:
                    if fluido[j+1, i] and fluido[j-1, i]: v[j, i] = -(phi[j+1, i] - phi[j-1, i]) / (2*dy)
                    elif fluido[j+1, i]: v[j, i] = -(phi[j+1, i] - phi[j, i]) / dy
                    elif fluido[j-1, i]: v[j, i] = -(phi[j, i] - phi[j-1, i]) / dy
    return u, v

def build_velocity_fisico(geo: Geometria) -> Tuple[np.ndarray, np.ndarray, float]:
    b, g  = geo.b, geo.g
    Q_in, Q_snk = b.Q_in_m3s, b.Q_in_m3s * b.target_removal
    v_ap = Q_snk / max((g.n_ap_top + g.n_ap_bottom) * g.W_apertura * 1e-6 * b.depth, 1e-30)
    u_in_mean = Q_in / (b.W_canale_in * 1e-6 * b.depth)
    u, v = solve_laplace_velocity(geo.fluido, geo.ap_top, geo.ap_bot, b.Nx, b.Ny, geo.dx, geo.dy, u_in_mean, Q_snk, b.depth)
    return u, v, v_ap

@njit(parallel=True, fastmath=True, cache=True)
def _solver_fisico(C, u, v, fluido, precam, ap_top, ap_bot, D, dx, dy, dt, Nt, C_in, v_ap_base, save_ev, i1_precam, battito_amp, battito_freq):
    Ny, Nx, inv_dx, inv_dy = C.shape[0], C.shape[1], 1.0/dx, 1.0/dy
    inv_dx2, inv_dy2 = inv_dx*inv_dx, inv_dy*inv_dy
    
    for step in range(Nt):
        Cold = C.copy()
        pulse_factor = max(0.0, 1.0 + battito_amp * np.sin(2.0 * np.pi * battito_freq * (step * dt)))
        k_ap_t = abs(v_ap_base * pulse_factor) / dy

        for j in prange(1, Ny - 1):
            for i in range(1, Nx - 1):
                if fluido[j, i] == 0: continue
                c = Cold[j, i]
                uij, vij = u[j, i] * pulse_factor, v[j, i] * pulse_factor
                
                dcdx = (c - (Cold[j, i-1] if fluido[j, i-1] else c)) * inv_dx if uij > 0.0 else ((Cold[j, i+1] if fluido[j, i+1] else c) - c) * inv_dx
                dcdy = (c - (Cold[j-1, i] if fluido[j-1, i] else c)) * inv_dy if vij > 0.0 else ((Cold[j+1, i] if fluido[j+1, i] else c) - c) * inv_dy

                ce = Cold[j, i+1] if fluido[j, i+1] else c
                cw = Cold[j, i-1] if fluido[j, i-1] else c
                cn = Cold[j+1, i] if fluido[j+1, i] else c
                cs = Cold[j-1, i] if fluido[j-1, i] else c
                lap = (ce - 2.0*c + cw)*inv_dx2 + (cn - 2.0*c + cs)*inv_dy2
                sink = -k_ap_t * c if (ap_top[j, i] == 1 or ap_bot[j, i] == 1) else 0.0

                Cn = c + dt * (D * lap - uij*dcdx - vij*dcdy + sink)
                if Cn < 0.0: Cn = 0.0
                if Cn > C_in: Cn = C_in
                C[j, i] = Cn

        for j in range(Ny):
            if fluido[j, 0]: C[j, 0] = C_in
            if fluido[j, Nx-1]: C[j, Nx-1] = C[j, Nx-2]
    return C

def run_fisico(C, u, v, geo: Geometria, base: ParamBase, v_ap: float, dt: float, Nt: int, save_ev: int = 500):
    return _solver_fisico(np.asarray(C, dtype=np.float64), np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64),
                          np.asarray(geo.fluido, dtype=np.int32), np.asarray(geo.precam, dtype=np.int32), np.asarray(geo.ap_top, dtype=np.int32),
                          np.asarray(geo.ap_bot, dtype=np.int32), float(base.D_farmaco), float(geo.dx), float(geo.dy), float(dt), int(Nt), 
                          float(base.C_in), float(v_ap), int(save_ev), int(geo.i1_precam), float(base.battito_amp), float(base.battito_freq))

# ================================================================================
# FUNZIONI DI RICERCA CON NUOVO FITNESS SCORE (Delta + CV)
# ================================================================================

def valuta_geometria(base: ParamBase, g: ParamGeom, verbose: bool = False) -> Dict:
    result = {'geom': g, 'valid': False, 'removal': 0.0, 'C_out': 0.0, 'delta': 100.0, 'CV': 0.0, 'score': 999.0}
    try:
        geo = Geometria(base, g)
        if not geo.valid: return result

        u, v, v_ap = build_velocity_fisico(geo)
        uv_max = max(max(np.max(np.abs(u)), 1e-12), max(np.max(np.abs(v)), 1e-12))
        dt = min(base.CFL * min(geo.dx, geo.dy) / uv_max, base.CFL * min(geo.dx, geo.dy)**2 / (2.0 * base.D_farmaco))

        cells_pre = geo.precam == 1
        tau_res = base.L_precam * 1e-6 / max(np.mean(u[cells_pre]) if cells_pre.any() else uv_max, 1e-12)
        Nt = max(min(int(12 * tau_res / dt), 150_000), 4_000)

        C = run_fisico(np.zeros((base.Ny, base.Nx), dtype=np.float64), u, v, geo, base, v_ap, dt, Nt, save_ev=max(Nt // 20, 100))

        mask_col = geo.fluido[:, geo.i1_precam] == 1
        C_out = float(np.mean(C[:, geo.i1_precam][mask_col])) if mask_col.any() else 0.0
        removal = (1.0 - C_out / base.C_in) * 100.0
        delta = abs(removal - base.target_removal * 100.0)
        
        mask_cam = (geo.camera == 1) & (geo.fluido == 1)
        C_cam = C[mask_cam]
        C_mean = float(np.mean(C_cam)) if C_cam.size else 0.0
        C_std = float(np.std(C_cam)) if C_cam.size else 0.0
        CV = (C_std / C_mean * 100) if C_mean > 0 else 0.0
        
        # NUOVA FUNZIONE OBIETTIVO: Penalizza pesantemente i CV alti! (Ogni 5% di CV "costa" 1% di Delta)
        score = delta + (CV * 0.2)

        result.update({'valid': True, 'removal': removal, 'C_out': C_out, 'delta': delta, 'CV': CV, 'score': score, 'v_ap': v_ap, 'tau_res': tau_res, 'C_field': C, 'geo_obj': geo, 'u': u, 'v': v})
        if verbose: print(f"  {'✓' if delta < base.tolleranza * 100 else ' '} {g.label()} → removal={removal:.1f}%  (Δ={delta:.1f}%) CV={CV:.1f}% [Score: {score:.1f}]")
    except Exception as e:
        if verbose: print(f"  ✗ {g.label()} → ERRORE: {e}")
    return result

def grid_search(base: ParamBase, n_ap_values: List[int], W_ap_values: List[float],
                n_col_values: List[int], n_rpc_values: List[int], W_rombo_values: List[float], H_rombo_values: List[float],
                end_stag_values: List[bool], L_distr: float, W_distr: float, seen: Optional[Dict] = None, verbose: bool = True) -> List[Dict]:
    if seen is None: seen = {}
    print(f"\n{'='*70}\nFASE 1 – GRID SEARCH GROSSOLANA (Ottimizzazione Combinata Delta+CV)\n{'='*70}")
    
    combos = list(itertools.product(n_ap_values, W_ap_values, n_col_values, n_rpc_values, W_rombo_values, H_rombo_values, end_stag_values))
    results = []

    for idx, (n_ap, W_ap, n_col, n_rpc, W_r, H_r, end_stag) in enumerate(combos):
        if (n_ap * W_ap + (n_ap + 1) * 10.0) > base.L_precam: continue
        if (n_rpc * H_r - (n_rpc - 1) * 10.0) > (base.W_precam - 30.0): continue
        if (n_col * W_r * 1.5) > base.L_precam: continue

        g = ParamGeom(n_col_rombi=n_col, n_rombi_per_col=n_rpc, W_rombo=W_r, H_rombo=H_r, 
                      n_ap_top=n_ap, n_ap_bottom=n_ap, W_apertura=W_ap, H_apertura=W_ap, 
                      L_distribuzione=L_distr, W_distribuzione=W_distr, end_staggered=end_stag)
        sig = geom_signature(g)
        if sig in seen: results.append(seen[sig]); continue
            
        r = valuta_geometria(base, g, verbose=verbose)
        seen[sig] = r
        results.append(r)
    return results

def raffinamento_locale(base: ParamBase, top_results: List[Dict], seen: Optional[Dict] = None, verbose: bool = True) -> List[Dict]:
    if seen is None: seen = {}
    print(f"\n{'='*70}\nFASE 2 – RAFFINAMENTO LOCALE\n{'='*70}")
    all_refined = []
    for rank, r in enumerate(top_results):
        g0 = r['geom']
        print(f"\nCandidato {rank+1}: {g0.label()} → removal={r['removal']:.1f}%, CV={r['CV']:.1f}%")
        combos = list(itertools.product(
            [max(2, g0.n_ap_top - 2), g0.n_ap_top, g0.n_ap_top + 2],
            [max(50, g0.W_apertura - 20), g0.W_apertura, g0.W_apertura + 20],
            [max(100, g0.H_rombo - 15), g0.H_rombo, g0.H_rombo + 15]
        ))
        for (n_ap, W_ap, H_r) in combos:
            if (n_ap * W_ap + (n_ap + 1) * 10.0) > base.L_precam: continue
            if (g0.n_rombi_per_col * H_r - (g0.n_rombi_per_col - 1) * 10.0) > (base.W_precam - 30.0): continue
                
            g_ref = ParamGeom(n_col_rombi=g0.n_col_rombi, n_rombi_per_col=g0.n_rombi_per_col,
                              W_rombo=g0.W_rombo, H_rombo=H_r, n_ap_top=n_ap, n_ap_bottom=n_ap,
                              W_apertura=W_ap, H_apertura=W_ap, L_distribuzione=g0.L_distribuzione, 
                              W_distribuzione=g0.W_distribuzione, end_staggered=g0.end_staggered)
            sig = geom_signature(g_ref)
            if sig not in seen:
                r_ref = valuta_geometria(base, g_ref, verbose=verbose)
                seen[sig] = r_ref
                all_refined.append(r_ref)
    return all_refined

def stampa_tabella(results: List[Dict], base: ParamBase, top_n: int = 15):
    # ORA ORDINIAMO PER IL NUOVO SCORE (Delta + CV penalizzato)
    valid = sorted([r for r in results if r['valid']], key=lambda r: r['score'])
    print(f"\n{'='*105}\nCLASSIFICA CONFIGURAZIONI ROMBI (top {min(top_n, len(valid))})\nTarget: {base.target_removal*100:.0f}% rimozione | Ordinato per Score (Delta + CV Penalty)\n{'='*105}")
    print(f"{'#':>3} {'Score':>7} | {'removal':>7} {'C_out':>6} {'Δ%':>5} | {'CV%':>5} | {'nCol':>4} {'nRom':>4} {'W_r':>5} {'H_r':>5} | {'n_ap':>4} {'W_ap':>5} | STATO\n" + "-"*105)
    for i, r in enumerate(valid[:top_n]):
        g, ok = r['geom'], "✓✓✓" if r['delta'] < base.tolleranza * 100 else ("≈  " if r['delta'] < 5 else "   ")
        stag_mark = "+C" if g.end_staggered else "  "
        print(f"{i+1:>3} {r['score']:>7.1f} | {r['removal']:>6.1f}% {r['C_out']:>6.2f} {r['delta']:>5.1f} | {r['CV']:>5.1f} | "
              f"{g.n_col_rombi:>4} {str(g.n_rombi_per_col)+stag_mark:>4} {g.W_rombo:>5.0f} {g.H_rombo:>5.0f} | {g.n_ap_top+g.n_ap_bottom:>4} {g.W_apertura:>5.0f} | {ok}")
    return valid

# ================================================================================
# GRAFICI 
# ================================================================================

def plot_migliore(r: Dict, base: ParamBase, filename: str = 'chip_ottimizzatore_rombi.png'):
    geo, C, u, v, g = r['geo_obj'], r['C_field'], r['u'], r['v'], r['geom']
    C_plot = C.astype(float).copy()
    C_plot[geo.fluido == 0] = np.nan

    mask_col = geo.fluido[:, geo.i1_precam] == 1
    C_out = float(np.mean(C[:, geo.i1_precam][mask_col])) if mask_col.any() else 0.0
    removal, mask_cam = (1.0 - C_out / base.C_in) * 100.0, (geo.camera == 1) & (geo.fluido == 1)
    C_cam = C[mask_cam]
    C_mean, C_std = float(np.mean(C_cam)) if C_cam.size else 0.0, float(np.std(C_cam)) if C_cam.size else 0.0
    CV = (C_std / C_mean * 100) if C_mean > 0 else 0.0

    extent, x_fine_precam, x_fine_distr = [0, geo.L_tot, 0, geo.W_tot], base.L_canale_in + base.L_precam, base.L_canale_in + base.L_precam + g.L_distribuzione
    fig = plt.figure(figsize=(22, 12)); gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, :2])
    im1 = ax1.imshow(C_plot, origin='lower', extent=extent, aspect='auto', cmap='viridis', vmin=0, vmax=base.C_in)
    plt.colorbar(im1, ax=ax1, label='C [%]')
    py, px = np.where(geo.pillar == 1)
    if len(px): ax1.scatter(px*geo.dx*1e6, py*geo.dy*1e6, c='white', s=2, alpha=0.4, label='Barriera Rombo')
    ay_t, ax_t = np.where(geo.ap_top == 1); ay_b, ax_b = np.where(geo.ap_bot == 1)
    if len(ax_t): ax1.scatter(ax_t*geo.dx*1e6, ay_t*geo.dy*1e6, c='cyan', s=3, alpha=0.6, label='Ap. top')
    if len(ax_b): ax1.scatter(ax_b*geo.dx*1e6, ay_b*geo.dy*1e6, c='magenta', s=3, alpha=0.6, label='Ap. bot')
    ax1.axvline(x_fine_precam, color='cyan', ls='--', lw=1.5, label='Fine pre-cam'); ax1.axvline(x_fine_distr, color='orange', ls='--', lw=1.5, label='Fine distr.')
    ax1.set_title(f'Concentrazione – Modello Rombi (no k_sink)\n{g.label()}'); ax1.legend(fontsize=6); ax1.set_xlabel('L [μm]')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(C_plot, origin='lower', extent=extent, aspect='auto', cmap='plasma', vmin=0, vmax=base.C_in)
    ax2.set_xlim([base.L_canale_in - 20, base.L_canale_in + base.L_precam + 20]); ax2.set_ylim([0, geo.W_tot]); ax2.set_title('ZOOM: Pre-camera')

    ax3 = fig.add_subplot(gs[0, 3])
    ax3.imshow(C_plot, origin='lower', extent=extent, aspect='auto', cmap='coolwarm', vmin=0, vmax=max(C_mean * 2.5, base.C_in * 0.15))
    ax3.set_xlim([x_fine_distr - 20, geo.L_tot]); ax3.set_ylim([0, geo.W_tot])
    ax3.text(0.97, 0.97, f'C̄={C_mean:.1f}%\nCV={CV:.1f}%\nTarget=10%', transform=ax3.transAxes, ha='right', va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.9)); ax3.set_title('ZOOM: Camera coltura')

    ax4 = fig.add_subplot(gs[1, :2])
    xp, Cp = np.linspace(0, geo.L_tot, base.Nx), np.array([np.mean(C[:, i][geo.fluido[:, i] == 1]) if geo.fluido[:, i].any() else np.nan for i in range(base.Nx)])
    ax4.fill_between(xp, np.nan_to_num(Cp), alpha=0.2, color='steelblue'); ax4.plot(xp, Cp, 'steelblue', lw=2, label='C(x) medio')
    ax4.axhline(base.C_in*(1-base.target_removal), color='lime', ls='--', lw=2, label=f'Target {(1-base.target_removal)*100:.0f}%')
    ax4.axvline(x_fine_precam, color='cyan', ls='--', lw=1.2, label='Fine pre-cam'); ax4.axvline(x_fine_distr, color='orange', ls='--', lw=1.2, label='Fine distr.')
    ax4.set_xlabel('L [μm]'); ax4.set_ylabel('C [%]'); ax4.set_title('Profilo longitudinale'); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

    ax5 = fig.add_subplot(gs[1, 2])
    spd = np.sqrt(u**2 + v**2); spd[geo.fluido == 0] = np.nan
    # Calcola il minimo e massimo reale per scalare i colori dinamicamente
    v_min_log = np.log10(max(np.nanmin(spd), 1e-6))
    v_max_log = np.log10(max(np.nanmax(spd), 1e-2))
    im5 = ax5.imshow(np.log10(spd + 1e-12), origin='lower', extent=extent, aspect='auto', cmap='inferno', vmin=v_min_log, vmax=v_max_log)
    plt.colorbar(im5, ax=ax5, label='log₁₀|v| [m/s]'); ax5.set_title('Campo velocità (Risolto con Laplace)')

    ax6 = fig.add_subplot(gs[1, 3])
    cs, ce = geo.i1_dist, geo.i1_cam
    for pos, col6 in zip([0.1, 0.35, 0.65, 0.9], plt.cm.plasma(np.linspace(0, 0.85, 4))):
        idx = int(cs + pos * (ce - cs))
        if idx < base.Nx: m = geo.fluido[:, idx] == 1; ax6.plot(C[:, idx][m], np.arange(base.Ny)[m] * geo.dy * 1e6, color=col6, lw=2, label=f'{int(pos*100)}%') if m.any() else None
    ax6.axvline(C_mean, color='red', ls='--', alpha=0.7, label=f'C̄={C_mean:.1f}%'); ax6.axvline(base.C_in*(1-base.target_removal), color='lime', ls=':', lw=2, label='Target 10%')
    ax6.set_xlabel('C [%]'); ax6.set_ylabel('Y [μm]'); ax6.set_title('Profili trasversali camera'); ax6.legend(title='Posizione', fontsize=7); ax6.grid(alpha=0.3)

    ax7 = fig.add_subplot(gs[2, :]); ax7.axis('off')
    frac_Q = (r['v_ap'] * (g.n_ap_top + g.n_ap_bottom) * g.W_apertura * 1e-6 * base.depth) / base.Q_in_m3s * 100
    ok_str = "✓✓✓ TARGET RAGGIUNTO" if r['delta'] < base.tolleranza * 100 else f"≈  Vicino ({r['delta']:.1f}% da target)" if r['delta'] < 5 else f"⚠  Distante dal target ({r['delta']:.1f}%)"
    stag_str = " + 1 Rombo Centrale Finale" if g.end_staggered else ""
    info = (f"{'─'*75}\n  MODELLO FISICO (ROMBI) – NO k_sink – PARAMETRI OTTIMALI\n{'─'*75}\n"
            f"  GEOMETRIA\n    Pre-camera    : {base.L_precam:.0f} × {base.W_precam:.0f} μm²\n"
            f"    Barriere      : {g.n_col_rombi} colonna/e | Rombi per colonna: {g.n_rombi_per_col}{stag_str}\n"
            f"    Dimens. Rombo : W = {g.W_rombo:.0f} μm | H = {g.H_rombo:.0f} μm\n"
            f"    Aperture      : {g.n_ap_top} top + {g.n_ap_bottom} bottom | {g.W_apertura:.0f} × {g.H_apertura:.0f} μm²\n"
            f"    Distribuzione : {g.L_distribuzione:.0f} × {g.W_distribuzione:.0f} μm²\n{'─'*75}\n"
            f"  FISICA (Campo Reale di Pressione)\n    Q_in          = {base.Q_in:.2f} μL/min\n"
            f"    v_ap (fisico) = {r['v_ap']*1e6:.1f} μm/s\n    Q_rimosso est.= {frac_Q:.1f}% di Q_in\n"
            f"    τ_residenza   = {r['tau_res']:.1f} s\n{'─'*75}\n"
            f"  RISULTATI (Score = Δ + CV/5: {r['score']:.1f})\n    C_out precam  = {C_out:.2f}%  →  Rimozione = {removal:.1f}%\n"
            f"    C_camera (̄)  = {C_mean:.2f}%  (CV = {CV:.1f}%)\n    STATO         : {ok_str}\n{'─'*75}")
    ax7.text(0.5, 0.5, info, transform=ax7.transAxes, ha='center', va='center', fontfamily='monospace', fontsize=8.5, bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.95))
    plt.suptitle('Chip Microfluidico – Geometria Stampabile a Rombi', fontsize=14, fontweight='bold', y=0.998)
    plt.savefig(filename, dpi=220, bbox_inches='tight'); print(f"\nFigura salvata: {filename}"); plt.show()

# ================================================================================
# MAIN
# ================================================================================

def main():
    print("=" * 80)
    print("CHIP MICROFLUIDICO – OTTIMIZZATORE STAMPABILE (ROMBI)")
    print("Fisica potenziata: Solutore Flusso Laplace + Scelta Libera Chiusura")
    print("=" * 80)
    
    print("\n[Parametri Fisici]")
    D_farmaco = input_with_default("Diffusività farmaco (m²/s)", 5e-10)
    mu, rho = input_with_default("Viscosità (Pa·s)", 1.2e-3), input_with_default("Densità (kg/m³)", 1025.0)
    C_in, Q_in = input_with_default("Concentrazione ingresso (%)", 100.0), input_with_default("Portata Q_in (μL/min)", 1.0)
    depth = input_with_default("Profondità canale (m)", 200e-6)
    
    print("\n[Parametri Battito Cardiaco]")
    battito_freq, battito_amp = input_with_default("Frequenza (Hz)", 1.0), input_with_default("Ampiezza pulsazione (0-1)", 0.5)
    
    print("\n[Dimensioni Fisse Chip - μm]")
    L_precam, W_precam = input_with_default("Lunghezza pre-camera", 1800.0), input_with_default("Larghezza pre-camera", 400.0)
    L_canale_in, W_canale_in = input_with_default("Lunghezza canale ingresso", 500.0), input_with_default("Larghezza canale ingresso", 150.0)
    L_camera, W_camera = input_with_default("Lunghezza camera", 2000.0), input_with_default("Larghezza camera", 400.0)
    L_outlet, W_outlet = input_with_default("Lunghezza outlet", 500.0), input_with_default("Larghezza outlet", 150.0)
    W_filtro = input_with_default("Larghezza filtro", 200.0)
    
    print("\n[Parametri Geometrici Canale Distribuzione - μm]")
    L_distribuzione, W_distribuzione = input_with_default("Lunghezza canale distribuzione", 600.0), input_with_default("Larghezza canale distribuzione", 400.0)
    
    print("\n[Griglia e Target]")
    Nx, Ny = int(input_with_default("N punti griglia X", 400, int)), int(input_with_default("N punti griglia Y", 224, int))
    target_removal, tolleranza = input_with_default("Target rimozione (0-1)", 0.90), input_with_default("Tolleranza target (0-1)", 0.008)
    CFL = input_with_default("Numero CFL", 0.3)
    
    base = ParamBase(D_farmaco=D_farmaco, mu=mu, rho=rho, C_in=C_in, Q_in=Q_in, depth=depth,
                     battito_freq=battito_freq, battito_amp=battito_amp, L_precam=L_precam, W_precam=W_precam, 
                     L_canale_in=L_canale_in, W_canale_in=W_canale_in, L_camera=L_camera, W_camera=W_camera,
                     L_outlet=L_outlet, W_outlet=W_outlet, W_filtro=W_filtro, Nx=Nx, Ny=Ny, target_removal=target_removal, tolleranza=tolleranza, CFL=CFL)
    
    print("\n[Range Ottimizzazione Barriere a Rombo (Esplorazione Libera)]")
    n_ap_list = input_list("Numero aperture per lato (int)", [10, 12, 14, 16], int)
    W_ap_list = input_list("Larghezza aperture (μm)", [80.0, 100.0, 120.0], float)
    n_col_list = input_list("Numero colonne di barriere (int)", [1, 2, 3], int)
    n_rpc_list = input_list("Numero rombi per colonna (int)", [1, 2], int)
    W_rombo_list = input_list("Larghezza rombo/i (μm)", [150.0, 200.0, 250.0], float)
    H_rombo_list = input_list("Altezza rombo/i (μm) [max 185 per impilati]", [160.0, 175.0, 185.0], float)
    
    # L'algoritmo deciderà se conviene o meno chiudere la geometria con un rombo sfalsato centrale!
    end_stag_list = [True, False]
    
    print("\nCompilazione Numba in corso...")
    g_test = ParamGeom(L_distribuzione=L_distribuzione, W_distribuzione=W_distribuzione)
    geo_t  = Geometria(base, g_test)
    u_t, v_t, vap_t = build_velocity_fisico(geo_t)
    dt_test = base.CFL * min(geo_t.dx, geo_t.dy) / max(np.max(np.abs(u_t)), 1e-12)
    run_fisico(np.zeros((base.Ny, base.Nx)), u_t, v_t, geo_t, base, vap_t, dt_test, 5, save_ev=5)
    print("Completata!\n")

    seen_geometries = {}
    all_results = grid_search(base, n_ap_list, W_ap_list, n_col_list, n_rpc_list, W_rombo_list, H_rombo_list, end_stag_list, L_distribuzione, W_distribuzione, seen=seen_geometries)
    valid_1 = sorted([r for r in all_results if r['valid']], key=lambda r: r['score']) # ORDINIAMO PER SCORE
    
    if valid_1:
        top5 = valid_1[:5]
        print(f"\n{'='*70}\nTop-5 candidati Grid Search:\n{'='*70}")
        for i, r in enumerate(top5): print(f"  {i+1}. {r['geom'].label()} → removal={r['removal']:.1f}% CV={r['CV']:.1f}% [Score: {r['score']:.1f}]")
        refined = raffinamento_locale(base, top5, seen=seen_geometries)
        valid = stampa_tabella(all_results + refined, base, top_n=10)
        if valid:
            best = valid[0]
            stag_str = " + 1 centrale finale" if best['geom'].end_staggered else ""
            print(f"\n>>> OTTIMALE TROVATO: {best['geom'].n_col_rombi} barriere da {best['geom'].n_rombi_per_col} rombi{stag_str}, {best['geom'].W_rombo}x{best['geom'].H_rombo}µm")
            print(f">>> Rimozione: {best['removal']:.2f}% (Delta: {best['delta']:.2f}%)")
            print(f">>> CV nella camera: {best['CV']:.2f}%")
            if 'C_field' in best: plot_migliore(best, base)
    else:
        print("\nNessuna geometria valida trovata.")

if __name__ == "__main__":
    main()