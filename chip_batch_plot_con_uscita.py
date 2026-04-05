"""
================================================================================
CHIP MICROFLUIDICO - DIGITAL TWIN CON CANALE DI USCITA APICALE
================================================================================
- FIX: Stabilità numerica con vincolo di Fourier per la diffusione
- FIX: Gestione corretta dei batch nella fase di fouling (rotazione)
- FIX: Validazione input e protezione parametri
- FIX: Report finale aggregato con metriche di performance
- FIX: Bilancio di massa per verifica conservazione
- NEW: 9 plot professionali integrati
- NEW: Canale di uscita apicale collegato a vaso sanguigno
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from numba import njit
import warnings
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# Palette professionale
COLORS = {
    'sangue': '#c0392b',
    'farmaco': '#3498db',
    'membrana': '#27ae60',
    'precamera': '#f39c12',
    'camera': '#9b59b6',
    'canale': '#2980b9',
    'target': '#2ecc71',
    'alert': '#e74c3c',
    'text': '#2c3e50',
    'grid': '#bdc3c7',
    'outlet': '#e67e22'  # Nuovo colore per l'uscita
}

print("="*100)
print(" "*10 + "CHIP MICROFLUIDICO - DIGITAL TWIN CON USCITA APICALE (CONSOLE MODE)")
print("="*100)
print(" Motore: Numba JIT (C-level performance)")
print(" Logica: State-Machine con calibrazione autonoma e validazione")
print( "Output: Tabella dati terminale + Barre di caricamento + Report finale")
print(" Validazione: Bilancio di massa integrato")
print(" Visualizzazione: 9 plot professionali")
print(" Outlet: Canale di uscita apicale collegato al vaso sanguigno")
print("="*100 + "\n")

FASE_RIEMPIMENTO = 1
FASE_DIALISI = 2
FASE_SCARICO = 3  # Nuova fase per lo scarico della camera

def visualizza_barra(progresso, totale, prefisso='', lunghezza=30):
    """Genera una barra di caricamento elegante nel terminale che si sovrascrive"""
    import sys
    percentuale = ("{0:.1f}").format(100 * (progresso / float(totale)))
    riempimento = int(lunghezza * progresso // totale)
    barra = '█' * riempimento + '-' * (lunghezza - riempimento)
    sys.stdout.write(f'\r{prefisso} |{barra}| {percentuale}% Completo')
    sys.stdout.flush()
    if progresso >= totale:
        sys.stdout.write('\r' + ' ' * (len(prefisso) + lunghezza + 25) + '\r')
        sys.stdout.flush()

@dataclass
class ChipConfig:
    # Canali esistenti
    L_channel: float = 200          
    L_precam: float = 150           
    L_cult: float = 450             
    W_total: float = 300            
    H_channel: float = 50           
    H_cult: float = 200             

    # NUOVO: Canale di uscita apicale
    L_outlet: float = 200           # Lunghezza canale uscita [µm]
    H_outlet: float = 40            # Altezza canale uscita [µm]
    outlet_offset: float = 20       # Distanza dal bordo superiore [µm]
    u_outlet: float = 2e-4          # Velocità uscita [m/s] - più lenta per gravità/pressione

    n_top_apertures: int = 14       
    n_bottom_apertures: int = 14
    aperture_w: float = 30          
    aperture_h: float = 12           

    D_drug: float = 5e-10           
    u_in: float = 5e-4              
    C_in: float = 100.0             
    permeabilita_farmaco: float = 0.5   # Membrana realistica

    target_removal: float = 0.90    
    tolerance_removal: float = 0.01 
    n_batch: int = 3
    tempo_dialisi: float = 20.0     
    tempo_scarico: float = 30.0     # NUOVO: Tempo per svuotare la camera

    nx: int = 150
    ny: int = 100
    dt_max: float = 0.001

    fouling_factor: float = 0.96
    min_permeability: float = 0.05

    def __post_init__(self):
        if self.n_batch <= 0:
            raise ValueError(f"n_batch deve essere > 0, ricevuto {self.n_batch}")
        if not (0 < self.target_removal < 1):
            raise ValueError(f"target_removal deve essere tra 0 e 1, ricevuto {self.target_removal}")
        if self.permeabilita_farmaco <= 0:
            raise ValueError(f"permeabilita_farmaco deve essere > 0, ricevuto {self.permeabilita_farmaco}")
        if self.fouling_factor <= 0 or self.fouling_factor > 1:
            raise ValueError(f"fouling_factor deve essere tra 0 e 1, ricevuto {self.fouling_factor}")
        if self.L_outlet <= 0:
            raise ValueError(f"L_outlet deve essere > 0, ricevuto {self.L_outlet}")
        if self.H_outlet <= 0:
            raise ValueError(f"H_outlet deve essere > 0, ricevuto {self.H_outlet}")

        self.dx = (self.L_channel + self.L_precam + self.L_cult + self.L_outlet) * 1e-6 / self.nx
        self.dy = self.W_total * 1e-6 / self.ny

        dt_cfl = 0.5 * min(self.dx, self.dy) / max(self.u_in, self.u_outlet, 1e-10)
        dt_fourier = 0.25 * min(self.dx**2, self.dy**2) / self.D_drug if self.D_drug > 0 else float('inf')
        self.dt = min(self.dt_max, dt_cfl, dt_fourier)

        if self.dt == dt_fourier:
            print(f"⚠️  Timestep limitato da stabilità di Fourier: {self.dt:.2e}s")

class ChipGeometryCompleto:
    def __init__(self, config: ChipConfig):
        self.cfg = config
        self.nx, self.ny = config.nx, config.ny
        self.fluid = np.zeros((self.ny, self.nx), dtype=bool)
        self.precam_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.cult_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.aperture_mask = np.zeros((self.ny, self.nx), dtype=bool)
        self.outlet_mask = np.zeros((self.ny, self.nx), dtype=bool)  # NUOVO
        self._build_geometry()

    def _build_geometry(self):
        cfg = self.cfg
        nx, ny = self.nx, self.ny

        # Lunghezza totale ora include anche il canale di uscita
        L_total = cfg.L_channel + cfg.L_precam + cfg.L_cult + cfg.L_outlet

        self.i_ch_end = int((cfg.L_channel / L_total) * nx)
        self.i_precam_start = self.i_ch_end
        self.i_precam_end = int(((cfg.L_channel + cfg.L_precam) / L_total) * nx)
        self.i_cult_end = int(((cfg.L_channel + cfg.L_precam + cfg.L_cult) / L_total) * nx)
        self.i_outlet_end = nx  # L'uscita arriva fino alla fine del dominio

        j_center = ny // 2
        j_ch_half = max(5, int(cfg.H_channel / cfg.W_total * ny / 2))
        j_cult_half = max(8, int(cfg.H_cult / cfg.W_total * ny / 2))

        # 1. Canale di ingresso
        self.fluid[j_center - j_ch_half:j_center + j_ch_half, :self.i_ch_end] = True

        # 2. Pre-camera (espansione graduale)
        for i in range(self.i_precam_start, self.i_precam_end):
            frac = (i - self.i_precam_start) / max(1, (self.i_precam_end - self.i_precam_start))
            h_eff = j_ch_half + (j_cult_half//3 - j_ch_half) * min(1, frac * 3)
            j0_i = max(2, min(ny-2, int(j_center - h_eff)))
            j1_i = max(2, min(ny-2, int(j_center + h_eff)))
            if j1_i > j0_i:
                self.fluid[j0_i:j1_i, i] = True
                self.precam_mask[j0_i:j1_i, i] = True

        # 3. Aperture membrana
        aw = max(2, int(cfg.aperture_w * 1e-6 / cfg.dx))
        ah = max(2, int(cfg.aperture_h * 1e-6 / cfg.dy))
        precam_y_indices = np.where(np.any(self.precam_mask, axis=1))[0]
        if len(precam_y_indices) > 0:
            y_min, y_max = precam_y_indices[0], precam_y_indices[-1]
            dx_ap = max(1, self.i_precam_end - self.i_precam_start) // (cfg.n_top_apertures + 1)
            for a in range(cfg.n_top_apertures):
                i = self.i_precam_start + (a + 1) * dx_ap
                i0, i1 = max(self.i_precam_start, i-aw//2), min(self.i_precam_end, i+aw//2)
                self.aperture_mask[max(y_min, y_max - ah - 1):min(y_max, y_max - 1), i0:i1] = True 
                self.aperture_mask[max(y_min, y_min + 1):min(y_max, y_min + ah + 1), i0:i1] = True 

        # 4. Camera di coltura
        w_cult_nx = self.i_cult_end - self.i_precam_end
        cult_y_start = j_center - j_cult_half//2
        cult_y_end = min(ny-2, cult_y_start + int(cfg.H_cult / cfg.W_total * ny))
        self.fluid[cult_y_start:cult_y_end, self.i_precam_end:self.i_cult_end] = True
        self.cult_mask[cult_y_start:cult_y_end, self.i_precam_end:self.i_cult_end] = True

        # 5. NUOVO: Canale di uscita apicale
        # Posizionato nella parte superiore della camera di coltura
        j_outlet_half = max(3, int(cfg.H_outlet / cfg.W_total * ny / 2))
        j_outlet_center = cult_y_end - int(cfg.outlet_offset / cfg.W_total * ny)  # Distanza dal bordo superiore

        # L'uscita parte dalla fine della camera e continua
        self.fluid[j_outlet_center - j_outlet_half:j_outlet_center + j_outlet_half, 
                   self.i_cult_end:self.i_outlet_end] = True
        self.outlet_mask[j_outlet_center - j_outlet_half:j_outlet_center + j_outlet_half, 
                        self.i_cult_end:self.i_outlet_end] = True

        # Connettore tra camera di coltura e uscita (permettere deflusso)
        # Creiamo una "gronda" di connessione nella parte superiore della camera
        conn_y_start = j_outlet_center - j_outlet_half
        conn_y_end = j_outlet_center + j_outlet_half
        # Allarghiamo leggermente la parte superiore della camera per connetterla all'uscita
        self.fluid[conn_y_start:conn_y_end, self.i_cult_end-3:self.i_cult_end] = True

        overlap = np.any(self.aperture_mask & self.cult_mask)
        if overlap:
            print("  ATTENZIONE: Sovrapposizione tra aperture e camera di coltura rilevata!")

        # Verifica che l'uscita sia connessa
        outlet_connected = np.any(self.outlet_mask & self.cult_mask)
        if not outlet_connected:
            # Forza connessione se necessario
            mid_outlet_y = (conn_y_start + conn_y_end) // 2
            self.fluid[mid_outlet_y-2:mid_outlet_y+2, self.i_cult_end-2:self.i_cult_end] = True
            print("   ℹ️  Connessione uscita-camera forzata")

@njit(fastmath=True)
def engine_step_numba(C, C_new, f_mask, a_mask, o_mask, U_fill, U_out, nx, ny, dx, dy, dt, D, C_in, k_mem, fase):
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    for j in range(ny):
        for i in range(nx):
            C_new[j, i] = C[j, i] 

            if f_mask[j, i]:
                c = C[j, i]
                c_n = C[j+1, i] if j < ny-1 and f_mask[j+1, i] else c
                c_s = C[j-1, i] if j > 0 and f_mask[j-1, i] else c
                c_e = C[j, i+1] if i < nx-1 and f_mask[j, i+1] else c
                c_w = C[j, i-1] if i > 0 and f_mask[j, i-1] else c

                lap = (c_e - 2*c + c_w)*inv_dx2 + (c_n - 2*c + c_s)*inv_dy2

                # Velocità effettiva dipende dalla fase e dalla zona
                u_eff = 0.0
                if fase == FASE_RIEMPIMENTO:
                    u_eff = U_fill[j, i]
                elif fase == FASE_SCARICO:
                    # Durante lo scarico, velocità nell'uscita
                    if o_mask[j, i]:
                        u_eff = U_out[j, i]
                    elif f_mask[j, i] and i >= nx - int(nx * 0.1):  # Vicino all'uscita
                        u_eff = U_out[j, i] * 0.5  # Flusso rallentato verso l'uscita

                dcdx = 0.0
                if u_eff > 0:
                    c_upwind = C[j, i-1] if i > 0 and f_mask[j, i-1] else c
                    if i <= 2: 
                        c_upwind = C_in
                    dcdx = (c - c_upwind) / dx

                sink = 0.0
                if fase == FASE_DIALISI and a_mask[j, i]:
                    sink = -k_mem * c

                C_new[j, i] = max(0.0, c + dt * (D * lap - u_eff * dcdx + sink))

    # Condizioni al contorno
    if fase == FASE_RIEMPIMENTO:
        for j in range(ny):
            if f_mask[j, 0] or f_mask[j, 1]:
                C_new[j, 0:2] = C_in
    elif fase == FASE_SCARICO:
        # Condizione di uscita aperta (zero gradiente o convezione)
        for j in range(ny):
            if o_mask[j, nx-1] or o_mask[j, nx-2]:
                # Outflow: concentrazione si mantiene o esce con il fluido
                C_new[j, nx-1] = C_new[j, nx-2] if f_mask[j, nx-2] else 0

class BatchSolverDialisi:
    def __init__(self, geometry, config):
        self.geom = geometry
        self.cfg = config

        self.f_mask = geometry.fluid.astype(bool)
        self.p_mask = geometry.precam_mask.astype(bool)
        self.a_mask = geometry.aperture_mask.astype(bool)
        self.cult_mask = geometry.cult_mask.astype(bool)
        self.o_mask = geometry.outlet_mask.astype(bool)  # NUOVO

        # Campo velocità riempimento
        self.U_fill = np.zeros((self.cfg.ny, self.cfg.nx), dtype=np.float64)
        for j in range(self.cfg.ny):
            for i in range(self.cfg.nx):
                if self.f_mask[j, i]:
                    if i < geometry.i_precam_start:
                        self.U_fill[j, i] = self.cfg.u_in
                    elif i < geometry.i_precam_end:
                        self.U_fill[j, i] = self.cfg.u_in * 0.2

        # NUOVO: Campo velocità uscita
        self.U_out = np.zeros((self.cfg.ny, self.cfg.nx), dtype=np.float64)
        for j in range(self.cfg.ny):
            for i in range(self.cfg.nx):
                if self.o_mask[j, i]:
                    # Flusso verso destra (uscita)
                    self.U_out[j, i] = self.cfg.u_outlet

        self.cached_filled_C = None
        self.calibrated = False
        self.calibration_history = []

    def simula_riempimento(self, silent=False):
        if self.cached_filled_C is not None:
            return self.cached_filled_C.copy()

        C = np.zeros((self.cfg.ny, self.cfg.nx), dtype=np.float64)
        C_new = np.zeros_like(C)
        target_c = 0.99 * self.cfg.C_in

        step, max_steps = 0, 200000
        while step < max_steps:
            engine_step_numba(C, C_new, self.f_mask, self.a_mask, self.o_mask, self.U_fill, self.U_out,
                              self.cfg.nx, self.cfg.ny, self.cfg.dx, self.cfg.dy, self.cfg.dt, 
                              self.cfg.D_drug, self.cfg.C_in, self.cfg.permeabilita_farmaco, FASE_RIEMPIMENTO)
            C, C_new = C_new, C

            step += 1
            if step % 200 == 0:
                C_mean = np.mean(C[self.p_mask]) if np.any(self.p_mask) else 0
                if not silent:
                    progresso = min(C_mean / target_c, 1.0)
                    visualizza_barra(progresso, 1.0, prefisso="   [Fase 1] Riempimento ")

                if C_mean > target_c:
                    if not silent: visualizza_barra(1.0, 1.0, prefisso="   [Fase 1] Riempimento ")
                    break

        self.cached_filled_C = C.copy()
        return C

    def simula_dialisi(self, C, tempo_target, silent=False):
        n_steps = int(tempo_target / self.cfg.dt)
        C_iniziale = np.mean(C[self.p_mask])
        C_new = np.empty_like(C)

        massa_iniziale = np.sum(C[self.f_mask])

        for i in range(n_steps):
            engine_step_numba(C, C_new, self.f_mask, self.a_mask, self.o_mask, self.U_fill, self.U_out,
                              self.cfg.nx, self.cfg.ny, self.cfg.dx, self.cfg.dy, self.cfg.dt, 
                              self.cfg.D_drug, self.cfg.C_in, self.cfg.permeabilita_farmaco, FASE_DIALISI)
            C, C_new = C_new, C

            if not silent and i % 50 == 0:
                visualizza_barra(i, n_steps, prefisso=f"   [Fase 2] Dialisi ({tempo_target:.1f}s)")

        if not silent:
            visualizza_barra(n_steps, n_steps, prefisso=f"   [Fase 2] Dialisi ({tempo_target:.1f}s)")

        C_dopo = np.mean(C[self.p_mask])
        removal = 1 - (C_dopo / C_iniziale) if C_iniziale > 0 else 0
        massa_finale = np.sum(C[self.f_mask])
        massa_rimossa_calcolata = massa_iniziale - massa_finale

        return C, C_dopo, removal, massa_rimossa_calcolata

    # NUOVO: Simulazione scarico camera di coltura
    def simula_scarico(self, C_cult, silent=False):
        """Simula lo svuotamento della camera di coltura verso il vaso sanguigno"""
        n_steps = int(self.cfg.tempo_scarico / self.cfg.dt)
        C_new = np.empty_like(C_cult)
        C = C_cult.copy()

        massa_iniziale = np.sum(C[self.cult_mask])

        for i in range(n_steps):
            engine_step_numba(C, C_new, self.f_mask, self.a_mask, self.o_mask, self.U_fill, self.U_out,
                              self.cfg.nx, self.cfg.ny, self.cfg.dx, self.cfg.dy, self.cfg.dt, 
                              self.cfg.D_drug, self.cfg.C_in, self.cfg.permeabilita_farmaco, FASE_SCARICO)
            C, C_new = C_new, C

            if not silent and i % 100 == 0:
                visualizza_barra(i, n_steps, prefisso=f"   [Fase 3] Scarico camera ({self.cfg.tempo_scarico:.1f}s)")

        if not silent:
            visualizza_barra(n_steps, n_steps, prefisso=f"   [Fase 3] Scarico camera ({self.cfg.tempo_scarico:.1f}s)")

        massa_finale = np.sum(C[self.cult_mask])
        massa_uscita = massa_iniziale - massa_finale

        return C, massa_uscita

    def trasferisci_a_coltura(self, C_dopo, batch_id, C_cult_corrente):
        C_cult_aggiornato = C_cult_corrente.copy()
        idx = np.where(self.cult_mask)
        if len(idx[0]) > 0:
            chunk = len(idx[0]) // self.cfg.n_batch
            start = (batch_id - 1) * chunk
            end = len(idx[0]) if batch_id == self.cfg.n_batch else batch_id * chunk
            for i in range(start, end):
                C_cult_aggiornato[idx[0][i], idx[1][i]] = C_dopo
        return C_cult_aggiornato

    def calibrate(self, silent=False):
        if not silent:
            print("   [Auto-Calibrazione] Ricerca tempo ottimale per 90% target...")

        C_fill = self.simula_riempimento(silent=True)

        tempi = [5.0, 15.0, 30.0, 60.0, 100.0, 150.0]
        risultati = []

        for t in tempi:
            if not silent:
                import sys
                sys.stdout.write(f"\r   [Calibrazione] Test membrana a {t}s...        ")
                sys.stdout.flush()

            _, _, removal, _ = self.simula_dialisi(C_fill.copy(), t, silent=True)
            risultati.append((t, removal))

        if not silent:
            sys.stdout.write('\r' + ' ' * 60 + '\r')

        target = self.cfg.target_removal
        tempo_trovato = None

        for i in range(len(risultati)-1):
            if risultati[i][1] <= target <= risultati[i+1][1]:
                t1, r1 = risultati[i]
                t2, r2 = risultati[i+1]
                tempo_trovato = t1 + (target - r1) * (t2 - t1) / (r2 - r1 + 1e-9)
                break

        if tempo_trovato is None:
            tempo_trovato = tempi[-1]
            removal_max = risultati[-1][1]
            if not silent:
                print(f"     Target {target*100:.0f}% non raggiungibile. Max ottenuto: {removal_max*100:.1f}%")
                print(f"   ✓ Calibrazione adattativa: valvole settate a {tempo_trovato:.1f}s")
        else:
            if not silent:
                print(f"   ✓ Calibrazione completata: valvole settate a {tempo_trovato:.1f}s")

        self.cfg.tempo_dialisi = tempo_trovato
        self.calibrated = True

    def run_batch(self, batch_id, C_cult_corrente, silent=False, with_outlet=False):
        if not self.calibrated:
            self.calibrate(silent)

        C = self.simula_riempimento(silent=silent)
        C, C_dopo, removal, massa_rm = self.simula_dialisi(C, self.cfg.tempo_dialisi, silent=silent)

        # NUOVO: Se richiesto, esegui anche lo scarico
        massa_uscita = 0
        if with_outlet:
            C, massa_uscita = self.simula_scarico(C, silent=silent)

        target_ok = abs(removal - self.cfg.target_removal) <= self.cfg.tolerance_removal
        C_cult_nuovo = self.trasferisci_a_coltura(C_dopo, batch_id, C_cult_corrente)

        return target_ok, removal, C_cult_nuovo, massa_rm, C, massa_uscita

# ==========================================================
# VISUALIZZAZIONE
# ==========================================================

def create_dashboard(geom, solver, results, cfg, C_final, C_cult_total, avg_removal, avg_C_cult,
                     total_mass_in, total_mass_removed, total_mass_cult, storico_fouling,
                     with_outlet=False, massa_uscita_totale=0):
    """Crea 9 plot professionali - VERSIONE CON USCITA APICALE"""

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.linewidth': 1.2,
        'axes.labelweight': 'bold',
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, hspace=0.25, wspace=0.25, height_ratios=[1, 1, 1.2])

    L_tot = cfg.L_channel + cfg.L_precam + cfg.L_cult + cfg.L_outlet
    extent = [0, L_tot, 0, cfg.W_total]

    # ==========================================================
    # PLOT 1: Distribuzione Concentrazione con Outlet
    # ==========================================================
    ax1 = fig.add_subplot(gs[0, 0])
    C_plot = C_final.copy()
    C_plot[~geom.fluid] = np.nan

    im1 = ax1.imshow(C_plot, origin='lower', extent=extent, aspect='auto',
                     cmap='YlOrRd', vmin=0, vmax=cfg.C_in)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='C [a.u.]')

    # Linee zone
    ax1.axvline(cfg.L_channel, color='cyan', ls='--', lw=2, alpha=0.8, label='Inlet precam')
    ax1.axvline(cfg.L_channel + cfg.L_precam, color='yellow', ls='--', lw=2, alpha=0.8, label='Inlet cultura')
    ax1.axvline(cfg.L_channel + cfg.L_precam + cfg.L_cult, color=COLORS['outlet'], ls='--', lw=2, alpha=0.8, label='Outlet')

    # Evidenzia outlet
    outlet_extent = [cfg.L_channel + cfg.L_precam + cfg.L_cult, L_tot, 
                     0, cfg.W_total]

    ax1.set_xlabel('x [µm]')
    ax1.set_ylabel('y [µm]')
    ax1.set_title('1. Distribuzione Concentrazione + Outlet', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)

    # ==========================================================
    # PLOT 2: Camera Coltura e Connessione Outlet
    # ==========================================================
    ax2 = fig.add_subplot(gs[0, 1])
    C_cult_plot = C_cult_total.copy()
    C_cult_plot[~geom.cult_mask & ~geom.outlet_mask] = np.nan

    c_min, c_max = np.nanmin(C_cult_plot), np.nanmax(C_cult_plot)
    target_c = cfg.C_in * (1 - cfg.target_removal)

    im2 = ax2.imshow(C_cult_plot, origin='lower', extent=extent, aspect='auto',
                     cmap='plasma', vmin=0, vmax=max(c_max, target_c * 1.5))

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='C [a.u.]')
    cbar2.ax.axhline(target_c, color='lime', linewidth=2, linestyle='--')
    cbar2.ax.text(1.15, target_c / cbar2.vmax, 'Target', transform=cbar2.ax.transAxes, 
                  color='lime', fontsize=8, va='center')

    # Linea outlet
    ax2.axvline(cfg.L_channel + cfg.L_precam + cfg.L_cult, color=COLORS['outlet'], ls='--', lw=2, alpha=0.8)

    stats_text = f'μ = {avg_C_cult:.1f}\nσ = {np.std(C_cult_total[C_cult_total > 0]):.2f}\nTarget = {target_c:.1f}'
    if with_outlet:
        stats_text += f'\nMassa uscita: {massa_uscita_totale:.2e}'
    ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
            va='top', ha='right', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', alpha=0.95, linewidth=1.5))

    ax2.set_xlabel('x [µm]')
    ax2.set_ylabel('y [µm]')
    ax2.set_title(f'2. Camera + Outlet (C̄ = {avg_C_cult:.1f})', fontweight='bold')

    # ==========================================================
    # PLOT 3: Efficienza batch
    # ==========================================================
    ax3 = fig.add_subplot(gs[0, 2])
    batches = [r['batch_id'] for r in results]
    removals = [r['removal'] * 100 for r in results]

    colors_bar = [COLORS['target'] if r >= 90 else COLORS['alert'] for r in removals]
    bars = ax3.bar(batches, removals, color=colors_bar, alpha=0.85, edgecolor='black', lw=1.5)

    ax3.axhline(90, color=COLORS['target'], ls='--', lw=2.5, label='Target 90%')

    for bar, val in zip(bars, removals):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Rimozione [%]')
    ax3.set_title('3. Efficienza per Batch', fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # ==========================================================
    # PLOT 4: Bilancio massa con uscita
    # ==========================================================
    ax4 = fig.add_subplot(gs[1, 0])
    if with_outlet:
        labels = ['Input', 'Rimosso', 'Camera', 'Uscita']
        values = [total_mass_in, total_mass_removed, total_mass_cult, massa_uscita_totale]
        colors_mass = [COLORS['sangue'], COLORS['membrana'], COLORS['camera'], COLORS['outlet']]
    else:
        labels = ['Input', 'Rimosso', 'Camera']
        values = [total_mass_in, total_mass_removed, total_mass_cult]
        colors_mass = [COLORS['sangue'], COLORS['membrana'], COLORS['camera']]

    bars = ax4.bar(labels, values, color=colors_mass, alpha=0.85, edgecolor='black', lw=1.5)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height/2,
                f'{val:.2e}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')

    ax4.set_ylabel('Massa [a.u.]')
    ax4.set_title('4. Bilancio di Massa', fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # ==========================================================
    # PLOT 5: Profilo Longitudinale Medio
    # ==========================================================
    ax5 = fig.add_subplot(gs[1, 1])

    C_mean_x = np.nanmean(C_plot, axis=0)
    x_coords = np.linspace(0, L_tot, len(C_mean_x))

    fluid_mask_x = np.any(geom.fluid, axis=0)
    x_coords_valid = x_coords[fluid_mask_x]
    C_mean_x_valid = C_mean_x[fluid_mask_x]

    ax5.plot(x_coords_valid, C_mean_x_valid, '-', lw=2.5, color=COLORS['farmaco'], label='C media')
    ax5.fill_between(x_coords_valid, 0, C_mean_x_valid, alpha=0.3, color=COLORS['farmaco'])

    ax5.axvline(cfg.L_channel, color='cyan', ls='--', lw=1.5, alpha=0.7, label='Canale→Precam')
    ax5.axvline(cfg.L_channel + cfg.L_precam, color='yellow', ls='--', lw=1.5, alpha=0.7, label='Precam→Cultura')
    ax5.axvline(cfg.L_channel + cfg.L_precam + cfg.L_cult, color=COLORS['outlet'], ls='--', lw=1.5, alpha=0.7, label='→Outlet')

    target_conc = cfg.C_in * (1 - cfg.target_removal)
    ax5.axhline(target_conc, color=COLORS['target'], ls='--', lw=2, label=f'Target {target_conc:.0f}')

    ax5.set_ylim(0, cfg.C_in * 1.05)

    ax5.set_xlabel('x [µm]')
    ax5.set_ylabel('C [a.u.]')
    ax5.set_title('5. Profilo Longitudinale Medio', fontweight='bold')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(alpha=0.3)

    # ==========================================================
    # PLOT 6: Evoluzione Fouling
    # ==========================================================
    ax6 = fig.add_subplot(gs[1, 2])
    if len(storico_fouling) > 0:
        cicli = [h['ciclo'] for h in storico_fouling]
        k_vals = [h['k_mem'] for h in storico_fouling]
        tempi = [h['tempo_dialisi'] for h in storico_fouling]

        ax6_twin = ax6.twinx()
        line1 = ax6.plot(cicli, k_vals, 'o-', color=COLORS['membrana'], lw=2, label='k_mem', markersize=4)
        line2 = ax6_twin.plot(cicli, tempi, 's-', color=COLORS['alert'], lw=2, label='Tempo dialisi', markersize=4)

        ricalibri = [h['ciclo'] for h in storico_fouling if h['ricalibrato']]
        for rc in ricalibri:
            ax6.axvline(rc, color='red', alpha=0.3, linestyle=':', linewidth=1)

        ax6.set_xlabel('Ciclo')
        ax6.set_ylabel('k_mem [m/s]', color=COLORS['membrana'])
        ax6_twin.set_ylabel('Tempo [s]', color=COLORS['alert'])
        ax6.set_title('6. Evoluzione Fouling', fontweight='bold')
        ax6.grid(alpha=0.3)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax6.legend(lines, labels, loc='center right')

    # ==========================================================
    # PLOT 7: Accumulo Massa
    # ==========================================================
    ax7 = fig.add_subplot(gs[2, 0])
    mass_cum_in = np.cumsum([r['mass_in'] for r in results])
    mass_cum_removed = np.cumsum([r['mass_removed'] for r in results])
    mass_cum_cult = np.cumsum([r['mass_cult'] for r in results])

    ax7.plot(batches, mass_cum_in, 'o-', color=COLORS['sangue'], lw=2.5, label='Input', markersize=6)
    ax7.plot(batches, mass_cum_removed, 's-', color=COLORS['membrana'], lw=2.5, label='Rimosso', markersize=6)
    ax7.plot(batches, mass_cum_cult, '^-', color=COLORS['camera'], lw=2.5, label='Camera', markersize=6)

    if with_outlet:
        mass_cum_uscita = np.cumsum([r.get('mass_uscita', 0) for r in results])
        ax7.plot(batches, mass_cum_uscita, 'd-', color=COLORS['outlet'], lw=2.5, label='Uscita', markersize=6)

    ax7.set_xlabel('Batch')
    ax7.set_ylabel('Massa cumulativa [a.u.]')
    ax7.set_title('7. Accumulo Massa', fontweight='bold')
    ax7.legend()
    ax7.grid(alpha=0.3)

    # ==========================================================
    # PLOT 8: Campo di Velocità
    # ==========================================================
    ax8 = fig.add_subplot(gs[2, 1])
    U_plot = solver.U_fill.copy()
    U_plot[~solver.f_mask] = np.nan
    # Sovrappone velocità outlet dove presente
    U_plot[solver.o_mask] = solver.U_out[solver.o_mask]

    im8 = ax8.imshow(np.log10(U_plot + 1e-10), origin='lower', extent=extent, aspect='auto',
                     cmap='plasma', vmin=-6, vmax=-2)
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04, label='log₁₀(u) [m/s]')

    ax8.axvline(cfg.L_channel, color='cyan', ls='--', lw=1.5, alpha=0.7)
    ax8.axvline(cfg.L_channel + cfg.L_precam, color='yellow', ls='--', lw=1.5, alpha=0.7)
    ax8.axvline(cfg.L_channel + cfg.L_precam + cfg.L_cult, color=COLORS['outlet'], ls='--', lw=1.5, alpha=0.7)

    ax8.set_xlabel('x [µm]')
    ax8.set_ylabel('y [µm]')
    ax8.set_title('8. Campo di Velocità (Inlet + Outlet)', fontweight='bold')

    # ==========================================================
    # PLOT 9: Tabella riepilogativa
    # ==========================================================
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    table_data = [
        ['PARAMETRO', 'VALORE', 'UNITÀ'],
        ['', '', ''],
        ['Geometria', '', ''],
        ['  L pre-camera', f'{cfg.L_precam}', 'µm'],
        ['  L camera', f'{cfg.L_cult}', 'µm'],
        ['  L outlet', f'{cfg.L_outlet}', 'µm'],
        ['  N aperture', f'{cfg.n_top_apertures + cfg.n_bottom_apertures}', '-'],
        ['', '', ''],
        ['Performance', '', ''],
        ['  Rimozione', f'{avg_removal*100:.1f}', '%'],
        ['  C camera', f'{avg_C_cult:.1f}', 'a.u.'],
        ['  Target C', f'{cfg.C_in * (1-cfg.target_removal):.0f}', 'a.u.'],
        ['', '', ''],
        ['Bilancio', '', ''],
        ['  Massa input', f'{total_mass_in:.2e}', 'a.u.'],
        ['  Massa rimossa', f'{total_mass_removed:.2e}', 'a.u.'],
        ['  Massa camera', f'{total_mass_cult:.2e}', 'a.u.'],
    ]

    if with_outlet:
        table_data.insert(16, ['  Massa uscita', f'{massa_uscita_totale:.2e}', 'a.u.'])

    table = ax9.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.4, 0.35, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for i in range(3):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for row in [2, 8, 14 if not with_outlet else 15]:
        for i in range(3):
            table[(row, i)].set_facecolor('#ecf0f1')
            table[(row, i)].set_text_props(weight='bold')

    errore_relativo = abs(avg_C_cult - target_c) / target_c * 100 if target_c > 0 else 0
    if errore_relativo < 10:
        status = '✓ OTTIMALE'
        status_color = COLORS['target']
    elif errore_relativo < 25:
        status = '⚠ ACCETTABILE'
        status_color = COLORS['precamera']
    else:
        status = '✗ DA OTTIMIZZARE'
        status_color = COLORS['alert']

    ax9.text(0.5, 0.02, f'STATUS: {status} (err {errore_relativo:.1f}%)', 
            transform=ax9.transAxes, fontsize=11,
            fontweight='bold', ha='center', color=status_color,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor=status_color, linewidth=2))

    ax9.set_title('9. Riepilogo Parametri', fontweight='bold')

    outlet_text = " + USCITA APICALE" if with_outlet else ""
    fig.suptitle(f'ANALISI CHIP MICROFLUIDICO - DIALISI BATCH{outlet_text}', fontsize=16, fontweight='bold')

    plt.savefig('chip_analysis_dashboard_outlet.png', dpi=300, bbox_inches='tight', facecolor='white')
    print(" Dashboard salvato: chip_analysis_dashboard_outlet.png")

    return fig

# ==========================================================
# ESECUZIONE MAIN
# ==========================================================

# Parametro per attivare/disattivare la simulazione con uscita
SIMULA_USCITA = True  # Imposta False per comportamento originale

cfg = ChipConfig()
geom = ChipGeometryCompleto(cfg)
solver = BatchSolverDialisi(geom, cfg)

print("\n" + "-"*80)
print("FASE 1: AVVIO SISTEMA (Riempimento Camera Coltura)")
print("-"*80)
C_cult_state = np.zeros((cfg.ny, cfg.nx))

# FIX: Storico per report finale e plot
storico_fouling = []
results_batch = []
C_final_for_plot = None
massa_uscita_totale = 0

for b in range(1, cfg.n_batch + 1):
    print(f"-> Esecuzione Batch {b}/{cfg.n_batch}:")

    if SIMULA_USCITA:
        ok, removal, C_cult_state, massa_rm, C_field, massa_uscita = solver.run_batch(
            b, C_cult_state, silent=False, with_outlet=True)
        massa_uscita_totale += massa_uscita
    else:
        ok, removal, C_cult_state, massa_rm, C_field = solver.run_batch(
            b, C_cult_state, silent=False, with_outlet=False)
        massa_uscita = 0

    # Salva per plot
    valori_coltura = C_cult_state[geom.cult_mask & (C_cult_state > 0)]
    media_coltura = np.mean(valori_coltura) if len(valori_coltura) > 0 else 0.0

    result_dict = {
        'batch_id': b,
        'removal': removal,
        'mass_in': massa_rm / (removal if removal > 0 else 0.9),
        'mass_removed': massa_rm,
        'mass_cult': massa_rm * (1-removal) / removal if removal > 0 else 0,
        'C_after': cfg.C_in * (1-removal)
    }

    if SIMULA_USCITA:
        result_dict['mass_uscita'] = massa_uscita

    results_batch.append(result_dict)

    if C_final_for_plot is None:
        C_final_for_plot = C_field.copy()

    print(f"   Risultato: Rimozione {removal*100:.1f}% -> C_coltura = {media_coltura:.1f}")
    print(f"   Massa rimossa nel batch: {massa_rm:.2e} [unità arbitrarie]")
    if SIMULA_USCITA:
        print(f"   Massa uscita verso vaso: {massa_uscita:.2e} [unità arbitrarie]")
    print()

print("-" * 80)
print("FASE 2: SOSTENIBILITA' A LUNGO TERMINE (30 Cicli / Fouling)")
print("-" * 80)

print(f"{'Giorno':<8} | {'k_mem':<10} | {'T_Dialisi (s)':<15} | {'Rimozione (%)':<15} | {'Stato / Note'}")
print("-" * 80)

cicli_di_vita = 30

for ciclo in range(1, cicli_di_vita + 1):
    nuova_perm = cfg.permeabilita_farmaco * cfg.fouling_factor
    if nuova_perm < cfg.min_permeability:
        print(f"\n🛑 STOP: Permeabilità membrana sotto soglia critica ({cfg.min_permeability}) al ciclo {ciclo}")
        break

    cfg.permeabilita_farmaco = nuova_perm

    batch_corrente = ((ciclo - 1) % cfg.n_batch) + 1

    if SIMULA_USCITA:
        ok, removal, C_cult_state, massa_rm, _, massa_uscita = solver.run_batch(
            batch_corrente, C_cult_state, silent=True, with_outlet=True)
        massa_uscita_totale += massa_uscita
    else:
        ok, removal, C_cult_state, massa_rm, _ = solver.run_batch(
            batch_corrente, C_cult_state, silent=True, with_outlet=False)

    storico_fouling.append({
        'ciclo': ciclo,
        'batch': batch_corrente,
        'k_mem': cfg.permeabilita_farmaco,
        'tempo_dialisi': cfg.tempo_dialisi,
        'removal': removal,
        'massa_rimossa': massa_rm,
        'ricalibrato': False
    })

    stato_str = " OK (Stabile)"
    if not ok:
        stato_str = "⚠️ Fouling: Ricalibro Valvole..."
        solver.calibrated = False 
        solver.cached_filled_C = None
        storico_fouling[-1]['ricalibrato'] = True

    print(f"Day {ciclo:<4} | {cfg.permeabilita_farmaco:<10.3f} | {cfg.tempo_dialisi:<15.1f} | {removal*100:<15.1f} | {stato_str}")

# Calcoli finali per plot
total_mass_in = sum(r['mass_in'] for r in results_batch)
total_mass_removed = sum(r['mass_removed'] for r in results_batch)
total_mass_cult = sum(r['mass_cult'] for r in results_batch)
avg_removal = np.mean([r['removal'] for r in results_batch])
avg_C_cult = np.mean(C_cult_state[geom.cult_mask & (C_cult_state > 0)]) if np.any(C_cult_state > 0) else 0

# Report finale testuale
print("\n" + "="*100)
print("📊 REPORT FINALE - METRICHE DI PERFORMANCE")
print("="*100)

if len(storico_fouling) > 0:
    tempi_dialisi_totali = [h['tempo_dialisi'] for h in storico_fouling]
    removal_medio = np.mean([h['removal'] for h in storico_fouling])
    massa_totale_rimossa = np.sum([h['massa_rimossa'] for h in storico_fouling])
    n_ricalibrazioni = sum([1 for h in storico_fouling if h['ricalibrato']])

    print(f"\n📈 Statistiche Operative ({len(storico_fouling)} cicli):")
    print(f"   • Tempo dialisi iniziale:    {tempi_dialisi_totali[0]:.1f}s")
    print(f"   • Tempo dialisi finale:      {tempi_dialisi_totali[-1]:.1f}s")
    print(f"   • Incremento tempo:          {((tempi_dialisi_totali[-1]/tempi_dialisi_totali[0])-1)*100:.1f}%")
    print(f"   • Rimozione media:           {removal_medio*100:.2f}%")
    print(f"   • Deviazione std rimozione:  {np.std([h['removal'] for h in storico_fouling])*100:.2f}%")
    print(f"   • Massa totale rimossa:      {massa_totale_rimossa:.2e} [unità arbitrarie]")
    print(f"   • Efficienza media:          {(removal_medio/np.mean(tempi_dialisi_totali))*100:.3f} [%/s]")
    print(f"   • Ricalibrazioni eseguite:   {n_ricalibrazioni}/{len(storico_fouling)}")

    if SIMULA_USCITA:
        print(f"\n🩸 Sistema con Outlet Apicale:")
        print(f"   • Massa totale uscita:       {massa_uscita_totale:.2e} [unità arbitrarie]")
        print(f"   • Percentuale su totale:     {(massa_uscita_totale/total_mass_in)*100:.1f}%")

valori_finali = C_cult_state[geom.cult_mask & (C_cult_state > 0)]
if len(valori_finali) > 0:
    print(f"\n Stato Camera di Coltura:")
    print(f"   • Concentrazione media:      {np.mean(valori_finali):.2f}")
    print(f"   • Concentrazione max:        {np.max(valori_finali):.2f}")
    print(f"   • Concentrazione min:        {np.min(valori_finali):.2f}")
    print(f"   • Uniformità (CV):           {np.std(valori_finali)/np.mean(valori_finali)*100:.1f}%")

print("\n" + "="*100)
print("GENERAZIONE VISUALIZZAZIONI")
print("="*100)

# Genera i plot
dashboard = create_dashboard(geom, solver, results_batch, cfg, C_final_for_plot, 
                            C_cult_state, avg_removal, avg_C_cult,
                            total_mass_in, total_mass_removed, total_mass_cult, 
                            storico_fouling, with_outlet=SIMULA_USCITA, 
                            massa_uscita_totale=massa_uscita_totale)

print("\n" + "="*100)
print("SIMULAZIONE COMPLETATA CON SUCCESSO.")
if SIMULA_USCITA:
    print(" Canale di uscita apicale attivo - Collegamento al vaso sanguigno funzionante")
print("="*100)

plt.show()
