import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend non interattivo: salva file ovunque, anche headless
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from scipy.sparse.linalg import eigsh
from scipy.io import wavfile

from skfem import (MeshTri, Basis, ElementTriP1, asm, condense,
                   BilinearForm)
from skfem.helpers import dot, grad

# ===========================================================================
# 1. PARAMETRI -- modifica liberamente per le tue ipotesi
# ===========================================================================

# --- Geometria dell'ellisse ---
SEMI_A = 0.30          # semiasse maggiore [m]
SEMI_B = 0.18          # semiasse minore  [m]   (b < a -> ellisse; b = a -> cerchio)
MESH_REFINE = 5        # raffinamento mesh: 4 = veloce, 5 = buono, 6 = fine (lento)

# --- Proprietà fisiche della membrana ---
# La frequenza di una membrana ideale: f = (1/2pi) * sqrt(T/sigma) * sqrt(lambda)
# dove lambda è l'autovalore [1/m^2] restituito dal FEM (già dimensionale,
# perché la mesh è in metri reali).
TENSION = 2000.0       # tensione superficiale T [N/m]
SIGMA   = 0.26         # densità di massa per unità di area [kg/m^2]
# Esempi indicativi di sigma (spessore * densità volumetrica):
#   Mylar/PET 0.1mm  : ~0.139    | Mylar/PET 0.25mm : ~0.347
#   pelle batter. ~0.3mm: ~0.3   | gomma 0.5mm     : ~0.5
# Esempi di T: pelle di tamburo accordata ~1000-4000 N/m.

# --- Modi da calcolare / visualizzare ---
N_MODES = 12           # quanti modi calcolare e plottare

# --- Sintesi audio ---
SR = 44100             # sample rate [Hz]
DUR = 2.0              # durata di ogni tono [s]
FADE = 0.02            # fade in/out [s] per evitare click
AUDIO_NORMALIZE_F0 = None
# Se None -> usa le frequenze FISICHE reali (da T, sigma, dimensione).
# Se imposti un numero (es. 220.0) -> il modo fondamentale viene portato a
# quella frequenza in Hz e gli altri scalati in proporzione (rapporti modali).
# Utile se le frequenze fisiche cadono fuori dalla banda udibile.

OUTDIR = "output_ellisse"
os.makedirs(OUTDIR, exist_ok=True)

# ===========================================================================
# 2. MESH ELLITTICA
#    Parto da un disco unitario raffinato e lo "schiaccio" nei semiassi.
# ===========================================================================

def build_ellipse_mesh(a, b, refine):
    """Mesh triangolare di un'ellisse di semiassi a, b."""
    # Disco unitario di scikit-fem, poi raffinato.
    m = MeshTri.init_circle(refine)   # disco di raggio 1 centrato nell'origine
    p = m.p.copy()
    p[0, :] *= a    # x -> a*x
    p[1, :] *= b    # y -> b*y
    return MeshTri(p, m.t)

mesh = build_ellipse_mesh(SEMI_A, SEMI_B, MESH_REFINE)
basis = Basis(mesh, ElementTriP1())
print(f"Mesh: {mesh.p.shape[1]} nodi, {mesh.t.shape[1]} triangoli")

# ===========================================================================
# 3. PROBLEMA AGLI AUTOVALORI  K v = lambda M v
#    K = rigidezza  (integrale di grad u . grad v)
#    M = massa      (integrale di u v)
#    Bordo fisso: u = 0 sul perimetro dell'ellisse (membrana ancorata).
# ===========================================================================

@BilinearForm
def stiffness(u, v, _):
    return dot(grad(u), grad(v))

@BilinearForm
def mass(u, v, _):
    return u * v

K = asm(stiffness, basis)
M = asm(mass, basis)

# Nodi interni (escludo il bordo, dove u=0)
boundary_dofs = basis.get_dofs()          # DOF sul bordo
interior = basis.complement_dofs(boundary_dofs)

Kc = K[interior][:, interior]
Mc = M[interior][:, interior]

# Risolvo per i piu' piccoli autovalori (modi di frequenza piu' bassa).
# sigma=0 + which='LM' su shift-invert è il modo robusto in scipy.
vals, vecs = eigsh(Kc.tocsc(), k=N_MODES, M=Mc.tocsc(),
                   sigma=0.0, which='LM')

order = np.argsort(vals)
vals = vals[order]
vecs = vecs[:, order]

# Ricostruisco i modi sull'intera mesh (zeri sul bordo).
modes = np.zeros((basis.N, N_MODES))
modes[interior, :] = vecs

# lambda [1/m^2] -> frequenze fisiche [Hz]
lam = np.clip(vals, 0, None)
c = np.sqrt(TENSION / SIGMA)               # velocità d'onda [m/s]
freqs_phys = (c / (2.0 * np.pi)) * np.sqrt(lam)

print("\nModo |   sqrt(lambda) [1/m] |  f fisica [Hz]")
print("-----+----------------------+----------------")
for i in range(N_MODES):
    print(f" {i+1:>3} | {np.sqrt(lam[i]):>20.4f} | {freqs_phys[i]:>13.2f}")

# ===========================================================================
# 3b. CLASSIFICAZIONE DEI MODI -- i due indici (m, n) e la parità
#
# Un modo ellittico si etichetta con DUE indici, come nel cerchio:
#   m  = indice angolare  -> numero di linee nodali che attraversano il centro
#   n  = indice radiale   -> numero di linee nodali (quasi-)ellittiche concentriche
# e una PARITA' rispetto all'asse maggiore (x):
#   even (ce_m)  -> simmetrico:    u(x, -y) = +u(x, y)
#   odd  (se_m)  -> antisimmetrico: u(x, -y) = -u(x, y)
# Notazione classica (McLachlan): modi ce_{m,n} (pari) e se_{m,n} (dispari).
#
# ATTENZIONE: il FEM ordina i modi per frequenza, non assegna gli indici.
# Qui li attribuiamo a posteriori con un'EURISTICA (parità esatta;
# m, n contati dai nodi). Va SEMPRE verificata a occhio sulle figure e,
# per il paper, validata contro la trattazione analitica (Mathieu /
# Gutierrez-Vega). I numeri possono richiedere correzioni manuali sui modi
# alti, dove le linee nodali si infittiscono.
# ===========================================================================

from scipy.interpolate import griddata

# --- OVERRIDE MANUALE degli indici ----------------------------------------
# Dopo aver guardato le figure, correggi qui gli indici sbagliati.
# Chiave = numero del modo (1-based). Valore = (m, n).  La parità NON si
# tocca (e' calcolata in modo esatto). Lascia vuoto {} per fidarti dell'auto.
# Esempio: MANUAL_INDEX = {5: (2, 0), 8: (0, 1), 11: (1, 1)}
MANUAL_INDEX = {2: (1, 0), 3: (1, 0), 5: (2, 0), 6: (3, 0),
                7: (3, 0), 10: (4, 0), 11: (1, 1), 12: (5, 0)}
# ---------------------------------------------------------------------------

def classify_mode(mode):
    """
    Restituisce (m, n, parità) stimati per un modo.

    parità: 'even' (simmetrico rispetto all'asse maggiore x) o 'odd'.
    m: linee nodali angolari (radiali).
    n: linee nodali ellittiche concentriche.
    """
    x = mesh.p[0]; y = mesh.p[1]
    z = mode / (np.max(np.abs(mode)) + 1e-15)

    # --- Parità rispetto all'asse maggiore (riflessione y -> -y): esatta ---
    z_refl = griddata((x, y), z, (x, -y), method='linear', fill_value=0.0)
    v = ~np.isnan(z_refl)
    overlap = np.sum(z[v]*z_refl[v]) / (np.sum(z[v]*z[v]) + 1e-15)
    parity = 'even' if overlap >= 0 else 'odd'

    # --- m: cambi di segno lungo un anello a ~0.8 del bordo -----------------
    th = np.linspace(0, 2*np.pi, 1440, endpoint=False)
    rx = 0.80*SEMI_A*np.cos(th); ry = 0.80*SEMI_B*np.sin(th)
    ring = griddata((x, y), z, (rx, ry), method='cubic', fill_value=0.0)
    # liscio leggero per togliere il rumore di interpolazione
    ring = np.convolve(ring, np.ones(7)/7, mode='same')
    sgn = np.sign(ring); sgn = sgn[sgn != 0]
    m = int(round(np.sum(np.abs(np.diff(sgn)) > 0) / 2))

    # --- n: anelli nodali ellittici, lungo il semiasse minore (asse y) ------
    # Campiono il modo su una linea dal centro verso il bordo lungo y, mi
    # fermo a 0.88 del bordo (oltre, il modo crolla a ~0 e genera falsi nodi),
    # liscio e conto i cambi di segno della curva. Stabile perché è un
    # campionamento 1-D, non su griglia 2-D.
    sy = np.linspace(0.02, 0.88, 200) * SEMI_B
    axisline = griddata((x, y), z, (np.zeros_like(sy), sy),
                        method='cubic', fill_value=0.0)
    axisline = np.convolve(axisline, np.ones(9)/9, mode='same')
    # azzero le code piccolissime per non contare oscillazioni di rumore
    axisline[np.abs(axisline) < 0.02*np.max(np.abs(axisline) + 1e-15)] = 0.0
    asgn = np.sign(axisline); asgn = asgn[asgn != 0]
    crossings = int(np.sum(np.abs(np.diff(asgn)) > 0)) if asgn.size else 0
    # se il modo è dispari rispetto a x, il centro è già un nodo radiale:
    # il primo crossing non è un anello -> lo scalo via.
    n = max(crossings - (1 if parity == 'odd' and m > 0 else 0), 0)

    return m, n, parity

mode_labels = []      # etichetta breve, es. "ce(2,0)"
mode_descr  = []      # etichetta estesa per titolo figura
print("\nModo | (m, n) | parità | sqrt(lambda) | f [Hz]   (auto; override applicato se presente)")
print("-----+--------+---------+--------------+--------")
for i in range(N_MODES):
    m, n, par = classify_mode(modes[:, i])
    if (i+1) in MANUAL_INDEX:                 # override manuale
        m, n = MANUAL_INDEX[i+1]
    sym = 'ce' if par == 'even' else 'se'     # ce = coseno-ellittico, se = seno-ellittico
    label = f"{sym}({m},{n})"
    mode_labels.append(label)
    mode_descr.append(f"{label}  [{par}]")
    flag = '  <- override' if (i+1) in MANUAL_INDEX else ''
    print(f" {i+1:>3} | ({m},{n})  | {par:<7} | {np.sqrt(lam[i]):>12.3f} | {freqs_phys[i]:>6.1f}{flag}")

# ===========================================================================
# 4. FIGURE DEI PATTERN NODALI
#    Una figura riepilogativa a griglia + figure singole ad alta qualità.
# ===========================================================================

triang = mtri.Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

# Palette personalizzata viola <-> arancione (diverging, bianco al centro).
# Ampiezza negativa = viola, positiva = arancione, zero (linea nodale) = bianco.
from matplotlib.colors import LinearSegmentedColormap
PURPLE_ORANGE = LinearSegmentedColormap.from_list(
    "purple_orange",
    ["#5B2A86",   # viola intenso  (ampiezza -1)
     "#9B6BC4",   # viola chiaro
     "#F4F0F7",   # quasi bianco   (nodo, ampiezza 0)
     "#F2A65A",   # arancione chiaro
     "#E8761A"],  # arancione intenso (ampiezza +1)
    N=256)

def plot_mode(ax, mode, title):
    z = mode / np.max(np.abs(mode))        # normalizzo ampiezza a [-1, 1]
    # mappa di colore dell'ampiezza
    ax.tripcolor(triang, z, shading='gouraud', cmap=PURPLE_ORANGE,
                 vmin=-1, vmax=1)
    # LINEA NODALE: contorno a z = 0 (dove la membrana resta ferma)
    ax.tricontour(triang, z, levels=[0.0], colors='k', linewidths=1.2)
    # contorno dell'ellisse
    th = np.linspace(0, 2*np.pi, 400)
    ax.plot(SEMI_A*np.cos(th), SEMI_B*np.sin(th), 'k-', lw=1.5)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=10)

# 4a. Griglia riepilogativa
ncol = 4
nrow = int(np.ceil(N_MODES / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(3*ncol, 2.6*nrow))
axes = np.atleast_1d(axes).ravel()
for i in range(N_MODES):
    plot_mode(axes[i], modes[:, i],
              f"{mode_descr[i]}\nf = {freqs_phys[i]:.1f} Hz")
for j in range(N_MODES, len(axes)):
    axes[j].axis('off')
fig.suptitle(f"Modi della membrana ellittica  (a={SEMI_A} m, b={SEMI_B} m)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.97])
grid_path = os.path.join(OUTDIR, "modi_griglia.png")
fig.savefig(grid_path, dpi=200)
plt.close(fig)
print(f"\nSalvata griglia: {grid_path}")

# 4b. Figure singole (per inserimento individuale nel paper)
for i in range(N_MODES):
    fig, ax = plt.subplots(figsize=(4, 3.2))
    plot_mode(ax, modes[:, i],
              f"{mode_descr[i]}  -  f = {freqs_phys[i]:.1f} Hz")
    fig.tight_layout()
    # nome file con indici, es. modo_01_ce-2-0.png
    safe = mode_labels[i].replace('(', '-').replace(')', '').replace(',', '-')
    p = os.path.join(OUTDIR, f"modo_{i+1:02d}_{safe}.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)
print(f"Salvate {N_MODES} figure singole in {OUTDIR}/")

# ===========================================================================
# 5. SINTESI AUDIO -- toni puri alle frequenze modali
# ===========================================================================

def pure_tone(freq, dur, sr, fade):
    t = np.linspace(0, dur, int(sr*dur), endpoint=False)
    wave = np.sin(2*np.pi*freq*t)
    nf = int(sr*fade)
    if nf > 0:
        env = np.ones_like(wave)
        env[:nf] = np.linspace(0, 1, nf)
        env[-nf:] = np.linspace(1, 0, nf)
        wave *= env
    return wave

# Scelgo le frequenze da sonificare: fisiche oppure normalizzate.
if AUDIO_NORMALIZE_F0 is None:
    audio_freqs = freqs_phys.copy()
    freq_label = "fisiche"
else:
    audio_freqs = freqs_phys / freqs_phys[0] * AUDIO_NORMALIZE_F0
    freq_label = f"normalizzate (f0 -> {AUDIO_NORMALIZE_F0} Hz)"

print(f"\nSintesi audio con frequenze {freq_label}.")

# 5a. Ogni modo come tono separato, in sequenza (separati da silenzio)
gap = np.zeros(int(SR*0.3))
seq = []
for f in audio_freqs:
    seq.append(pure_tone(f, DUR, SR, FADE))
    seq.append(gap)
seq = np.concatenate(seq)
seq /= np.max(np.abs(seq)) + 1e-12
wavfile.write(os.path.join(OUTDIR, "modi_sequenza.wav"),
              SR, (seq*0.9*32767).astype(np.int16))

# 5b. Accordo: tutti i modi sovrapposti (la "voce" complessiva dell'ellisse)
tmax = max(audio_freqs.size and DUR, DUR)
chord = np.zeros(int(SR*DUR))
for k, f in enumerate(audio_freqs):
    # peso decrescente sui modi alti, cosi' la fondamentale domina
    chord += pure_tone(f, DUR, SR, FADE) / (k+1)
chord /= np.max(np.abs(chord)) + 1e-12
wavfile.write(os.path.join(OUTDIR, "modi_accordo.wav"),
              SR, (chord*0.9*32767).astype(np.int16))

print(f"Salvati: modi_sequenza.wav, modi_accordo.wav in {OUTDIR}/")
print("\nFatto. Tutti gli output sono nella cartella:", os.path.abspath(OUTDIR))
