# elleipsis

## MODI DI VIBRAZIONE DI UNA MEMBRANA ELLITTICA

### Analisi modale via FEM (scikit-fem) + figure nodali + sintesi audio
----------------------------------------------------------------------------
Per: "Dalla forma fisica alla forma d'ascolto" (CIM 2026)

Azioni:
  1. Costruzione di una mesh triangolare del dominio ellittico.
  2. Risoluzione del problema agli autovalori della membrana (Helmholtz):
         K v = λ M v    con bordo fisso (Dirichlet u=0).
  3. Salvataggio delle figure dei primi modi con le LINEE NODALI evidenziate.
  4. Sintetizzazione dell'audio (toni puri) usando le frequenze fisiche
     calcolate da TENSIONE / DENSITÀ / DIMENSIONE (variabili sotto).

La geometria (forma dei modi e autovalori sqrt(lambda)) si calcola UNA volta:
dipende solo dall'eccentricità dell'ellisse. Tensione, densità e dimensione
sono variabili che scalano gli autovalori in frequenze assolute (Hz).

Portabile: Colab, Replit, locale. Dipendenze: scikit-fem, numpy, scipy,
matplotlib. (Su Colab: !pip install scikit-fem)
