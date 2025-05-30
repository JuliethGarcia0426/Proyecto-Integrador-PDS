# -*- coding: utf-8 -*-
"""
Created on Thu May 29 16:14:31 2025

@author: Usuario
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

# === Cargar archivo
archivo =  "estresl.csv"  # Cambia a 'reposos.csv'o "estresl.csv" para la otra señal
df = pd.read_csv(archivo)
t = df["Tiempo (s)"]
ecg = df["ECG (V)"]

# === Graficar señal cruda
plt.figure(figsize=(12, 4))
plt.plot(t, ecg)
plt.title(f"Señal ECG cruda – {archivo}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.tight_layout()
plt.show()
# === Filtrado Butterworth 0.5–40 Hz
fs = 500  # Frecuencia de muestreo
lowcut = 0.5
highcut = 40
orden = 4

b, a = butter(orden, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
ecg_filtrada = filtfilt(b, a, ecg)

# === Graficar señal filtrada
plt.figure(figsize=(12, 4))
plt.plot(t, ecg_filtrada)
plt.title(f"Señal ECG filtrada – {archivo}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Detección de picos R
peaks, _ = find_peaks(ecg_filtrada, distance=int(0.6 * fs), height=0.4)

# === Calcular intervalos R–R
rr_intervals = np.diff(t[peaks])  # en segundos

# === Graficar señal con picos R
plt.figure(figsize=(12, 5))
plt.plot(t, ecg_filtrada, label="ECG filtrada")
plt.plot(t[peaks], ecg_filtrada[peaks], "ro", label="Picos R detectados")
plt.title(f"Detección de picos R – {archivo}")
plt.xlabel("Tiempo [s]")
plt.ylabel("Voltaje [V]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# === Graficar los intervalos R–R
plt.figure(figsize=(10, 4))
plt.plot(rr_intervals, marker='o', linestyle='-')
plt.title(f"Intervalos R–R – {archivo}")
plt.xlabel("Latido n")
plt.ylabel("Intervalo R–R [s]")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Guardar resultados
nombre_salida = "rr_" + archivo.replace(".csv", "") + ".csv"
df_rr = pd.DataFrame({"RR (s)": rr_intervals})
df_rr.to_csv(nombre_salida, index=False)

print(f"✅ Señal procesada correctamente.")
print(f"📁 Intervalos R–R guardados como: {nombre_salida}")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt

# === Cargar señal R–R
rr = pd.read_csv("rr_reposos.csv")["RR (s)"].values  # cambia por 'rr_reposoreal.csv' después

# === Transformada Wavelet Continua
wavelet = 'morl'
scales = np.arange(1, 64)  # más escalas = más detalle

coef, freqs = pywt.cwt(rr, scales, wavelet)

# === Gráfico de espectrograma
plt.figure(figsize=(12, 5))
plt.imshow(np.abs(coef), extent=[0, len(rr), freqs[-1], freqs[0]],
           cmap='jet', aspect='auto')
plt.title("Espectrograma Wavelet de la señal RR")
plt.xlabel("Latido")
plt.ylabel("Frecuencia (escala)")
plt.colorbar(label="Magnitud")
plt.tight_layout()
plt.show()

import pandas as pd
import numpy as np

# === Cargar archivo de R–R
rr_reposo = pd.read_csv("rr_reposos.csv")
rr_estrés = pd.read_csv("rr_reposoreal.csv")  # recuerda que están invertidos

# === Cálculo de métricas
media_reposo = np.mean(rr_reposo["RR (s)"])
sdnn_reposo = np.std(rr_reposo["RR (s)"])

media_estres = np.mean(rr_estrés["RR (s)"])
sdnn_estres = np.std(rr_estrés["RR (s)"])

# === Mostrar resultados
print("📊 HRV en dominio del tiempo:")
print(f"--- REPOSO ---")
print(f"Media RR: {media_reposo:.3f} s")
print(f"SDNN:     {sdnn_reposo:.3f} s")

print(f"--- ESTRÉS ---")
print(f"Media RR: {media_estres:.3f} s")
print(f"SDNN:     {sdnn_estres:.3f} s")
