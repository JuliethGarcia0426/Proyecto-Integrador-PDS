# Proyecto-Integrador-PDS
## *Fundamento teórico*
Actividad simpática y parasimpática del sistema nervioso autónomo; El sistema nervioso autónomo: 
regula funciones involuntarias del cuerpo, como el ritmo cardíaco, la digestión y la respiración. Se divide en:
### Sistema simpático: 
Prepara al cuerpo para situaciones de estrés ("respuesta de lucha o huida"). Aumenta la frecuencia cardiaca, la presión arterial y el flujo sanguíneo hacia los músculos.
### Sistema parasimpático:
Favorece el estado de reposo y recuperación ("respuesta de reposo y digestión"). Disminuye la frecuencia cardiaca, promueve la digestión y la conservación de energía.
### *Efecto de la actividad simpática y parasimpática en la frecuencia cardiaca*

### Actividad simpática:
- Aumenta la frecuencia cardiaca (taquicardia).
- Disminuye el intervalo R-R (el tiempo entre latidos consecutivos se acorta).

### Actividad parasimpática:
- Disminuye la frecuencia cardiaca (bradicardia).
- Aumenta el intervalo R-R (el tiempo entre latidos consecutivos se alarga).
- Estos efectos permiten que la frecuencia cardiaca se ajuste rápidamente a cambios internos o externos, como el ejercicio o el descanso.

### Variabilidad de la frecuencia cardiaca (HRV)
La HRV mide las fluctuaciones en los intervalos de tiempo entre latidos consecutivos del corazón (intervalo R-R).

###Frecuencias de interés en análisis HRV:
- ULF (Ultra Low Frequency): < 0.003 Hz
- VLF (Very Low Frequency): 0.003 – 0.04 Hz
- LF (Low Frequency): 0.04 – 0.15 Hz (actividad combinada simpática y parasimpática)
- HF (High Frequency): 0.15 – 0.4 Hz (actividad parasimpática)

### Transformada Wavelet
La Transformada Wavelet es una herramienta matemática que descompone una señal en componentes de diferentes escalas o resoluciones, permitiendo analizar cambios tanto en el tiempo como en la frecuencia.

### Usos en señales biológicas:

- Análisis de señales no estacionarias como el ECG, EMG, HRV.
- Detección de cambios rápidos en la señal, como eventos cardíacos súbitos.
- Tipos de wavelets más comunes en señales biológicas:
- Daubechies (db4, db6, etc.): Muy usada para ECG y HRV por su forma parecida a las ondas del ECG.
- Symlets: Versión más simétrica de las Daubechies.
- Coiflets: Alta regularidad, usadas en señales con características suaves.
# Diagrama de flujo
![image](https://github.com/user-attachments/assets/dc0d8876-ce0f-44f5-81de-bb38d50e4947)

## toma de datos 
las acciones realizadas por el paciente para la toma de los datos fue:
- Sistema parasimpatico: relacion total, hasta casi dormir
- Sistema simpatico: Aguantar la resíracion y jugar en el celular
### codigo de recoleccion de datos:
```python
import numpy as np
import pandas as pd
import PyDAQmx as nidaq
from PyDAQmx import Task

class AnalogInput(Task):
    def _init_(self, channel="Dev3/ai0", rate=500.0, duration=300):
        Task._init_(self)
        self.rate = rate
        self.duration = duration
        self.samples = int(rate * duration)
        self.data = np.zeros((self.samples,), dtype=np.float64)

        # Configurar el canal analógico
        self.CreateAIVoltageChan(channel, "", nidaq.DAQmx_Val_Cfg_Default,
                                 -10.0, 10.0, nidaq.DAQmx_Val_Volts, None)
        self.CfgSampClkTiming("", rate, nidaq.DAQmx_Val_Rising,
                              nidaq.DAQmx_Val_FiniteSamps, self.samples)
```
## Codigo 
*Librerías*
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pywt
```
- pandas: manejar archivos .csv y estructuras tipo tabla.
- numpy: operaciones matemáticas y estadísticas.
- matplotlib.pyplot: graficar señales.
- scipy.signal: crear y aplicar filtros digitales, detectar picos.
- pywt: aplicar la transformada wavelet para analizar la señal en tiempo-frecuencia.

*Configuración inicial*
```python
archivo_reposo = "reposo.csv"
archivo_estres = "es.csv"
Se definen las rutas de los archivos CSV con las señales de ECG capturadas en dos condiciones: reposo y estrés.
```
*Definición de funciones*
Carga de la señal
```python
def cargar_ecg(nombre_archivo):
    df = pd.read_csv(nombre_archivo)
    return df["Tiempo (s)"], df["ECG (V)"]
```
Carga el archivo .csv y extrae las columnas de tiempo y voltaje del ECG.

*Filtrado de la señal*
```python
def filtrar_ecg(ecg, fs=500, lowcut=0.5, highcut=40, orden=4):
    b, a = butter(orden, [lowcut / (fs / 2), highcut / (fs / 2)], btype='band')
    return filtfilt(b, a, ecg)
```
filtro digital IIR pasabanda Butterworth, que elimina el ruido fuera del rango [0.5–40 Hz], filtfilt aplica el filtro hacia adelante y hacia atrás (sin desfase).

![image](https://github.com/user-attachments/assets/314b7e72-1dbe-41c3-9fe3-55e6389c49bc)

![image](https://github.com/user-attachments/assets/1b851f7f-3b11-4d0a-8030-d337efbcf0b3)

*Detección de picos R e intervalos R-R*
```python
def detectar_picos(ecg_filtrada, t, fs=500):
    peaks, _ = find_peaks(ecg_filtrada, distance=int(0.6 * fs), height=0.4)
    rr = np.diff(t[peaks])
    return peaks, rr
```
Detecta los picos R de la señal ECG y calcula los intervalos R-R (tiempo entre latidos consecutivos), que son la base para analizar la HRV.

![image](https://github.com/user-attachments/assets/5ef7cf53-9125-4191-b28b-1536ba96e56a)

![image](https://github.com/user-attachments/assets/7260b843-2e12-496a-b56a-407d312bdc5a)

![image](https://github.com/user-attachments/assets/20c2ece5-6c40-49e9-abc3-d5b5c2ee38b9)

![image](https://github.com/user-attachments/assets/bd1196a9-9f22-4253-8a6c-7367218b928a)

*Visualización completa del procesamiento*
```python
def graficar_todo(t, ecg, ecg_filtrada, peaks, rr, nombre):
```
Esta función muestra 4 gráficos por condición:
ECG cruda.
ECG filtrada.
Detección de picos R.
Serie de intervalos R-R.
Sirve para validar visualmente cada paso del procesamiento.

*Cálculo de parámetros de HRV*
```python
def calcular_estadisticas(rr, nombre):
```
Calcula:
- Media RR: promedio del intervalo R–R.
- SDNN: desviación estándar de los intervalos R–R.
- Ambos son indicadores de la variabilidad cardíaca en el dominio del tiempo.

*Transformada Wavelet*
```python
def wavelet_rr(rr, nombre):
```
Aplica la transformada wavelet continua (CWT) usando la wavelet Morlet ('morl'), ideal para señales biológicas como ECG. El resultado es un espectrograma que muestra cómo cambia la frecuencia a lo largo del tiempo.

*Procesamiento de las señales*

📌 Reposo
```python
t1, ecg1 = cargar_ecg(archivo_reposo)
ecgl1_filt = filtrar_ecg(ecg1)
peaks1, rr1 = detectar_picos(ecgl1_filt, t1)
graficar_todo(t1, ecg1, ecgl1_filt, peaks1, rr1, archivo_reposo)
media1, sdnn1 = calcular_estadisticas(rr1, "reposo")
wavelet_rr(rr1, "reposo")
```
Secuencia completa del análisis de la señal en reposo.

![image](https://github.com/user-attachments/assets/f33c6f3b-4560-4ed1-b382-54ac030738fe)

📌 Estrés
```python
t2, ecg2 = cargar_ecg(archivo_estres)
ecgl2_filt = filtrar_ecg(ecg2)
peaks2, rr2 = detectar_picos(ecgl2_filt, t2)
graficar_todo(t2, ecg2, ecgl2_filt, peaks2, rr2, archivo_estres)
media2, sdnn2 = calcular_estadisticas(rr2, "estrés")
wavelet_rr(rr2, "estrés")
```
Secuencia completa del análisis de la señal en estrés.

![image](https://github.com/user-attachments/assets/b16e4c7f-3de6-4460-87fd-ce34068b1b08)

*Comparación final entre condiciones*
```python
comparativo = pd.DataFrame({
    "Condición": ["Reposo", "Estrés"],
    "Media RR (s)": [media1, media2],
    "SDNN (s)": [sdnn1, sdnn2]
})

print("\n📋 Resultados comparativos HRV:\n")
print(comparativo)
```
![image](https://github.com/user-attachments/assets/53973764-ddf2-4491-933d-313d7e404ff4)

### Analisis:

Se ve una actividad espectral bien distribuida en todo el rango de escalas (especialmente entre 0.15 y 0.4, correspondientes a la banda HF), Las oscilaciones en la banda HF son más frecuentes y de mayor magnitud en reposo, indicando una mayor influencia parasimpática; Esto quiere decir que en condiciones de reposo, el sistema nervioso autónomo está dominado por la actividad parasimpática, la cual regula la frecuencia cardiaca con variabilidad natural. Esto se refleja en una HRV alta y mayor potencia en la banda HF del espectrograma.

![image](https://github.com/user-attachments/assets/5689be3c-58ae-4aed-94c9-b387ee8f3612)

### Analisis:

La potencia espectral se concentra más en las primeras escalas (frecuencias más bajas) al inicio del registro. Hay una reducción general de la magnitud espectral, especialmente en la banda HF, lo cual indica una disminución de la variabilidad.
Las zonas más intensas se ubican en la región de LF (0.04–0.15), y la actividad HF se reduce de forma notable. Esto quierede decir  que a Bajo estrés, el cuerpo activa el sistema simpático, reduciendo la variabilidad natural de la frecuencia cardiaca y esta activación produce una HRV baja con predominancia en la banda LF y atenuación en la HF, lo que se observa en el espectrograma.
