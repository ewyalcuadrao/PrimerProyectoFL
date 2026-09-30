# Federated learning para una casa IoT

Demo local de aprendizaje federado con tres hogares simulados. Cada cliente conserva un corpus de palabras derivado de sus sensores y entrena localmente un modelo Word2Vec Skip-Gram con negative sampling. El servidor agrega los embeddings mediante FedAvg; no mezcla las secuencias de los hogares.

## Estructura

```text
iot_casa_fl/
  clients/
    casa_centro/       # Configuracion y datos del cliente 1
    casa_norte/        # Configuracion y datos del cliente 2
    casa_sur/          # Configuracion y datos del cliente 3
    client.py          # Entrenamiento Word2Vec local
  server/
    server.py          # Orquestacion, FedAvg y evaluacion
    artifacts/         # Modelo global generado al ejecutar
  shared/
    model.py           # Skip-Gram, negative sampling y embeddings
    vocabulary.json    # Vocabulario comun e indices de palabras
  generate_data.py     # Generacion reproducible de corpus textuales
```

Cada casa tiene un perfil de ocupacion, clima interior y uso de aparatos distinto. Sus ficheros `train.txt` y `test.txt` contienen secuencias de palabras sinteticas, una linea por dia y un token por estado observado: por ejemplo `hora_manana`, `hogar_ocupado`, `consumo_alto`, `aparatos_activos` y `clima_encendido`. Los datos son ficticios y no describen una vivienda real.

## Ejecutar

Desde esta carpeta, con Python 3.10 o posterior:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python generate_data.py
python -m server.server
```

La semilla y la configuracion hacen reproducibles los corpus. El vocabulario es compartido para que el indice de cada palabra sea identico en todos los clientes. El servidor promedia las matrices de embeddings ponderando por el numero de pares palabra-contexto. Imprime la perdida Skip-Gram sobre los datos de prueba y palabras cercanas a algunos tokens, y guarda ambos embeddings en `server/artifacts/global_model.json`.

Para cambiar el experimento:

```bash
python -m server.server --rounds 30 --local-epochs 4 --learning-rate 0.5
```

## Datos de los sensores

Las palabras codifican franja horaria, ocupacion, nivel de consumo, actividad de aparatos, estado de climatizacion, temperatura interior/exterior, humedad e iluminacion. Word2Vec aprende que tokens aparecen en contextos parecidos; no predice una etiqueta ni calcula accuracy.

Esta es una simulacion didactica de FedAvg en un solo proceso, no un despliegue distribuido ni una garantia de privacidad diferencial. El codigo del cliente solo devuelve pesos y numero de ejemplos al agregador.

## Demo de red NDN y Word2Vec federado

La carpeta `ndn_sim/` contiene una segunda demo independiente del experimento IoT:

```text
edge_centro  --\
edge_norte   ----> core ----> productores de video, guia, noticias y clima
edge_sur     --/       |
                       +--> FIB por prefijos agregados
coordinador FL <------ actualizaciones Word2Vec de los tres edges
```

Los routers de borde guardan sus sesiones de Interests localmente. El core usa
Longest Prefix Match: por ejemplo, `/ndn/video/canal1` es una entrada de FIB y
puede reenviar `/ndn/video/canal1/segment/001`; no se instala una entrada para
cada Interest exacto. La simulacion incluye forwarding Interest/Data, PIT y
Content Store. Cada nombre completo de Interest es un token Word2Vec, y cada
sesion local es una secuencia; por eso el modelo mide co-solicitudes cercanas,
no similitud semantica entre componentes del nombre.

Ejecuta desde esta carpeta despues de instalar `requirements.txt`:

```bash
python -m ndn_sim
python -m unittest discover -s tests -v
```

El comando muestra la FIB, rutas de ejemplo, las 20 rondas FedAvg y los
Interests asociados aprendidos; guarda los embeddings en
`server/artifacts/ndn_global_model.json`. Es una simulacion local simplificada,
no implementa paquetes NDN reales ni reemplaza ndnSIM/NFD. El vocabulario de
la demo es conocido y comun a los clientes para permitir la agregacion.

Al final compara 100 solicitudes: 80 nombres conocidos y 20 Interests OOV.
Diez OOV tienen un destino correcto inducible por su contexto, cinco tienen
contexto parecido pero no un destino (casos para medir falsos positivos) y
cinco no tienen contexto conocido. Word2Vec solo propone un nombre si su
similitud coseno es al menos `0.30`; luego la FIB resuelve ese nombre por
prefijo. La tabla muestra entregas correctas, falsos positivos y solicitudes
perdidas, y el detalle se guarda en
`server/artifacts/ndn_100_interest_comparison.csv`.

Tambien se imprimen los tres vecinos mas cercanos de cada Interest en el modelo
global, ordenados por similitud coseno. El listado completo se guarda en
`server/artifacts/ndn_global_relations.csv`.