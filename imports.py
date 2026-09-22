import nest_asyncio
nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

print(f'Clientes de entrenamiento EMNIST: {len(emnist_train.client_ids)}')
print(f'Estructura de los datos: {emnist_train.element_type_structure}')

example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = next(iter(example_dataset))

print(f'Etiqueta del ejemplo: {example_element["label"].numpy()}')

output_path = Path(__file__).with_name('emnist_example.png')
plt.imshow(example_element['pixels'].numpy(), cmap='gray', aspect='equal')
plt.grid(False)
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.close()
print(f'Imagen guardada en: {output_path}')
