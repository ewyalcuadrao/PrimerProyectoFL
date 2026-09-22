import nest_asyncio
nest_asyncio.apply()

import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

print(f'Clientes de entrenamiento EMNIST: {len(emnist_train.client_ids)}')
print(f'Estructura de los datos: {emnist_train.element_type_structure}')
