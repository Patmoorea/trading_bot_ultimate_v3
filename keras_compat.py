from tensorflow.keras import backend as K
if not hasattr(K, 'batch_outputs'):
    K.batch_outputs = []
