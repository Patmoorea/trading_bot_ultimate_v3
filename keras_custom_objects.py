from tensorflow.keras.optimizers import legacy

custom_objects = {
    'Adam': legacy.Adam,
    'Adagrad': legacy.Adagrad,
    'RMSprop': legacy.RMSprop
}

def get_custom_objects():
    return custom_objects

if __name__ == "__main__":
    print("Custom objects disponibles:", list(get_custom_objects().keys()))
