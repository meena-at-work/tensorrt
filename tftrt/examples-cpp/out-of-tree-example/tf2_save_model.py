# A sample TF2 file that creates a simple model, and saves it to disk.

import copy
import os
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

if __name__ == "__main__":
    ##################### download data #####################
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train/255.0
    x_test = x_test/255.0

    x_train = x_train.reshape(list(x_train.shape) + [1])
    x_test = x_test.reshape(list(x_test.shape) + [1])

    ##################### create a simple keras model #####################
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), padding="valid", activation="relu", input_shape=(28,28,1)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), padding="valid", activation="relu"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), padding="valid", activation="relu"),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    # define the loss function
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # compile and fit the model to the given data
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])

    ##################### save the model #####################
    model_path = "toy_model"
    #checkpoint_path = os.path.join(model_path)
    checkpoint_dir = os.path.dirname(model_path)
    print(checkpoint_dir)

    model.fit(x_train, y_train, epochs=1)

    model.save(model_path)  # save the final SavedModel for later use by TF-TRT

    # evaluate model accuracy
    model.evaluate(x_test, y_test, verbose=2)

    ###################### convert model to tftrt ######################
    params = copy.deepcopy(trt.DEFAULT_TRT_CONVERSION_PARAMS)
    params = params._replace(
        precision_mode=trt.TrtPrecisionMode.FP16,
        minimum_segment_size=3,
        allow_build_at_runtime=True  # allow to build on the final machine instead of the machine that converts the model
    )

    import pprint
    print("%" * 85)
    pprint.pprint(params)
    print("%" * 85)

    converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=model_path,
        conversion_params=params,
    )
    
    converter.convert()
    converter.save(
        os.path.join(model_path, "converted")
    )
