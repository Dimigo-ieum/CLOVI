import numpy as np
import tensorflow as tf


# === PARAMS ===
img_size = 224
batch_size = 32
val_split = 0.1
freeze_base = False
epochs = 5


# === LOAD & SHUFFLE DATA ===
(data_train, labels_train), (data_test, labels_test) = tf.keras.datasets.fashion_mnist.load_data()
class_names = [
    "t-shirt_top","trouser","pullover","dress","coat",
    "sandal","shirt","sneaker","bag","ankle_boot"
]

"""
print("data train len: ", len(data_train), sep=" ")     # 40000
print("labels train len: ", len(labels_train), sep=" ") # 40000
print(data_train.shape)                                 # (60000, 28, 28) why?
print(labels_train)                                     # array of integer 0-9
"""
data_train = data_train[..., None]
data_test = data_test[..., None]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

ds_train = tf.data.Dataset.from_tensor_slices((data_train, labels_train))
ds_train = ds_train.shuffle(10000, reshuffle_each_iteration=True)
ds_val   = tf.data.Dataset.from_tensor_slices((data_test, labels_test))
ds_val   = ds_val.shuffle(10000, reshuffle_each_iteration=True)

# === Augmentation ===
# NOT standardization(mobilenet will do everything)
aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])
def _map_data(data, label):
    data = tf.cast(data, tf.float32)
    # keep 0..255 uint8 or just cast, but NO /255 (double standardization)
    """
    if tf.reduce_max(data) > 1.1:  # if 0..255 -> normalize
        data = data / 255.0
    """
    data = aug(data)
    return data, label
ds_train = ds_train.map(_map_data, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

def _map_val(data, label):
    data = tf.cast(data, tf.float32)
    return data, label
ds_val = ds_val.map(_map_val, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.prefetch(tf.data.AUTOTUNE)

# === MODEL BUILDING ===
base = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights="imagenet"
)
if freeze_base:
    base.trainable = False

inputs = tf.keras.Input(shape=(img_size, img_size, 3))

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

# === FIT (moment of truth) ===
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "ckpt_best.tf"),  # <- no .keras
        monitor="val_acc",
        mode="max",
        save_best_only=True,
        save_format="tf",                        # <- force TF format
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_acc", mode="max", patience=5, restore_best_weights=True
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)


sm_dir = Path("export") / "saved_model"
tf.saved_model.save(model, str(sm_dir))

out_dir = Path("export")
out_dir.mkdir(parents=True, exist_ok=True)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_bytes = converter.convert()
path = out_dir / "model.tflite"
path.write_bytes(tflite_bytes)

return str(path)

print("\nTraining done.")
print(f"Classes: {class_names}")
print(f"Exported to: {out_dir.resolve()}")

