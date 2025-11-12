# train_clothes.py
# Usage:
#   # 1) Real images organized as data/<class>/*.jpg
#   python train_clothes.py --data_dir data --out_dir export --img_size 224 --epochs 12
#
#   # 2) Quick toy run with Fashion-MNIST
#   python train_clothes.py --use_fashion_mnist --out_dir export --epochs 5
#
# Requirements: tensorflow>=2.11, tensorflow-datasets (optional), pillow
# For export only on server/RPi: use tflite_runtime there (no need to train on Pi).

import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default=None,
                    help="Directory with class subfolders: data/<class>/*.jpg")
    ap.add_argument("--use_fashion_mnist", action="store_true",
                    help="Use Fashion-MNIST (toy) instead of a folder dataset")
    ap.add_argument("--out_dir", type=str, default="export",
                    help="Where to save models and artifacts")
    ap.add_argument("--img_size", type=int, default=224, help="Square image size")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--freeze_base", action="store_true",
                    help="Freeze backbone (for very small datasets)")
    return ap.parse_args()

def build_from_folder(data_dir, img_size, batch_size, val_split):
    # tf.keras.utils.image_dataset_from_directory expects class subfolders
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="training",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )
    class_names = train_ds.class_names
    return train_ds, val_ds, class_names

def build_from_fashion_mnist(img_size, batch_size, val_split):
    (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.fashion_mnist.load_data()
    x = np.concatenate([x_tr, x_te], axis=0)   # ~55MB, fine
    y = np.concatenate([y_tr, y_te], axis=0)

    # Stratified split to keep all 10 classes in both sets
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_split, random_state=42, stratify=y
    )

    def _prep(x, y):
        x = tf.expand_dims(x, -1)                    # (28,28,1)
        x = tf.image.resize(x, (img_size, img_size)) # (H,W,1)
        x = tf.image.grayscale_to_rgb(x)             # (H,W,3)
        x = tf.cast(x, tf.float32) / 255.0
        return x, y

    train_ds = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                .shuffle(8192)
                .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
                .batch(batch_size)
                .prefetch(tf.data.AUTOTUNE))

    val_ds = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
              .map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(batch_size)
              .prefetch(tf.data.AUTOTUNE))

    class_names = [
        "t-shirt_top","trouser","pullover","dress","coat",
        "sandal","shirt","sneaker","bag","ankle_boot"
    ]
    return train_ds, val_ds, class_names

def add_perf(ds, training=False):
    # Data augmentation + standardization
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    # Map to [0,1] and augment only on training
    def _map_train(x, y):
        x = tf.cast(x, tf.float32)
        if tf.reduce_max(x) > 1.1:  # if 0..255 -> normalize
            x = x / 255.0
        x = aug(x)
        return x, y
    def _map_val(x, y):
        x = tf.cast(x, tf.float32)
        if tf.reduce_max(x) > 1.1:
            x = x / 255.0
        return x, y

    if training:
        ds = ds.map(_map_train, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(_map_val, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def build_model(img_size, num_classes, freeze_base=False):
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
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model

def export_labels(out_dir, class_names):
    labels_path = Path(out_dir) / "labels.txt"
    with open(labels_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(f"{name}\n")
    return str(labels_path)

def export_saved_model(model, out_dir):
    sm_dir = Path(out_dir) / "saved_model"
    tf.saved_model.save(model, str(sm_dir))
    return str(sm_dir)

def representative_dataset_gen(dataset, sample_count=300):
    # Take a few hundred images from the *training* pipeline for int8 calibration
    it = iter(dataset.unbatch().take(sample_count).batch(1))
    for x, _ in it:
        x = tf.cast(x, tf.float32)
        if tf.reduce_max(x) > 1.1:
            x = x / 255.0
        yield [x]

def export_tflite(model, out_dir, rep_ds=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Float32 -> dynamic range INT8 (weights), widely compatible & small
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_bytes = conv.convert()
    p1 = out_dir / "model_float32.tflite"
    p1.write_bytes(tflite_bytes)

    # 2) Float16 quantization (weights in fp16, often fastest on desktop GPUs/CPUs)
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_bytes = conv.convert()
    p2 = out_dir / "model_fp16.tflite"
    p2.write_bytes(tflite_bytes)

    # 3) Full **int8** (activations+weights) for embedded/CPU; needs representative data
    p3 = None
    if rep_ds is not None:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
        conv.representative_dataset = lambda: representative_dataset_gen(rep_ds)
        # If you need full-integer (no float ops), uncomment next line:
        # conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # conv.inference_input_type = tf.int8
        # conv.inference_output_type = tf.int8
        tflite_bytes = conv.convert()
        p3 = out_dir / "model_int8.tflite"
        p3.write_bytes(tflite_bytes)

    return str(p1), str(p2), (str(p3) if p3 else None)

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_fashion_mnist:
        train_ds, val_ds, class_names = build_from_fashion_mnist(
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_split=args.val_split
        )
    else:
        if not args.data_dir:
            raise SystemExit("When not using --use_fashion_mnist, you must pass --data_dir pointing to class folders.")
        train_ds, val_ds, class_names = build_from_folder(
            data_dir=args.data_dir,
            img_size=args.img_size,
            batch_size=args.batch_size,
            val_split=args.val_split
        )

    train_ds = add_perf(train_ds, training=True)
    val_ds   = add_perf(val_ds, training=False)

    model = build_model(args.img_size, num_classes=len(class_names), freeze_base=args.freeze_base)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "ckpt_best.keras"),
            monitor="val_acc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_acc", mode="max", patience=5, restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Export artifacts
    labels_path = export_labels(out_dir, class_names)
    saved_model_dir = export_saved_model(model, out_dir)
    p_float32, p_fp16, p_int8 = export_tflite(model, out_dir, rep_ds=train_ds)

    # Write a small README for deployment
    with open(out_dir / "README.txt", "w", encoding="utf-8") as f:
        f.write(
            "Artifacts:\n"
            f"- SavedModel: {saved_model_dir}\n"
            f"- TFLite (float32): {p_float32}\n"
            f"- TFLite (fp16):    {p_fp16}\n"
            f"- TFLite (int8):    {p_int8}\n"
            f"- labels.txt:       {labels_path}\n\n"
            "Inference tip (Python + TFLite): load labels.txt, resize to IMG_SIZE,"
            " normalize to [0,1], run interpreter, pick top-k.\n"
        )

    print("\nTraining done.")
    print(f"Classes: {class_names}")
    print(f"Exported to: {out_dir.resolve()}")

if __name__ == "__main__":
    # Quiet TF logs a bit
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()

