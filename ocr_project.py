import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2 as cv
import glob
import pandas as pd

batch_size = 16
img_height = 50
img_width = 100
max_len = 7

NULL_CHAR = '<nul>'
characters = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"
characters = characters.upper()
characters = set(characters)
characters = sorted(characters)
characters.append(NULL_CHAR)

path = "./NAME/NAME/"
data_dir = Path(path)
images = []
labels = []
for file in glob.glob(path + "*"):
    images.append(file)
    file_name = os.path.basename(file)
    label = os.path.splitext(file_name)[0].split("_")[-1]
    label_padded = [NULL_CHAR] * max_len
    label_padded[:len(label)] = label
    labels.append(label_padded)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique character: ", len(characters))
print("Characters present", characters)

# def data_statistics():
#     from collections import Counter
#     temp = ''.join(labels)
#     count_chars = Counter(temp)
#     print("STATISTIC CHARACTER")
#     for c in characters:
#         if c not in count_chars:
#             count_chars[c] = 0
#     count_chars = {k: v for k, v in sorted(count_chars.items(), key=lambda item: item[1], reverse=True)}
#     for k, v in count_chars.items():
#         print("{}: {}".format(k, v))
# data_statistics()

list_img = glob.glob(os.path.join(path, "*"))

dict_img = {}
for file in list_img:
    file_name = os.path.basename(file)
    img = cv.imread(file)
    height, width = img.shape[:2]
    dict_img[file_name] = [height, width]

df = pd.DataFrame(dict_img)
df_T = df.transpose()
df_T.columns = ["Height", "Width"]
df_T["rate"] = df_T["Width"] / df_T["Height"]

df_descibe = df_T.describe()


# print("Describe of data: \n", df_descibe)

def add_padding(image, img_w=250, img_h=50):
    img = cv.imread(image)
    name = os.path.basename(image)
    hh, ww, cc = img.shape
    rate = ww / hh
    img = cv.resize(img, (round(rate * img_h), img_h), interpolation=cv.INTER_AREA)
    color = (0, 0, 0)
    result = np.full((img_h, img_w, cc), color, dtype=np.uint8)
    result[:img_h, :round(rate * img_h)] = img
    # cv.imwrite("./NAME/file_add_padding/" + name, result)
    # plt.imshow(result)
    # plt.show()


# t = 0
for i in images:
    # t += 1
    add_padding(i)
    # if t == 5:
    #     break


def split_data(img, labels, train_size=0.9):
    size = len(img)
    train_sample = int(size * train_size)
    x_train, y_train = img[:train_sample], labels[:train_sample]
    x_valid, y_valid = img[train_sample:], labels[train_sample:]
    return x_train, y_train, x_valid, y_valid


x_train, y_train, x_valid, y_valid = split_data(np.array(images), np.array(labels))
print(len(y_train))

char_to_num = layers.experimental.preprocessing.StringLookup(vocabulary=list(characters),
                                                             num_oov_indices=1, mask_token='')
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), invert=True)


def encode_single_sample(img_path, label):
    img = tf.io.read_file(img_path)

    img = tf.io.decode_png(img, channels=1)

    img = tf.image.convert_image_dtype(img, tf.float32)

    img = tf.image.resize(img, [img_height, img_width])

    img = tf.transpose(img, perm=[1, 0, 2])

    label = char_to_num(label)

    return {"image": img, "label": label}


train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
valid_dataset = (
    valid_dataset.map(encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(
        batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE))
# print(train_dataset)


_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    images = batch["image"]
    labels = batch["label"]
    for i in range(16):
        img = (images[i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(labels[i])).numpy().decode("utf-8")
        label = label.replace(NULL_CHAR, "")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()


class CTClayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred


def build_model():
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same",
                      name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same",
                      name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(62, return_sequences=True, dropout=0.25))(x)

    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)
    output = CTClayer(name="ctc_loss")(labels, x)

    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt)
    return model


model = build_model()
model.summary()

epochs = 100
early_stopping_patience = 10
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# history = model.fit(
#     train_dataset,
#     validation_data=valid_dataset,
#     epochs=epochs,
#     callbacks=[early_stopping],
# )





