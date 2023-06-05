import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf


def predict():

    # Create a list with the filepaths for training and testing
    train_dir = Path('dataset/train')
    train_filepaths = list(train_dir.glob(r'**/*.jpg'))

    test_dir = Path('../media/photos')
    test_filepaths = list(test_dir.glob(r'**/*.jpg'))

    val_dir = Path('dataset/validation')
    val_filepaths = list(test_dir.glob(r'**/*.jpg'))

    def proc_img(filepath):
        """ Create a DataFrame with the filepath and the labels of the pictures
        """
        # print(len(filepath))
        labels = [str(filepath[i]).split("\\")[-2] for i in range(len(filepath))]

        filepath = pd.Series(filepath, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        # Concatenate filepaths and labels
        df = pd.concat([filepath, labels], axis=1)

        # Shuffle the DataFrame and reset index
        df = df.sample(frac=1).reset_index(drop=True)

        return df


    train_df = proc_img(train_filepaths)
    test_df = proc_img(test_filepaths)
    # val_df = proc_img(val_filepaths)

    # print('-- Training set --\n')
    # print(f'Number of pictures: {train_df.shape[0]}\n')
    # print(f'Number of different labels: {len(train_df.Label.unique())}\n')
    # print(f'Labels: {train_df.Label.unique()}')

    # Create a DataFrame with one Label of each category
    # df_unique = train_df.copy().drop_duplicates(subset=["Label"]).reset_index()


    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    )

    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # val_images = train_generator.flow_from_dataframe(
    #     dataframe=val_df,
    #     x_col='Filepath',
    #     y_col='Label',
    #     target_size=(224, 224),
    #     color_mode='rgb',
    #     class_mode='categorical',
    #     batch_size=32,
    #     shuffle=True,
    #     seed=0,
    #     rotation_range=30,
    #     zoom_range=0.15,
    #     width_shift_range=0.2,
    #     height_shift_range=0.2,
    #     shear_range=0.15,
    #     horizontal_flip=True,
    #     fill_mode="nearest"
    # )

    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # Load the pretained model
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    pretrained_model.trainable = False

    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(128, activation='relu')(pretrained_model.output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(36, activation='softmax')(x)

    loaded_model = tf.keras.models.load_model('my_model')

    # Predict the label of the test_images
    predictions = loaded_model.predict(test_images)
    predictions = np.argmax(predictions, axis=1)
    # Map the label
    # print(len(predictions))
    # print(train_images.class_indices)
    labels = (train_images.class_indices)
    labels = dict((v, k) for k, v in labels.items())
    # print(labels)
    pred = [labels[k] for k in predictions]
    # print(predictions)
    # print(pred)

    return pred

