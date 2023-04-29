import matplotlib.pyplot as plt
import numpy as np  #pentru prelucrarea cu matrice si vectori
import PIL   #pentru prelucrarea de imagini
import tensorflow as tf

from tensorflow import keras  #pentru construirea si antrenarea retelelor neuronale
from keras import layers    #straturile de retele neuronale
from keras.models import Sequential #pentru definirea modelul CNN
import pathlib

def preprocesare_setDate():
    setul_de_date = pathlib.Path("./data")  #calea catre locul unde se afla datele

    nrTotal_CT = len(list(setul_de_date.glob("*/*.png")))   #aflam cate imagini sunt in folder
    print(nrTotal_CT)

    img_h =img_w = 224   #setam dimensiunea imaginilor la 224x224 pixeli
    batch = 32  #numarul imaginilor procesate la un pas

    #incarcarea datelor si crearea seturilor pentru antrenament si validare
    set_date_antrenare = tf.keras.utils.image_dataset_from_directory(
        setul_de_date,
        validation_split=0.2,
        subset="training",
        seed=1,
        image_size=(img_h, img_w),
        batch_size=batch)

    set_date_validare = tf.keras.utils.image_dataset_from_directory(
    setul_de_date,
    validation_split=0.2,
    subset="validation",
    seed=1,
    image_size=(img_h, img_w),
    batch_size=batch)
    
    clase_clasificare = set_date_antrenare.class_names
    
    #optimizarea performantei incarcarii datelor pentru antrenare si validare
    AUTOTUNE = tf.data.AUTOTUNE 
    set_date_antrenare = set_date_antrenare.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    set_date_validare = set_date_validare.cache().prefetch(buffer_size=AUTOTUNE)

    #normalizarea datelor de intrare pentru reteaua neuronala ca valorile pixelilor sa se incadreze intre 0 si 1
    layer_normalizare = layers.Rescaling(1./255)    

    #augmentarea datelor pt diversificare
    augmentarea_datelor = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1)
        ]
    )
    return (layer_normalizare,augmentarea_datelor,clase_clasificare,set_date_antrenare,set_date_validare)


def definirea_modelului(clase_clasificare,layer_normalizare,augmentarea_datelor):
    numar_clase = len(clase_clasificare)
    l2_lambda = 0.0001

    #definirea modelului
    model = Sequential([
    layer_normalizare,
        augmentarea_datelor,
        layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(numar_clase, kernel_regularizer=tf.keras.regularizers.l2(l2_lambda))
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    return model

def antrenarea(model,set_date_antrenare,set_date_validare):
    #antrenarea modelul in 20 de epoci
    epochs = 20
    history = model.fit(
        set_date_antrenare,
        validation_data=set_date_validare,
        epochs=epochs,
    )

    acuratete_antrenare = history.history['accuracy']
    acuratete_validare = history.history['val_accuracy']

    loss_antrenare = history.history['loss']
    loss_validare = history.history['val_loss']

    epochs_range = range(epochs)
    model.summary()

def predit_si_creareSubmisie(model,clase_clasificare):
    
    #prezicerea claselor pentru datele de test si salvarea pentru submisie
    with open ('./predictions.csv', 'w') as f:
        f.write("id,class\n")
        for id in range(17001, 22150):
            img = tf.keras.utils.load_img(
                "./predict/0" + str(id) + ".png", 
                target_size=(224, 224)
            )
            
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0]) #normalizarea matricea de probabilitati astfel incat fiecare element sa fie intre 0 si 1

            if clase_clasificare[np.argmax(score)] == "normal":
                f.write("0" + str(id) + ",0\n")
            else:
                f.write("0" + str(id) + ",1\n")
 
#Brain Anomaly Detection   
layer_normalizare,augmentarea_datelor,clase_clasificare,set_date_antrenare,set_date_validare =preprocesare_setDate()
model = definirea_modelului(clase_clasificare,layer_normalizare,augmentarea_datelor)
antrenarea(model,set_date_antrenare,set_date_validare)
predit_si_creareSubmisie(model,clase_clasificare)