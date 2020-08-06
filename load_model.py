#%%
import cv2
import tensorflow as tf 

CATEGORIES = ["Cat","Dog"]


def prepare(filepath):
    IMG_SIZE=60
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)


model=tf.keras.models.load_model("final_model")

prediction = model.predict([prepare('dog.PNG')])


print(CATEGORIES[int(prediction[0][0])])


# %%
