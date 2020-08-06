
#%%
#import libraries
import numpy as np 
import matplotlib.pyplot as plt
import cv2
import os

#%%
DATADIR = "D:/Python/nn/cat-or-dog/PetImages"
CATEGORIES = ["Cat","Dog"]
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR , category) #this is the path to cats or dog
        category_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                images = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_images= cv2.resize(images,(60,60))
                training_data.append([new_images,category_num])
            except Exception as e:
                pass

create_training_data()

#plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))


# %%
print(len(training_data))

# %%
import random
random.shuffle(training_data)

# %%
X=[]
y=[]

for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, 60, 60, 1)


# %%
import pickle
pickle_out =open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()

# To read pickle
# pickle_in=open("X.pickle","rb")
# X=pickle.load(pickle_in)


# %%
