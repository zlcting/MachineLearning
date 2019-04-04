import os
import numpy as np
from PIL import Image
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPooling2D,Dense,Activation
from keras.optimizers import Adam
from keras.utils import np_utils

#Pre process images
class PreFile(object):
    def __init__(self,FilePath):
        self.FilePath = FilePath

    def FileResize(self,Width,Height,Output_folder):
            files = os.listdir(self.FilePath)
            for i in files:
                img_open = Image.open(self.FilePath +'/' + i)
                conv_RGB = img_open.convert('RGB') #统一转换一下RGB格式 统一化
                new_img = conv_RGB.resize((Width,Height),Image.BILINEAR)
                new_img.save(os.path.join(Output_folder,os.path.basename(i)))


class Training(object):
    def __init__(self,batch_size,number_batch,categories,train_folder):
        self.batch_size = batch_size
        self.number_batch = number_batch
        self.categories = categories
        self.train_folder = train_folder

    #读取图片返回一个numpy数组
    def read_train_images(self,filename):
        img = Image.open(self.train_folder+filename)
        return np.array(img)

    def train(self):
        train_img_list = []
        train_label_list = []
        for file in os.listdir(self.train_folder):
            files_img_in_array =  self.read_train_images(filename=file)
            train_img_list.append(files_img_in_array) #Image list 
            if(file.split('.')[0] == 'cat'):
                input_type = 1
            else:
                input_type = 0
            
            train_label_list.append(int(input_type)) #lable list 

        train_img_list = np.array(train_img_list)
        train_label_list = np.array(train_label_list)
        train_label_list = np_utils.to_categorical(train_label_list,self.categories) #format into binary [0,0,0,0,1,0,0]

        train_img_list = train_img_list.astype('float32')
        train_img_list /= 255

        #-- setup Neural network CNN
        model = Sequential()
        #CNN Layer - 1
        model.add(Convolution2D(
            filters=32, #Output for next later layer output (100,100,32)
            kernel_size= (5,5) , #size of each filter in pixel
            padding= 'same', #边距处理方法 padding method
            input_shape=(100,100,3) , #input shape ** channel last(TensorFlow)
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(
            pool_size=(2,2), #Output for next layer (50,50,32)
            strides=(2,2),
            padding='same',
        ))

        #CNN Layer - 2
        model.add(Convolution2D(
            filters=64,  #Output for next layer (50,50,64)
            kernel_size=(2,2),
            padding='same',
        ))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(  #Output for next layer (25,25,64)
            pool_size=(2,2),
            strides=(2,2),
            padding='same',
        ))

    #Fully connected Layer -1
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
    # Fully connected Layer -2
        model.add(Dense(512))
        model.add(Activation('relu'))
    # Fully connected Layer -3
        model.add(Dense(256))
        model.add(Activation('relu'))
    # Fully connected Layer -4
        model.add(Dense(self.categories))
        model.add(Activation('softmax'))
    # Define Optimizer
        adam = Adam(lr = 0.0001)
    #Compile the model
        model.compile(optimizer=adam,
                      loss="categorical_crossentropy",
                      metrics=['accuracy']
                      )
    # Fire up the network
        model.fit(
            train_img_list,
            train_label_list,
            epochs=self.number_batch,
            batch_size=self.batch_size,
            verbose=1,
        )
        #SAVE your work -model
        model.save('./catvsdogfinder.h5')




def MAIN():

    #****FILE Pre processing****
    #FILE = PreFile(FilePath='Raw_Img/')

    #****FILE Rename and Resize****
    #FILE.FileReName()
    #FILE.FileResize(Height=100,Width=100,Output_folder='train/')

    #Trainning Neural Network
    #type 1=>cat 2=>dog

    Train = Training(batch_size=128,number_batch=30,categories=2,train_folder='train/')
    Train.train()

if __name__ == "__main__":

    MAIN()