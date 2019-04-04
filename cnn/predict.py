from keras.models import load_model
import matplotlib.image as processimage
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class Prediction(object):
    def __init__(self,ModelFile,PredictFile,Type,Width=100,Height=100):
        self.modelfile = ModelFile
        self.predict_file = PredictFile
        self.Width = Width
        self.Height = Height
        self.Type = Type

    def Predict(self):
        model = load_model(self.modelfile)
        img_open = Image.open(self.predict_file)
        conv_RGB = img_open.convert('RGB')
        new_img = conv_RGB.resize((self.Width,self.Height),Image.BILINEAR)
        new_img.save(self.predict_file)
        print ('Image Processed')

        image = processimage.imread(self.predict_file)
        image_to_array = np.array(image)/255.0
        image_to_array = image_to_array.reshape(-1,100,100,3)
        print ('Image reshaped')

        prediction = model.predict(image_to_array)
        print (prediction)
        Final_prediction = [result.argmax() for result in prediction][0]
        print (Final_prediction)

        #延伸教程
        count = 0
        for i in prediction[0]:
            print (i)
            percentage = '%.2f%%' % (i * 100)
            print (self.Type[count],'概率:' ,percentage)
            count +=1


    def ShowPredImg(self):
        image = processimage.imread(self.predict_file)
        plt.imshow(image)
        plt.show()


Type = ['dog', 'cat']
Pred = Prediction(PredictFile='download/cat1.jpeg',ModelFile='catvsdogfinder.h5',Width=100,Height=100,Type=Type)
Pred.Predict()
