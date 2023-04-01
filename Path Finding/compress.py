from PIL import Image as img
import glob

def compressImage(cnt = 0):

    for image in glob.iglob('E://tempImageStorage/*'):
        cnt +=1
        testImage = img.open(image)
        width, height = testImage.size
        
        resolutionOrig = width * height
        resolutionNew = 5000
        scaler =  (resolutionNew / resolutionOrig) ** 0.5
        resizedImage = testImage.resize(((int(width * scaler)), (int(height * scaler))))
        width, height = resizedImage.size
        resizedImage.save(image)