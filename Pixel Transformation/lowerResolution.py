
''' lowerResolution.py '''


from PIL import Image as img
import glob


def compressImage(cnt = 0):

    for image in glob.iglob('D://VALIDATION/*'):
        cnt +=1
        testImage = img.open(image)
        width, height = testImage.size
        
        resolutionOrig = width * height
        resolutionNew = 500000
        scaler =  (resolutionNew / resolutionOrig) ** 0.5
        resizedImage = testImage.resize(((int(width * scaler)), (int(height * scaler))))
        width, height = resizedImage.size
        print (width, height)
        resizedImage.save('D:\\COMPRESSED_VALIDATION{image}'.format(image=(image.strip('D://VALIDATION'))))


def main():
    compressImage()

if __name__ == '__main__':
    main()
