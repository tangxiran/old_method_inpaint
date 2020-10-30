import cv2

if __name__ == '__main__':
    file = 'data//0001_01_0'
    # for i in range(1,100+1,1):
    file = file + str(0) + '.png'
    file = 'data//22.png'
    pic = cv2.imread(file)
    cv2.imshow("origin", pic)
    dst = cv2.fastNlMeansDenoising(src=pic, dst=None, h=3, templateWindowSize=7, searchWindowSize=21)
    cv2.imshow("Nl-mean ",dst)

    dst  = cv2.fastNlMeansDenoisingColored(src=pic)
    cv2.imshow("Nl-meancolor", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
