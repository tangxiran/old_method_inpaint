def feibo(number ):
    if number ==1 or number==2:return 1
    else:
        return  feibo(number-1)+feibo(number-2)

if __name__ == '__main__':
    txt  = 'AFFAfafaf  af a a a '

    x = txt.casefold()

    print(x)
    feibo18 = feibo(3)
    print(feibo18)
    import cv2

    originpic = 'origin_mask.png'

    pic = cv2.imread(originpic, 0)
    print(pic)
    pic_copy = pic.copy()