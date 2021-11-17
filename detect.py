import cv2
import numpy as np
from time import time

### config
hMin = 30
hMax = 30
sMin = 40
sMax = 255
vMin = 0
vMax = 255

thresh = 150
gas = 7

### templates
### timg, tsize, tcolor, tval
Timg = [
    (tlabel, timg, timg.shape[1::-1], threshold)
    for (tlabel, timg, threshold)
    in [
        ('percentage', cv2.imread('templates/percentage.png', cv2.IMREAD_GRAYSCALE), 0.8),
        ('lparenthesis', cv2.imread('templates/lparenthesis.png', cv2.IMREAD_GRAYSCALE), 0.8),
        ('dot', cv2.imread('templates/dot.png', cv2.IMREAD_GRAYSCALE), 0.8),
    ]
]

### knn
knn = cv2.ml.KNearest_create()

with np.load('knn/knn_digits.npz') as data:
    train = data['train']
    train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

def recognizeNumber(numberArr):
    sample = numberArr.reshape((1, 140)).astype(np.float32)
    ret, r, neighbours, dist = knn.findNearest(sample, k=1)

    return str(int(r[0][0]))

def recognizeImage(image):
    # demo = np.copy(image)
    # demo = cv2.cvtColor(demo, cv2.COLOR_GRAY2BGR)
    # templateMask = np.zeros_like(demo, np.uint8)

    ### template
    percentagePoints = []
    lparenthesisPoints = []
    dotPoints = []
    for (tlabel, timg, tsize, tval) in Timg:
        tresult = cv2.matchTemplate(image, timg, cv2.TM_CCOEFF_NORMED)
        loc = np.where(tresult >= tval)

        # print(f'{tlabel} {len(loc)}')
        # print((tlabel == 'percentage' or tlabel == 'lparenthesis') and len(loc) != 2)
        # print(loc)
        print('*' * 10)
        # if ((tlabel == 'percentage' or tlabel == 'lparenthesis') and len(loc) != 2):
        #     return None

        for pt in zip(*loc[::-1]):
            # cv2.rectangle(templateMask, pt, (pt[0] + tsize[0], pt[1] + tsize[1]), (0, 255, 0), cv2.FILLED)

            if (tlabel == 'percentage'):
                percentagePoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))
            elif (tlabel == 'lparenthesis'):
                lparenthesisPoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))
            elif (tlabel == 'dot'):
                dotPoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))

        # demo = cv2.addWeighted(demo, 1.0, templateMask, 5.0, 0.0)
        # cv2.imshow('demo', demo)

    ### find char
    charWidth = 10
    charHeight = 14

    paired = []
    for (ptl, pbr) in percentagePoints:
        start = None
        end = (ptl[0], ptl[1] + charHeight)
        dot = None

        for (ltl, lbr) in lparenthesisPoints:
            if (ptl[1] + 1 == ltl[1]):
                start = (lbr[0] + 1, ltl[1] - 1)
                break
        ### not found
        if (start == None):
            return None
        
        for (dtl, dbr) in dotPoints:
            if (ptl[1] + 1 == dtl[1]):
                dot = (dtl, dbr)
                break

        numberWidth = end[0] - start[0]
        print('dot')
        print(dot)
        hasDot = dot != None
        dotStripedWidth = numberWidth
        if (hasDot):
            dotStripedWidth -= 6
        numberCount = int(dotStripedWidth / charWidth)
        print('width: {} / {}, hasDot: {}, numberCount: {}'.format(numberWidth, dotStripedWidth, hasDot, numberCount))

        if (not hasDot):
            print('100%')
            paired.append('100%')
        else:
            ### detect number
            buffer = ''

            ### first
            y = start[1]
            x = start[0]

            numberArr = image[y:y + charHeight, x:x + charWidth]
            detectedNumber = recognizeNumber(numberArr)
            # print(detectedNumber)
            buffer += detectedNumber

            if (numberCount == 3):
                ### 2 / 3
                x += charWidth
                numberArr = image[y:y + charHeight, x:x + charWidth]
                detectedNumber = recognizeNumber(numberArr)
                # print(detectedNumber)
                buffer += detectedNumber
            
            ### jump dot
            x += 6
            # print('.')
            buffer += '.'

            ### last one
            x += charWidth
            numberArr = image[y:y + charHeight, x:x + charWidth]
            detectedNumber = recognizeNumber(numberArr)
            # print(detectedNumber)
            buffer += detectedNumber

            print(buffer + '%')
            paired.append(float(buffer))
    
    # detectNumber = 'HP: {0[0]}\nMP: {0[1]}'.format(paired)
    print(paired)
    print('=' * 10)
    return paired


def detectImage(image):
    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Convert to HSV format and color threshold
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    filtered = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresholded, (gas, gas), 0)

    # ### nameplate
    contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h

        if (w < 100 or h < 50):
            continue
            
        # cv2.imshow(f'{w} x {h} : {aspect}', gray[y:y+h, x:x+w])

        if (aspect < 2 or aspect > 4):
            continue

        # cv2.imshow(f'{w} x {h} : {aspect}', gray[y:y+h, x:x+w])
        detected = recognizeImage(gray[y:y+h, x:x+w])
        if (detected and len(detected) == 2):
            return detected[0]

    return None


if __name__=='__main__':
    img = cv2.imread('01.png')

    beginTime = time()
    result = detectImage(img)
    print('used time: {:.2f} ms'.format((time() - beginTime) * 1000))

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(result)
