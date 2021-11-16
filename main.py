import cv2
import numpy as np

### constant
WINDOW_TITLE = 'preview'
SHOW_CONFIG = 0
class COLOR:
    ### (b, g, r)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)

    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    RED = (0, 0, 255)
    
    MAGENTA = (255, 0, 255)
    CYAN = (255, 255, 0)
    YELLOW = (0, 255, 255)

    GOLD = (0, 215, 255)

### track variables
hMin = sMin = vMin = 0
hMax = sMax = vMax = 0
thresh = gas = 0
alpha = beta = gamma = 0

dity = 1

### track callback
def getTrackCallback(var):
    def cb(v):
        globals()[var] = v
        globals()['dity'] = 1
    
    return cb

### track options
tracks = [
    # var, cur, max
    ('hMin', 30, 179),
    ('hMax', 30, 179),

    ('sMin', 240, 255),
    ('sMax', 255, 255),
    
    ('vMin', 0, 255),
    ('vMax', 255, 255),

    ('thresh', 150, 255),

    ('gas', 7, 9),

    ('alpha', 10, 10),
    ('beta', 6, 10),
    ('gamma', 10, 10),
]

# Create a window
cv2.namedWindow(WINDOW_TITLE)

for (track_var, track_cur, track_max) in tracks:
    if (SHOW_CONFIG):
        cv2.createTrackbar(track_var, WINDOW_TITLE, 0, track_max, getTrackCallback(track_var))
        cv2.setTrackbarPos(track_var, WINDOW_TITLE, track_cur)
    globals()[track_var] = track_cur

# Load image
src = [
    cv2.imread('data/16771.png'),
    cv2.imread('data/45668.png'),
    cv2.imread('data/73757.png'),
    cv2.imread('data/118371.png'),
    cv2.imread('data/145646.png'),
    cv2.imread('data/189860.png'),
    cv2.imread('data/235711.png'),
    cv2.imread('data/267142.png'),
]

mh = mw = 0
images = [None] * len(src)
for image in src:
    h, w = image.shape[:2]
    print(f'image size: {h}x{w}')
    mh = max(mh, h)
    mw = max(mw, w)

print(f'max image size: {mh}x{mw}')

### templates
### timg, tsize, tcolor, tval
Timg = [
    (tlabel, timg, timg.shape[1::-1], color, threshold)
    for (tlabel, timg, color, threshold)
    in [
        ('percentage', cv2.imread('templates/percentage.png', cv2.IMREAD_GRAYSCALE), COLOR.YELLOW, 0.8),
        ('slash', cv2.imread('templates/slash.png', cv2.IMREAD_GRAYSCALE), COLOR.BLUE, 0.9),
        ('level', cv2.imread('templates/level.png', cv2.IMREAD_GRAYSCALE), COLOR.CYAN, 0.7),
        ('lparenthesis', cv2.imread('templates/lparenthesis.png', cv2.IMREAD_GRAYSCALE), COLOR.GREEN, 0.8),
        ('dot', cv2.imread('templates/dot.png', cv2.IMREAD_GRAYSCALE), COLOR.YELLOW, 0.999),
    ]
]

for i in range(len(src)):
    image = src[i]
    h, w = image.shape[:2]

    images[i] = cv2.copyMakeBorder(image, 0, mh - h, 0, mw - w, cv2.BORDER_REPLICATE)

### knn
knn = cv2.ml.KNearest_create()

with np.load('knn/knn_digits.npz') as data:
    print( data.files )
    train = data['train']
    train_labels = data['train_labels']
    knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

def recognizeNumber(numberArr):
    sample = numberArr.reshape((1, 140)).astype(np.float32)
    ret, r, neighbours, dist = knn.findNearest(sample, k=1)

    # cv2.imshow(str(int(r[0][0])), cv2.resize(numberArr, None, fx=10, fy=10))

    return str(int(r[0][0]))

while(1):
    if (not dity):
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        continue

    dity =  0
    # Set minimum and maximum HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    results = []
    for image in images:
        # Convert to HSV format and color threshold
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)
        filtered = cv2.bitwise_and(image, image, mask=mask)

        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        if (gas % 2 == 0):
            gas += 1

        blur = cv2.GaussianBlur(thresholded, (gas, gas), 0)

        display_type = 'image'
        # display_type = 'gray'
        # display_type = 'blur'

        if (display_type == 'image'):
            result = image.copy()
        elif (display_type == 'gray'):
            result = gray.copy()
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif (display_type == 'blur'):
            result = blur.copy()
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        ### nameplate
        contours, hierarchy = cv2.findContours(blur, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)

            if (w < 100 or h < 50):
                continue

            cv2.rectangle(result, (x, y), (x + w, y + h), COLOR.GREEN, 2)
            label = f'{w}x{h}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            size, _ = cv2.getTextSize(label, font, 0.5, 1)
            pos = (x + w - size[0], y + h + size[1])
            cv2.rectangle(result, pos, (pos[0] + size[0], pos[1] - size[1]), COLOR.GREEN, cv2.FILLED)
            cv2.putText(result, label, pos, font, 0.5, COLOR.RED)
        
        ### template
        percentagePoints = []
        lparenthesisPoints = []
        dotPoints = []
        templateMask = np.zeros_like(result, np.uint8)
        for (tlabel, timg, tsize, tcolor, tval) in Timg:
            tresult = cv2.matchTemplate(gray, timg, cv2.TM_CCOEFF_NORMED)
            loc = np.where(tresult >= tval)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(templateMask, pt, (pt[0] + tsize[0], pt[1] + tsize[1]), tcolor, cv2.FILLED)
                
                if (tlabel == 'percentage'):
                    percentagePoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))
                elif (tlabel == 'lparenthesis'):
                    lparenthesisPoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))
                elif (tlabel == 'dot'):
                    dotPoints.append((pt, (pt[0] + tsize[0], pt[1] + tsize[1])))

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
                break
            
            for (dtl, dbr) in dotPoints:
                if (ptl[1] + 1 == dtl[1]):
                    dot = (dtl, dbr)
                    break

            cv2.rectangle(templateMask, start, end, COLOR.MAGENTA, cv2.FILLED)
            numberWidth = end[0] - start[0]
            hasDot = dot != None
            dotStripedWidth = numberWidth
            if (hasDot):
                dotStripedWidth -= 6
            numberCount = int(dotStripedWidth / charWidth)
            # print('width: {} / {}, hasDot: {}, numberCount: {}'.format(numberWidth, dotStripedWidth, hasDot, numberCount))

            if (not hasDot):
                # print('100%')
                paired.append('100%')
            else:
                ### detect number
                buffer = ''

                ### first
                y = start[1]
                x = start[0]

                numberArr = gray[y:y + charHeight, x:x + charWidth]
                detectedNumber = recognizeNumber(numberArr)
                # print(detectedNumber)
                buffer += detectedNumber

                if (numberCount == 3):
                    ### 2 / 3
                    x += charWidth
                    numberArr = gray[y:y + charHeight, x:x + charWidth]
                    detectedNumber = recognizeNumber(numberArr)
                    # print(detectedNumber)
                    buffer += detectedNumber
                
                ### jump dot
                x += 6
                # print('.')
                buffer += '.'

                ### last one
                x += charWidth
                numberArr = gray[y:y + charHeight, x:x + charWidth]
                detectedNumber = recognizeNumber(numberArr)
                # print(detectedNumber)
                buffer += detectedNumber

                # print(buffer + '%')
                paired.append(buffer + '%')
        
        detectNumber = 'HP: {0[0]}\nMP: {0[1]}'.format(paired)
        print(detectNumber)
        print('=' * 10)
        result = cv2.addWeighted(result, alpha / 10.0, templateMask, beta / 10.0, gamma / 10.0)

        ### final result
        result = cv2.rectangle(result, (95, 115), (250, 155), COLOR.WHITE, cv2.FILLED)

        result = cv2.putText(result, detectNumber.splitlines()[0], (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR.RED)
        result = cv2.putText(result, detectNumber.splitlines()[1], (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR.RED)

        hp = float(paired[0][:-1])
        print(hp)
        ledMainColor = COLOR.GREEN
        if (hp < 30):
            ledMainColor = COLOR.RED
        elif (hp == 100):
            ledMainColor = COLOR.GOLD
        result = cv2.rectangle(result, (215, 120), (245, 150), ledMainColor, cv2.FILLED)

        results.append(result)

    if (len(results) == 8):
        cv2.imshow(WINDOW_TITLE, cv2.vconcat([
            # cv2.hconcat(results[:4]),
            # cv2.hconcat(results[4:]),
            cv2.hconcat(results[:2]),
            cv2.hconcat(results[2:4]),
            cv2.hconcat(results[4:6]),
            cv2.hconcat(results[6:]),
        ]))
    else:
        cv2.imshow(WINDOW_TITLE, cv2.resize(results[0], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
