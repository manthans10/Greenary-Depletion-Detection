import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




imageCollection = ['may-2001.jpg', 'march-2002.jpg', 'jan-2003.jpg',
                   'feb-2007.jpg', 'jan-2008.jpg', 'jan-2009.jpg', 'april-2009.jpg', 'june-2010.jpg']

originImgArray = []
maskImgArray = []
resultImgArray = []  
greenAreaArray = []


# Bring some raw data.
yearData = []
areaSqkm = []

# In my original code I create a series and run on that,
# so for consistency I create a series from the list.




def add_value_labels(ax, spacing=5):
    """Add labels to the end of each bar in a bar chart.

    Arguments:
        ax (matplotlib.axes.Axes): The matplotlib object containing the axes
            of the plot to annotate.
        spacing (int): The distance between the labels and the bars.
    """

    # For each bar: Place a label
    for rect in ax.patches:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = spacing
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        label = "{:.1f}".format(y_value)

        # Create annotation
        ax.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.





class MyImage:
    def __init__(self, img_name):
        self.img = cv2.imread(img_name)
        self.__name = img_name

    def __str__(self):
        return self.__name


def stackImages(imgArray, scale,labels):
    sizeW = imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(
                    imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(
                        imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(
                imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c*eachImgWidth, eachImgHeight*d), (c*eachImgWidth+len(
                    lables[d][c])*13+27, 30+eachImgHeight*d), (255, 255, 255), cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth*c+10, eachImgHeight *
                                                d+20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def empty(a):
    pass


while True:

    for i in imageCollection:
        img = cv2.imread(i)
        scale_percent = 30  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        imageTitle = MyImage(i)
        imageNamesplited = str(imageTitle).split('.')
        imageName = imageNamesplited[0]
        img_date = imageTitle
        # print(str(x))
        # _, img = cap.read()
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#####################################

        lower = np.array([78, 56, 0])  # h_min,s_min,v_min
        upper = np.array([86, 255, 83])  # h_max,s_max,v_max
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)

        maskImgShape = img.shape
        # print(maskImgShape[0], maskImgShape[1])
        totalArea = int((maskImgShape[0] * maskImgShape[1]) * 80.4059)/1000000
        whitepart = np.sum(mask >= 100)
        whitepart = int(whitepart*80.4059)/1000000
        areaSqkm.append(whitepart)
        yearData.append(imageName)
        greenAreaArray.append(str(imageName) + ' =   ' +
                              str(whitepart) + ' ' + 'sq km')

        # print('greaan area = ' + str(whitepart), 'sq mtr')
        # print('total area=',totalArea, 'sq mtr')

        # height, width, channels = mask.shape
        # number_of_white_pix = np.sum(img == 0)
        # print(height, width, channels)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # number_of_white_pix = np.sum(img == 255)
        hStack = np.hstack([img, mask, result])
        #cv2.imshow('Original', img)
        #cv2.imshow('HSV Color Space', imgHsv)
        resized_original = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        resized_original_text = cv2.putText(img=resized_original, text=str(imageTitle), org=(
            10, 20), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=(125, 246, 55), thickness=2)
        resized_mask = cv2.resize(mask, dim, interpolation=cv2.INTER_AREA)
        resized_result = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)
        # resized_hstack = cv2.resize(hStack, dim, interpolation= cv2.INTER_LINEAR)

        originImgArray.append(resized_original_text)
        maskImgArray.append(resized_mask)
        resultImgArray.append(resized_result)

        # cv2.imshow(i+'-Mask', resized_mask)
        # cv2.imshow(i+'-Result', resized_result)
        # cv2.imshow(i+'-original', resized_original)
    print(greenAreaArray)
    greenAreaArray.clear()
    stackedImages = stackImages((originImgArray, maskImgArray, resultImgArray), 0.8)
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', stackedImages.shape[1], stackedImages.shape[0])
    cv2.imshow('Resized Window', stackedImages)
    # print(stackedImages.shape[1])
    originImgArray.clear()
    maskImgArray.clear()
    resultImgArray.clear()
    freq_series = pd.Series(areaSqkm)

    # x_labels = [108300.0, 110540.0, 112780.0, 115020.0, 117260.0, 119500.0,
    #             121740.0, 123980.0, 126220.0, 128460.0, 130700.0]

    # Plot the figure.
    plt.figure(figsize=(12, 8))
    ax = freq_series.plot(kind='bar')
    ax.set_title('Greenary Detection')
    ax.set_xlabel('Years')
    ax.set_ylabel('Area(sq km)')
    ax.set_xticklabels(yearData)
    add_value_labels(ax)
    plt.show()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()
