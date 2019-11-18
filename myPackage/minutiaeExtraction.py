import cv2
import numpy as np
from myPackage import tools as tl
from os.path import join, altsep, basename

# def process(skeleton, name, plot= False, path= None):
def process(skeleton,name,label):
    print("Minutiae extraction...")
    # img = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
    (h,w) = skeleton.shape[:2]

    temp = []
    temps = np.zeros(3)
    data = np.zeros(shape=(h,w))

    # if path is not None:
    filename = 'minutiae_'+name+"_"+str(label)+'.csv'
    full_name = altsep.join((".\Result", filename))
    file = open(full_name, 'w')
    # file.write('# (x, y) position of minutiae and type as class (0: termination, 1: bifurcation)\n')
    file.write("Posisi X" +";"+ "Posisi Y" +";"+ "Type Minutiae" +";"+ "Nama Class"+ '\n')
    for i in range(h):
        for j in range(w):
            if skeleton[i, j] == 255:
                # En caso de valer 255 se analizan sus vecinos,
                # para saber si se trata de una terminación o de una bifurcación

                window = skeleton[i - 1:i + 2, j - 1:j + 2]
                neighbours = sum(window.ravel()) // 255

                if neighbours == 2:
                    temps[0] = i
                    temps[1] = j
                    temps[2] = 0

                    temp.append(temps)
                    # if path is not None:
                    file.write(str(i) +";"+str(j)+";"+str(0)+";"+str(label)+ "\n")
                    # cv2.circle(img, (j, i), 1, (0, 255, 0), 1)

                if neighbours > 3:
                    temps[0] = i
                    temps[1] = j
                    temps[2] = 1
                    file.write(str(i) +";"+str(j)+";"+str(1) +";"+str(label)+ "\n")

                    temp.append(temps)
                    # if path is not None:
                        # file.write(str(i) + ',' + str(j) + ',1\n')
                    # cv2.circle(img, (j, i), 1, (255, 0, 0), 1)

    # print(data)
    # if plot:
    #     cv2.imshow("Minutiae '{}'".format(name), img)
    #     cv2.waitKey(2000)
    #     cv2.destroyAllWindows()
    return temp
