import cv2 as cv
import sys
import math
import pywt
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 

class HW5(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.image_data = None

        self.setWindowTitle('HW5')
        self.resize(1400, 900)
        self.scene = QtWidgets.QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, 1000, 900)
        self.view = QtWidgets.QGraphicsView(self)
        self.view.setGeometry(0, 0, 1000, 900)
        self.view.setScene(self.scene)
        self.view.setBackgroundBrush(QtGui.QBrush(QtCore.Qt.white))
        
        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1095, 215)
        self.btn_open_file.resize(200, 50)
        self.btn_open_file.setText('Geometric Transformation')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.Geometric_Transformation )

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1095, 315)
        self.btn_open_file.resize(200, 50)
        self.btn_open_file.setText('image fusion')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.image_fusion)

        self.btn_open_file = QtWidgets.QPushButton(self)
        self.btn_open_file.move(1095, 415)
        self.btn_open_file.resize(200, 50)
        self.btn_open_file.setText('Regional_Segmentation')
        # self.btn_open_file.setStyleSheet("background-color: green;")
        self.btn_open_file.clicked.connect(self.Superpixel_based_Regional_Segmentation)

        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('關閉')
        self.btn_close.setGeometry(1135, 550, 100, 30)
        self.btn_close.resize(110, 40)
        self.btn_close.clicked.connect(self.closeFile)

        self.output_height = 0
        self.output_width = 0
        self.filter_size = None

    def closeFile(self):
        ret = QtWidgets.QMessageBox.question(self, 'question', '確定關閉視窗？')
        if ret == QtWidgets.QMessageBox.Yes:
            app.quit()
        else:
            return

    def Geometric_Transformation(self): 
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            self.image_data = cv.imread(file_name)
            self.image_data = cv.cvtColor(self.image_data, cv.COLOR_BGR2RGB)
            fig, ax = plt.subplots(figsize=(8, 6))
            self.pic = ax.imshow(self.image_data)
            ax.set_xticks([])
            ax.set_yticks([])
            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.clear()
            self.scene.addWidget(canvas)

        img = self.image_data
        rows, cols = img.shape[:2]

        src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
        dst_points = np.float32([[0,0], [cols-1,0], [int(0.22*cols),(rows-1)*0.78], [int(0.78*cols),(rows-1)*0.78]]) 
        projective_matrix = cv.getPerspectiveTransform(src_points, dst_points)
        img_output = cv.warpPerspective(img, projective_matrix, (cols,rows))
        Trapezoidal = cv.cvtColor(img_output, cv.COLOR_BGR2GRAY)

        Wavy = np.zeros(img.shape, dtype=img.dtype) 
        
        for i in range(rows): 
            for j in range(cols): 
                offset_x = int(25.0 * math.sin(2 * 3.14 * i / 160)) 
                offset_y = int(25.0 * math.cos(2 * 3.14 * j / 160)) 

                if i+offset_y <= rows and j+offset_x <= cols: 
                    Wavy[i, j] = img[np.mod(i + offset_y, rows), np.mod(j + offset_x, cols)]
                else: 
                    Wavy[i,j] = 0 

        Wavy = cv.cvtColor(Wavy, cv.COLOR_BGR2GRAY)

        gain = 0.5

        # set background color
        bgcolor = (0,0,0)

        # get dimensions

        xcent = cols / 2
        ycent = rows / 2
        rad = min(xcent,ycent)

        # set up the x and y maps as float32
        map_x = np.zeros((rows, cols), np.float32)
        map_y = np.zeros((rows, cols), np.float32)
        mask = np.zeros((rows, cols), np.uint8)

        # create map with the spherize distortion formula --- arcsin(r)
        # xcomp = arcsin(r)*x/r; ycomp = arsin(r)*y/r
        for y in range(rows):
            Y = (y - ycent)/ycent
            for x in range(cols):
                X = (x - xcent)/xcent
                R = math.hypot(X,Y)
                if R == 0:
                    map_x[y, x] = x
                    map_y[y, x] = y
                    mask[y,x] = 255
                elif R > 1:
                    map_x[y, x] = x
                    map_y[y, x] = y
                    mask[y,x] = 0
                elif gain >= 0:
                    map_x[y, x] = xcent*X*math.pow((2/math.pi)*(math.asin(R)/R), gain) + xcent
                    map_y[y, x] = ycent*Y*math.pow((2/math.pi)*(math.asin(R)/R), gain) + ycent
                    mask[y,x] = 255
                elif gain < 0:
                    gain2 = -gain
                    map_x[y, x] = xcent*X*math.pow((math.sin(math.pi*R/2)/R), gain2) + xcent
                    map_y[y, x] = ycent*Y*math.pow((math.sin(math.pi*R/2)/R), gain2) + ycent
                    mask[y,x] = 255

        # do the remap  this is where the magic happens
        result = cv.remap(img, map_x, map_y, cv.INTER_LINEAR, borderMode = cv.BORDER_REFLECT_101, borderValue=(0,0,0))

        # process with mask
        Circular = result.copy()
        Circular[mask==0] = bgcolor

        Circular = cv.cvtColor(Circular, cv.COLOR_BGR2GRAY)

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        ax[0, 0].imshow(img)
        ax[0, 0].set_title("RGB")
        ax[0, 0].axis('off')

        ax[0, 1].imshow(cv.cvtColor(Trapezoidal, cv.COLOR_RGB2BGR))
        ax[0, 1].set_title("Trapezoidal Transformation")
        ax[0, 1].axis('off')

        ax[1, 0].imshow(cv.cvtColor(Wavy, cv.COLOR_RGB2BGR))
        ax[1, 0].set_title('Wavy Transformation')
        ax[1, 0].axis('off')

        ax[1, 1].imshow(cv.cvtColor(Circular, cv.COLOR_RGB2BGR))
        ax[1, 1].set_title('Circular Transformation')
        ax[1, 1].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

    def image_fusion(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name:
            img1 = cv.imread(file_name)
    
        file_name1, _ = QtWidgets.QFileDialog.getOpenFileName()
        if file_name1:
            img2 = cv.imread(file_name1)
            
            # 確保圖像具有相同的大小
            rows, cols, _ = img1.shape
            img2 = cv.resize(img2, (cols, rows))

            # 應用2D離散小波變換
            coeffs1_1 = pywt.dwt2(img1, 'bior1.3')
            coeffs1_2 = pywt.dwt2(img2, 'bior1.3')
            coeffs2_1 = pywt.wavedec2(img1, 'bior1.3', level=2)
            coeffs2_2 = pywt.wavedec2(img2, 'bior1.3', level=2)
            coeffs3_1 = pywt.wavedec2(img1, 'bior1.3', level=3)
            coeffs3_2 = pywt.wavedec2(img2, 'bior1.3', level=3)

            # 近似子帶的融合規則
            fused_A = (coeffs1_1[0] + coeffs1_2[0]) / 2
            fused_2 = (coeffs2_1[0] + coeffs2_2[0]) / 4
            fused_3 = (coeffs3_1[0] + coeffs3_2[0]) / 6


            # 細節子帶的融合規則
            fused_H = np.maximum(coeffs1_1[1][0], coeffs1_2[1][0])
            fused_V = np.maximum(coeffs1_1[1][1], coeffs1_2[1][1])
            fused_D = np.maximum(coeffs1_1[1][2], coeffs1_2[1][2])

            fused_2H = np.maximum(coeffs2_1[1][0], coeffs2_2[1][0])
            fused_2V = np.maximum(coeffs2_1[1][1], coeffs2_2[1][1])
            fused_2D = np.maximum(coeffs2_1[1][2], coeffs2_2[1][2])

            fused_3H = np.maximum(coeffs3_1[1][0], coeffs3_2[1][0])
            fused_3V = np.maximum(coeffs3_1[1][1], coeffs3_2[1][1])
            fused_3D = np.maximum(coeffs3_1[1][2], coeffs3_2[1][2])

            fused_coeffs = (fused_A, (fused_H, fused_V, fused_D))
            fused_coeffs2 = (fused_2, (fused_2H, fused_2V, fused_2D))
            fused_coeffs3 = (fused_3, (fused_3H, fused_3V, fused_3D))

            # 逆2D離散小波變換
            fused_image = pywt.idwt2(fused_coeffs, 'bior1.3')
            fused_image2 = pywt.waverec2(fused_coeffs2, 'bior1.3')
            fused_image3 = pywt.waverec2(fused_coeffs3, 'bior1.3')

            # 將像素值正規化到0到255的範圍
            fused_image = np.clip(fused_image, 0, 255).astype(np.uint8)
            fused_image = cv.cvtColor(fused_image, cv.COLOR_BGR2RGB)

            fused_image2 = np.clip(fused_image2, 0, 255).astype(np.uint8)
            fused_image2 = cv.cvtColor(fused_image2, cv.COLOR_BGR2RGB)
            fused_image2 = cv.resize(fused_image2, (cols, rows))

            fused_image3 = np.clip(fused_image3, 0, 255).astype(np.uint8)
            fused_image3 = cv.cvtColor(fused_image3, cv.COLOR_BGR2RGB)
            fused_image3 = cv.resize(fused_image3, (cols, rows))

            fig, ax = plt.subplots(1, 3, figsize=(10, 8))
            ax[0].imshow(fused_image, cmap ='gray')
            ax[0].set_title('1 layer wavelet transform')
            ax[0].axis('off')

            ax[1].imshow(fused_image2, cmap ='gray')
            ax[1].set_title('2 layers wavelet transform')
            ax[1].axis('off')

            ax[2].imshow(fused_image3, cmap ='gray')
            ax[2].set_title('3 layers wavelet transform')
            ax[2].axis('off')

            canvas = FigureCanvas(fig)
            self.view.setAlignment(QtCore.Qt.AlignCenter)
            self.scene.addWidget(canvas)

    def Superpixel_based_Regional_Segmentation(self):   
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        if not file_name:
            return

        img1 = cv.imread(file_name)

        segments_params = [(250, 10), (500, 10), (1000, 10)]
        superpixel_images = []
        superpixel_image_avgs = []

        for n_segments, compactness in segments_params:
            segments = slic(img1, n_segments=n_segments, compactness=compactness)
            superpixel_image = mark_boundaries(img1, segments, color=(1, 1, 1))

            num_segments = np.max(segments) + 1
            superpixel_image_avg = np.zeros_like(superpixel_image, dtype=np.float32)

            for i in range(num_segments):
                mask = segments == i
                superpixel_image_avg[mask] = np.mean(superpixel_image[mask], axis=0)

            superpixel_image_avg = (superpixel_image_avg - superpixel_image_avg.min()) / (superpixel_image_avg.max() - superpixel_image_avg.min())

            superpixel_images.append(superpixel_image)
            superpixel_image_avgs.append(superpixel_image_avg)

        fig, ax = plt.subplots(2, len(segments_params), figsize=(10, 8))

        for i, (n_segments, compactness) in enumerate(segments_params):
            ax[0, i].imshow(superpixel_images[i], cmap='gray')
            ax[0, i].set_title(f'segments{n_segments}')
            ax[0, i].axis('off')

            ax[1, i].imshow(superpixel_image_avgs[i], cmap='gray')
            ax[1, i].set_title('superpixel_image')
            ax[1, i].axis('off')

        canvas = FigureCanvas(fig)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.scene.addWidget(canvas)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    ex = HW5()
    ex.show()
    sys.exit(app.exec_())



