import cv2
from LoadImages import LoadImages
import os
import numpy as np

def LoadImages(data):
    '''
    加载数据集
    params:
        data:训练集数据所在的目录，要求数据尺寸大小一样
    ret:
        images:[m,height,width]  m为样本数,height为高,width为宽
        names：名字的集合
        labels：标签
    '''
    images = []
    labels = [] 
    names = []

    label = 0
    # 过滤所有的文件夹
    for subDirname in os.listdir(data): #./face
        subjectPath = os.path.join(data, subDirname) #./face/s1
        if os.path.isdir(subjectPath): #判断是否为文件
            # 每一个文件夹下存放着一个人的照片
            names.append(subDirname) #添加子文件名到预先定义的列表中
            for fileName in os.listdir(subjectPath): #遍历文件夹下的文件名
                imgPath = os.path.join(subjectPath, fileName) #拼接路径 #./face/s1/1.bmp
                img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE) #读取图像
                #print(img) #自己加的
                images.append(img) #添加读取的图像到预先定义好的图像列表中
                labels.append(label) #制作标签
            label += 1
    images = np.asarray(images)#转换为numpy矩阵
    labels = np.asarray(labels)
    return images, labels, names


# 加载训练数据
X, y, names = LoadImages('./face')

model = cv2.face.EigenFaceRecognizer_create()
model.train(X, y)

# 创建一个级联分类器 加载一个 .xml 分类器文件. 它既可以是Haar特征也可以是LBP特征的分类器.
face_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
print('face_cascade:',face_cascade)

# 打开摄像头
camera = cv2.VideoCapture(0)
cv2.namedWindow('Dynamic')

while (True):
    # 读取一帧图像
    ret, frame = camera.read()
    # cv2.imshow('face',frame) #+
    # cv2.waitKey(0)
    # 判断图片读取成功？
    if ret:
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #转换为灰度图像
        # 人脸检测

        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x, y, w, h) in faces:
            # 在原图像上绘制矩形
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray_img[y:y + h, x:x + w]

            try:
                # 宽92 高112
                roi_gray = cv2.resize(roi_gray, (92, 112), interpolation=cv2.INTER_LINEAR)
                params = model.predict(roi_gray)
                print('Label:%s,confidence:%.2f' % (params[0], params[1]))
                cv2.putText(frame, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            except: 
                continue

        cv2.imshow('Dynamic', frame)
        # 如果按下q键则退出
        if cv2.waitKey(100) & 0xff == ord('q'):
            break
camera.release()
cv2.destroyAllWindows()
