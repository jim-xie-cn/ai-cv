import cv2
import numpy as np
'''
函数名：Init_Yolo，使用OpenCV初始化训练好的YOLO模型
输入参数conf，YOLO的参数配置文件
输入参数weights，训练好的YOLO权重文件
输入参数label_file：分类标签文件
返回值:YOLO模型和分类标签
'''
def Init_Yolo(conf,weights,label_file):
    with open(label_file, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    model = cv2.dnn.readNetFromDarknet(conf,weights)
    return model,labels
'''
函数名：Detect，使用YOLO模型进行目标检测
输入参数model，YOLO模型
输入参数labels，分类标签
输入参数img，图像像素数组
返回值:检测到的目标边界框和对应的分类
'''
def Detect(model,labels,img):
    #将输入的图像转换为模型的输入格式
    blobImg = cv2.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False) 
    model.setInput(blobImg)
    #得到模型的输出
    outInfo = model.getUnconnectedOutLayersNames() 
    outputs = model.forward(outInfo)
    #解析模型的输出，得到目标边界框，置信度和分类ID
    boxes,confidences,classIDs = [],[],[] #定义目标边界框，置信度和分类ID
    (H, W) = img.shape[:2]
    for out in outputs:
        for detection in out: 
            scores = detection[5:] 
            classID = np.argmax(scores) 
            confidence = scores[classID] 
            if confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H]) 
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    #使用非极大值抑制去重
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.1)
    #输出检测结果
    bboxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cate = labels[classIDs[i]] #分类标签
            bboxes.append((int(x),int(y),int(w),int(h),cate))
    return np.array(bboxes)
'''
函数名：Draw，显示目标检测的jieg
输入参数img，原始图片
输入参数bboxes，目标边界框和分类标签
返回值:带有目标检测结果的图像
'''
def Draw(img, bboxes):
    mask = img.copy()
    for box in bboxes:
        x,y,w,h,cate = int(box[0]),int(box[1]),int(box[2]),int(box[3]),box[4]
        p = (int(x)+int(w)/2,int(y)+int(h)/2)
        cv2.circle(mask,(int(p[0]),int(p[1])),7,(0,255,0),-1)
        cv2.rectangle(mask, (x, y), (x+w, y+h),(255,255,255), 2) 
        cv2.putText(mask, cate, (x-20, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    return mask

if __name__ == "__main__":
    model,labels = Init_Yolo('./models/yolov3-tiny.cfg','./models/yolov3-tiny.weights','./models/coco.names')
    img = cv2.imread('./images/ball_2.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #转换为RGB格式
    bboxes = Detect(model,labels,img)
    print(bboxes)
    mask = Draw(img,bboxes)
    cv2.imshow("目标检测结果",mask)
    cv2.waitKey(0)
