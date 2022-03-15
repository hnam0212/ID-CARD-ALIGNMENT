import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import cv2
import uuid
class Detector():
    def __init__(self,repo_or_dir='ultralytics/yolov5',model='custom',path='weights/best.pt',conf=0.6,iou=0.2):
        self.repo_or_dir = repo_or_dir
        self.model = model
        self.path = path
        self.conf = conf
        self.iou = iou
        self.detect = torch.hub.load(self.repo_or_dir,self.model,self.path)
        self.detect.conf = self.conf
        self.detect.iou = self.iou
    
    def detector(self,img):
        self.res = self.detect(img)
        return self.res

    def plot_image(self):
        plt.imshow(np.squeeze(self.res.render()))
        plt.show()

    def find_center_point(self):
        global num_to_labels
        num_to_labels = {
            0.0 :"bottom_left",
            1.0 :"bottom_right",
            2.0 :"top_left",
            3.0 :"top_right",
            }
        labels , boxes = self.res.xyxy[0][:,-1].numpy() , self.res.xyxy[0][:,:-2].numpy()
        final_labels = [num_to_labels[idx] for idx in labels]
        final_points = list(map(lambda box:(int((box[0]+box[2])//2),int((box[1]+box[3])//2)),boxes))
        self.label_boxes = dict(zip(final_labels,final_points))
        
        if len(self.label_boxes)==3:
            missing_label = num_to_labels.values() - self.label_boxes.keys()
            a = missing_label.pop()
            if a == 'bottom_left':
                mid_point = np.add(self.label_boxes['top_left'],self.label_boxes['bottom_right'])/2
                x = int(2 * mid_point[0] - self.label_boxes['top_right'][0])
                y = int(2 * mid_point[1] - self.label_boxes['top_right'][1])
                self.label_boxes['bottom_left'] = (x,y)
            elif a == 'bottom_right':
                mid_point = np.add(self.label_boxes['top_right'],self.label_boxes['bottom_left'])/2
                x = int(2 * mid_point[0] - self.label_boxes['top_left'][0])
                y = int(2 * mid_point[1] - self.label_boxes['top_left'][1])
                self.label_boxes['bottom_right'] = (x,y)
            elif a == 'top_left':
                mid_point = np.add(self.label_boxes['top_right'],self.label_boxes['bottom_left'])/2
                x = int(2 * mid_point[0] - self.label_boxes['bottom_right'][0])
                y = int(2 * mid_point[1] - self.label_boxes['bottom_right'][1])
                self.label_boxes['top_left'] = (x,y)
            elif a == 'top_right':
                mid_point = np.add(self.label_boxes['top_left'],self.label_boxes['bottom_right'])/2
                x = int(2 * mid_point[0] - self.label_boxes['bottom_left'][0])
                y = int(2 * mid_point[1] - self.label_boxes['bottom_left'][1])
                self.label_boxes['top_right'] = (x,y)
        
        return self.label_boxes

    def alignment_id_card(self,img):
        img = cv2.imread(img)
        dest_points = np.float32([[0,0], [500,0], [500,300], [0,300]])
        source_points = np.float32([self.label_boxes['top_left'], self.label_boxes['top_right'], self.label_boxes['bottom_right'], self.label_boxes['bottom_left']])
        M = cv2.getPerspectiveTransform(source_points, dest_points)
        dst = cv2.warpPerspective(img, M, (500, 300))
        
        return dst



if __name__ == "__main__":
    img = os.path.join("test_4.jpg")
    det = Detector()
    res = det.detector(img)
    res.print()
    label_boxes = det.find_center_point()
    print(label_boxes)
    det.plot_image()
    dst = det.alignment_id_card(img)
    #cv2.imshow('dst',dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows() 
    file_name = str(uuid.uuid1())+".jpg"
    align_img_path=os.path.join(file_name)
    cv2.imwrite(align_img_path,dst)
