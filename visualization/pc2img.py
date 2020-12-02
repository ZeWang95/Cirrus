import numpy as np
import os
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import csv
import cv2
import pdb
from box import *
from PIL import Image

'''
plotpointsonImage
Projecting point clouds on 2D RGB images. Color of points indicates distance.

plotboxesImage
Projecting 3D boxes onto RGB images.

'''

class Projection:

    def __init__(self, index=0, path=None):

        data_list = os.listdir(os.path.join(path, 'camera'))
        data_list.sort()
        self.index = index
        self.directory = ""

        self.imagefilename = os.path.join(path, 'camera', data_list[index])
        self.lidarfilename = os.path.join(path, 'lidar', data_list[index].replace('jpg', 'xyz'))
        self.annofilename = os.path.join('/data/dataset/volvo/data_all/dataset/txt_files', data_list[index].replace('jpg', 'txt'))



        self.calibration = np.array( [[ 9.99901502e-01, 4.82990604e-03, 1.31779743e-02, 7.60000000e-02],
                                      [-9.99901835e-04, 9.61051089e-01, -2.16368965e-01, -9.04600000e-01],
                                      [-1.39995427e-02, 0.76328566e-01, 9.60961256e-01, 1.12000000e+00], 
                                      [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


        self.projection_matrix = np.array([[1.17305810e+03+250, 0.00000000e+00, 9.70085603e+02+40, 0.00000000e+00],
                                           [0.00000000e+00, 1.17395992e+03+130, 6.02131848e+02-200, 0.00000000e+00],
                                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])


        
    def getlidarfrontpoints(self):
          self.lidarpath = os.path.join(self.directory + self.lidarfilename)
          print(self.lidarpath)
          #self.pointcloud = np.loadtxt(self.lidarpath, delimiter=" ")[:,:3]
          self.pointcloud = np.loadtxt(self.lidarpath, delimiter=',')[:,:3]
          #idx = [2, 0, 1]
          idx=[1,2,0]
          self.pointcloud = self.pointcloud[:,idx]
          self.pointcloud[:,0] *= -1
          self.pointcloud[:,1] *= -1


          self._d = np.sqrt(self.pointcloud[:, 0]**2 + self.pointcloud[:, 1]**2 + self.pointcloud[:, 2]**2)
          self.color = 120 - np.array(self._d.clip(1, 250) /(250/120), np.uint8)

    def get3Dboxes(self):
          self.annopath = self.annofilename
          print(self.annopath)

          objs = np.loadtxt(self.annopath, str).reshape(-1, 15)
          self.obj_cls = objs[:, 0]
          objs =np.array(objs[:, -7:], np.float32)
          boxes3d = []
          for obj in objs:
            box3d = get_3d_box(obj[:3][::-1], obj[-1], obj[3:6]) 
            boxes3d.append(box3d)
          
          self.boxes3d = np.array(boxes3d).reshape(-1, 8, 3)


    def applycalibration(self):

        self.datapoints = self.pointcloud.transpose()
        self.rotatmatrix = self.calibration[:3, :3]
        self.translation_matrix = self.calibration[:3, 3:4]

        self.lidar_to_cam = np.matmul(self.rotatmatrix, self.datapoints)

        self.lidar_to_cam[0] = self.lidar_to_cam[0] + self.translation_matrix[0]
        self.lidar_to_cam[1] = self.lidar_to_cam[1] + self.translation_matrix[1]
        self.lidar_to_cam[2] = self.lidar_to_cam[2] + self.translation_matrix[2]
        new_row = np.ones(self.lidar_to_cam.shape[1])
        self.pc_3D = np.vstack([self.lidar_to_cam, new_row])

        self.intrinsic = np.array([[1428.274393, 0.000000, 943.338144],
					[0.000000, 1426.040768, 72.297686],
					[0.000000, 0.000000, 1.000000]])


        #self.points_2D = np.matmul(self.intrinsic,self.datapoints)
        self.points_2D = np.matmul(self.projection_matrix,self.pc_3D)
        self.points_2D[0] = np.divide(self.points_2D[0],self.points_2D[2])
        self.points_2D[1] = np.divide(self.points_2D[1], self.points_2D[2])

    def applycalibrationbox(self):
        self.n_boxes = self.boxes3d.shape[0]
        self.boxes3d = self.boxes3d.reshape(-1, 3)

        self.datapoints = self.boxes3d.transpose()



        self.rotatmatrix = self.calibration[:3, :3]
        self.translation_matrix = self.calibration[:3, 3:4]


        self.lidar_to_cam = np.matmul(self.rotatmatrix, self.datapoints)

        self.lidar_to_cam[0] = self.lidar_to_cam[0] + self.translation_matrix[0]
        self.lidar_to_cam[1] = self.lidar_to_cam[1] + self.translation_matrix[1]
        self.lidar_to_cam[2] = self.lidar_to_cam[2] + self.translation_matrix[2]
        new_row = np.ones(self.lidar_to_cam.shape[1])
        self.pc_3D = np.vstack([self.lidar_to_cam, new_row])

        self.intrinsic = np.array([[1428.274393, 0.000000, 943.338144],
          [0.000000, 1426.040768, 72.297686],
          [0.000000, 0.000000, 1.000000]])


        #self.points_2D = np.matmul(self.intrinsic,self.datapoints)
        self.points_2D = np.matmul(self.projection_matrix,self.pc_3D)
        self.points_2D[0] = np.divide(self.points_2D[0],self.points_2D[2])
        self.points_2D[1] = np.divide(self.points_2D[1], self.points_2D[2])


    def plotpointsonImage(self):
        self.imagepath = os.path.join(self.directory + self.imagefilename)
        # img = cv2.imread(self.imagepath)
        img = Image.open(self.imagepath).convert('RGBA')

        im = np.zeros((img.size[1], img.size[0], 3), np.uint8)
        im = print_projection_plt(self.points_2D, self.color, im)[:,:,::-1]
        # alpha = np.zeros((img.size[1], img.size[0], 1), np.uint8)#*255
        alpha = im.sum(-1)
        im = Image.fromarray(im).convert('RGBA')

        alpha = (alpha >= 0.5)*1

        alpha = np.array(alpha, np.uint8) * 200 
        alpha = 255 - alpha
        alpha = Image.fromarray(alpha).convert('L')
        image = Image.composite(img, im, alpha)

        # image.save('points2/%06d.png' %(self.index))
        image.show()
        print('done')

    def plotboxesImage(self):
        self.imagepath = os.path.join(self.directory , self.imagefilename)
        img = cv2.imread(self.imagepath)
        # pdb.set_trace()
        self.points_2D = self.points_2D.transpose().reshape(-1, 8, 3)
        for i in range(self.points_2D.shape[0]):
          if self.obj_cls[i] in ['pedestrian', 'wheeled_pedestrian']:
            color = (0, 0,255)
          elif self.obj_cls[i] in ['bicycle']:
            color = (255, 191, 0)
          else:
            color = (0, 255, 0)
          draw_projected_box3d(img, self.points_2D[i], color=color, thickness=4)
        # img = print_projection_plt(self.points_2D, self.color, img)
        cv2.imwrite('boxes/b%06d.jpg' %(self.index), img)
        image = Image.fromarray(img[:,:,::-1])
        image.show()

        print('done')


    def plotpointsonImageBlank(self):
        self.imagepath = os.path.join(self.directory + self.imagefilename)


        imgo = Image.open(self.imagepath).convert('RGBA')

        im = np.zeros((imgo.size[1], imgo.size[0], 3), np.uint8)
        im = print_projection_plt(self.points_2D, self.color, im)[:,:,::-1]


        img = Image.fromarray(np.zeros((imgo.size[1], imgo.size[0], 3), np.uint8))

        im = np.zeros((img.size[1], img.size[0], 3), np.uint8)
        im = print_projection_plt(self.points_2D, self.color, im)[:,:,::-1]
        # alpha = np.zeros((img.size[1], img.size[0], 1), np.uint8)#*255
        alpha = im.sum(-1)
        im = Image.fromarray(im).convert('RGBA')

        
        alpha = (alpha >= 0.5)*1
        # pdb.set_trace()
        alpha = np.array(alpha, np.uint8) * 200 
        alpha = 255 - alpha
        alpha = Image.fromarray(alpha).convert('L')
        image = Image.composite(img, im, alpha)
        # image.save('hh/%06d.png' %(self.index))
        out = np.concatenate([np.array(imgo)[:,:imgo.size[0]//2,:], np.array(image)[:,imgo.size[0]//2:,:]], 1)
        out = Image.fromarray(out)
        
        # out.save('hh/%06d.png' %(self.index))
        out.show()

        print('done')


def draw_projected_box3d(image, qs, color=(0,255,0), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32)
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
       i,j=k,(k+1)%4
       # use LINE_AA for opencv3
       #cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
    return image

def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # pdb.set_trace()
    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]),np.int32(points[1][i])),2, (int(color[i]),255,255),-1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

if __name__ == '__main__':
    path = '/data/dataset/volvo/data_all/dataset/raw_data'
    # start = time.time()
    # for i in range(6000):
    #   proj = Projection(i)
    #   proj.getlidarfrontpoints()
    #   proj.applycalibration()
    #   proj.plotpointsonImage()

    for i in range(0, 6000):
      proj = Projection(i, path=path)
      proj.get3Dboxes()
      proj.applycalibrationbox()
      proj.plotboxesImage()
