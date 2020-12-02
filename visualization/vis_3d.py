import sys, os
import numpy as np 
import pdb
import mayavi.mlab as mlab
import mayavi.mlab as mlab
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, draw_lidar_raw
import json


def load_3d(file):
    f = np.loadtxt(file, delimiter=',')
    f = f.reshape(-1, 5)
    return f

def load_3d2(file):
    f = np.loadtxt(file, delimiter=',')
    return f


root_dir = 'dataset/raw_data/lidar'
label_dir = 'dataset/txt_files'


def property(velo, p):
    for i in range(4):
        p[0, 2*i] = velo[:, i].min()
        p[0, 2*i+1] = velo[:, i].max()
    return p


def filter_point(velo):
    p = []
    for v in velo:
        if v[0]>=0 and v[1]<=70 and v[1]<=40 and v[1]>= -40 and v[2]>=-2.5 and v[2]<=1:
            p.append(v)
    p = np.array(p).reshape(-1, 4)
    return p

# fig = mlab.figure(figure=None, bgcolor=(1,1,1),
#         fgcolor=None, engine=None, size=(1000, 500))




def get_3dboxes(objs):
    out = []
    for ob in cords:
        if ob[-1] == -10:
            continue
        x = -ob[10]
        # x = int(ob[10]*10)+400
        # y = 799-(int(ob[12]*10))
        y = ob[12]
        z = -ob[11]

        l = ob[9]
        w = ob[8]
        h = ob[7]
        # h = ob[7]
        # print(h)

        x1 = [l/2, -w/2]
        x2 = [l/2, w/2]
        x3 = [-l/2, w/2]
        x4 = [-l/2, -w/2]

        cen = [x, y]

        r = [[np.cos(ob[-1]), -np.sin(ob[-1])],
            [np.sin(ob[-1]), np.cos(ob[-1])]]

        xx1 = np.dot(r, x1) + cen
        xx2 = np.dot(r, x2) + cen
        xx3 = np.dot(r, x3) + cen
        xx4 = np.dot(r, x4) + cen

        xx = np.stack([xx1,xx2,xx3,xx4])
        xx = np.concatenate([xx, np.zeros((4,1))], 1)
        xx = np.concatenate([xx, xx], 0)
        xx[:, 2] += (z)
        xx[4:, 2] += h
        # xx[:, 2] += (z+h/2.0)
        # xx[4:, 2] -= h
        out.append(xx)
        # pdb.set_trace()
    return np.array(out).reshape(-1, 8, 3)[:,:,[1,0,2]]



# list_final=[]

ob_list = os.listdir(root_dir)
ob_list.sort()
name_list = [a.split('.')[0] for a in ob_list]
# for i in range(len(name_list)):
    # i = 0

# cp = (-187.94435869,   -7.52781885,  109.47565124)
# fc = (125.00749755,   7.81650162,   8.65250039)
cp = [-91.43186992, -47.6636809 ,  33.49564775]
fc = [125.00749755,   7.81650162,   8.65250039]

class constant_camera_view(object):
    def __init__(self):
        pass

    def __enter__(self):
        self.orig_no_render = mlab.gcf().scene.disable_render
        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = True
        cc = mlab.gcf().scene.camera
        self.orig_pos = cc.position
        self.orig_fp = cc.focal_point
        self.orig_view_angle = cc.view_angle
        self.orig_view_up = cc.view_up
        self.orig_clipping_range = cc.clipping_range

    def __exit__(self, t, val, trace):
        cc = mlab.gcf().scene.camera
        cc.position = self.orig_pos
        cc.focal_point = self.orig_fp
        cc.view_angle =  self.orig_view_angle 
        cc.view_up = self.orig_view_up
        cc.clipping_range = self.orig_clipping_range

        if not self.orig_no_render:
            mlab.gcf().scene.disable_render = False
        if t != None:
            print(t, val, trace)
            ipdb.post_mortem(trace)

bond = [
[0, -50, 2, 0],
[200, -50, 2, 0],
[0, 50, 2, 0],
[200, 50, 2, 0],
[0, -50, -3, 0],
[200, -50, -3, 0],
[0, 50, -3, 0],
[200, 50, -3, 0],
]
bond = np.array(bond)


def draw_gt_boxes3d2(gt_boxes3d, fig, line_width=1, draw_text=False, text_scale=(1,1,1), color_list=None):

    num = len(gt_boxes3d)
    for n in range(num):
        b = gt_boxes3d[n]
        if b[0,0] >= 70:
            color = (1,0,0)
        else:
            color = (0,1,0)
        if color_list is not None:
            color = color_list[n] 
        if draw_text: mlab.text3d(b[4,0], b[4,1], b[4,2], '%d'%n, scale=text_scale, color=color, figure=fig)
        for k in range(0,4):
            #http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i,j=k,(k+1)%4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k+4,(k+1)%4 + 4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)

            i,j=k,k+4
            mlab.plot3d([b[i,0], b[j,0]], [b[i,1], b[j,1]], [b[i,2], b[j,2]], color=color, tube_radius=None, line_width=line_width, figure=fig)
    #mlab.show(1)
    #mlab.view(azimuth=180, elevation=70, focalpoint=[ 12.0909996 , -1.04700089, -2.03249991], distance=62.0, figure=fig)
    return fig


with constant_camera_view():
    for i in range(20000000):
        fig = mlab.figure(figure=None, bgcolor=(0,0,0),
            fgcolor=None, engine=None, size=(10000, 5000))
        velo_name = '%s.xyz' %(name_list[i])

        velo = load_3d(os.path.join(root_dir, velo_name))[:, :4]

        objs = np.loadtxt(os.path.join(label_dir, '%s.txt' %(name_list[i])), str).reshape(-1, 15)

        cords = np.array(objs[:, 1:], np.float32)

        boxes = get_3dboxes(objs)

        bound_x = np.logical_and(
            velo[:, 2] >= -3, velo[:, 2] < 2)
        bound_y = np.logical_and(
            velo[:, 1] >= -50, velo[:, 1] < 50)
        bound_z = np.logical_and(
            velo[:, 0] >= 0, velo[:, 0] < 200)

        bound_box = np.logical_and(np.logical_and(bound_x, bound_y), bound_z)
        velo = velo[bound_box]
        velo = np.concatenate([velo, bond], 0)
        # draw_lidar(velo, fig=fig, pts_scale=10,pts_color=(1,0,0))
        draw_lidar_raw(velo, fig=fig, pts_scale=1)
        # draw_gt_boxes3d(boxes, color=(1,0,0), fig=fig)
        draw_gt_boxes3d2(boxes, fig=fig)

        # mlab.savefig('figs0/%d.png' %(i))
        mlab.show()
        mlab.close()


