import sys
import os
import glob
import time
import skimage.color as sc
from data import common
import pickle
import numpy as np
import imageio
import random
import torch
import torch.utils.data as data
import cv2

# from data import srdata

############################
## The inputs are 2 images (same subject, different pose)
############################
class MultiPIE(data.Dataset):
    def __init__(self, args, name='', train=True, testPose=[60]):
        self.args = args
        self.name = name
        self.train = train
        self.idx_scale = 0
        self.allPose = { 0: '051',
                        15: '050', -15: '140',
                        30: '041', -30: '130',
                        45: '190', -45: '080',
                        60: '200', -60: '090',
                        75: '010', -75: '120',
                        90: '240', -90: '110'}
        self.n_frames_video = []
        self._set_filesystem(args.dir_data)

        
        if train:
            # (#ofAllSession) * #ofIllu * #ofPose
            splitMid = 621 * 20 * 8
            self.begin, self.end =  [1, splitMid]
            self.images_hr, self.images_lr = self._scan()
        else:
            # [15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90]
            self.testPose =sum([testPose, [-x for x in testPose]], []) 
            self.testPose = [60]
            splitMid = 621 * 20 * len(self.testPose)
            splitEnd = 921 * 20 * len(self.testPose)
            self.begin, self.end =  [splitMid + 1, splitEnd]
            self.images_hr, self.images_lr = self._scan_test()
        
        self.num_images = len(self.images_hr)
        print("Number of images to load:", self.num_images)
        if train:
            self.repeat = max(args.batch_size * args.test_every // self.num_images, 1)
            print("repeat",self.repeat)

    # Below functions as used to prepare images
    def _scan(self):
        names_hr = []
        names_lr = []
        print(self.dir_hr_pose)
        all_hr_pose_names = sorted(glob.glob(os.path.join(self.dir_hr_pose, '*.png')))
        for hr_pose_name in all_hr_pose_names:
            hr_p = hr_pose_name.split('/')[-1]
            pose_ind = hr_p.split('_')[3]
            pose = list(self.allPose.keys())[list(self.allPose.values()).index(pose_ind)]
            # python2.x: self.allPose.keys()[self.allPose.values().index(pose_ind)]
            if pose in [90, -90, 75, -75]:
               continue

            hr_f = hr_p[0:10] + "051" + hr_p[13::]
            names_hr.append(os.path.join(self.dir_hr_frontal, hr_f))
            names_lr.append(os.path.join(self.dir_hr_pose, hr_p))

        # names_hr : hr_f
        # names_lr : hr_p
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]

        return names_hr, names_lr


    def _scan_test(self):
        names_hr = []
        names_lr = []
        print(self.dir_hr_pose)
        all_hr_pose_names = sorted(glob.glob(os.path.join(self.dir_hr_pose, '*.png')))
        for hr_pose_name in all_hr_pose_names:
            hr_p = hr_pose_name.split('/')[-1]
            pose_ind = hr_p.split('_')[3]
            pose = list(self.allPose.keys())[list(self.allPose.values()).index(pose_ind)]
            # python2.x: self.allPose.keys()[self.allPose.values().index(pose_ind)]
            if pose not in self.testPose:
                continue

            hr_f = hr_p[0:10] + "051" + hr_p[13::]
            names_hr.append(os.path.join(self.dir_hr_frontal, hr_f))
            names_lr.append(os.path.join(self.dir_hr_pose, hr_p))

        # names_hr : hr_f
        # names_lr : hr_p
        names_hr = names_hr[self.begin - 1:self.end]
        names_lr = names_lr[self.begin - 1:self.end]

        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, self.name.split('_')[0])
        self.dir_hr_frontal = os.path.join(self.apath, 'cropped/HR_frontal')
        self.dir_lr_frontal = os.path.join(self.apath, 'cropped/LR_frontal')
        self.dir_hr_pose = os.path.join(self.apath, 'cropped/HR_pose')
        self.dir_lr_pose = os.path.join(self.apath, 'cropped/LR_frontal') #LR_pose
        self.dir_gallery = os.path.join(self.apath, 'cropped/gallery')
        self.dir_mask = os.path.join(self.apath, 'cropped/mask_hair_ele_face')

    def __getitem__(self, idx):
        # lrs : hr_p, lr_f_x2, lr_f_x4
        # hrs : hr_f
        lrs, hrs, mask_3parts, filenames = self._load_file(idx, self.args.patch_size)
        lrs = lrs[0] #np.array(common.set_channel(*lrs, n_channels=self.args.n_colors))
        hrs = hrs[0]
        w, h, _ = hrs.shape

        ######################## input 
        ## hr_pose, lr_pose_x2, lr_pose_x4, lr_frontal_x2, lr_frontal_x4, hr_frontal
        lr_p_x2 = cv2.resize(lrs, (w//2, h//2), interpolation = cv2.INTER_CUBIC)
        lr_p_x4 = cv2.resize(lrs, (w//4, h//4), interpolation = cv2.INTER_CUBIC)

        lr_f_x2 = cv2.resize(hrs, (w//2, h//2), interpolation = cv2.INTER_CUBIC)
        lr_f_x4 = cv2.resize(hrs, (w//4, h//4), interpolation = cv2.INTER_CUBIC)
        pair = [[lrs, lr_p_x2, lr_p_x4, lr_f_x2, lr_f_x4], hrs]

        lr_tensors, hr_tensors = common.np2Tensor(*pair,  rgb_range=self.args.rgb_range)
        mask_3parts = torch.from_numpy(mask_3parts.copy())
        
        # poseID = int(int(filenames.split('_')[3])/15)
        subID = int(filenames.split('_')[0])
        gallery_tensors = self.loadGallery(subID) #torch.from_numpy(gallery).float()

        return lr_tensors, hr_tensors, gallery_tensors, mask_3parts, subID-1, filenames

    def loadGallery(self,subID):
        gallery = cv2.imread(os.path.join(self.dir_gallery, "{:03d}_cropped.png".format(subID)), cv2.IMREAD_GRAYSCALE)
        gallery = np.reshape(gallery, (128, 128, 1))
        gallery = gallery.transpose(2,0,1)

        tensor = torch.from_numpy(gallery).float()
        tensor.mul_(self.args.rgb_range / 255)
        return tensor

    def __len__(self):
        if self.train:
            return len(self.images_hr) * self.repeat
        else:
            return 10# len(self.images_hr)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.images_hr)
        else:
            return idx

    def _load_file(self, idx, patch_size):
        '''
        Read image from given image directory
        Return: 1 * H * W * C numpy array and list of corresponding filenames
        self.images_hr : hr_f
        self.images_lr : hr_p
        '''
        idx = self._get_index(idx)
        f_hrs = self.images_hr[idx]
        f_lrs = self.images_lr[idx]

        filenames = os.path.splitext(os.path.basename(f_lrs))[0] 
        hrs = np.array([imageio.imread(hr_name) for hr_name in [f_hrs]])
        lrs = np.array([imageio.imread(lr_name) for lr_name in [f_lrs]])

        # load mask (# mask_all: 0 background, 1 hair, 2 face features, 3 skin)
        mask_3parts = np.zeros([3, patch_size, patch_size], dtype=np.float32)

        front_filenames = os.path.splitext(os.path.basename(f_hrs))[0] 
        mask_all = imageio.imread(os.path.join(self.dir_mask, front_filenames+'_masks3.png'))
        for i in range(len(mask_3parts)):
            mask_3parts[i][np.where(mask_all == i+1)] = 1
        

        # inverse image if pose < 0
        pose = filenames.split('_')[3]
        if pose in ['080', '090', '110', '120', '130', '140']:
            hrs = hrs[:, :,::-1,:]
            lrs = lrs[:, :,::-1,:]
            mask_3parts = mask_3parts[:,:,::-1]

        return lrs, hrs, mask_3parts, filenames
