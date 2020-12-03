
from matplotlib import pyplot as plt
from tqdm import *
import cv2

from mxnet.gluon import nn
from mxnet import nd, gluon
from mxnet.gluon import data
import cv2
import numpy as np
import os
import random
import datautils
import json
import scipy.io as scio
import time
import math
import mxnet as mx
import argparse, time, logging, random, math
from mxnet import autograd as ag
from mxnet_CharDet import CharDet
from post_process import normalizeMeanVariance,cvt2HeatmapImg,getDetBoxes_core


class CharDet_Data(data.Dataset):
    """
    1.将带有字符级标注的数据准备成三元组（原图，字符热力图，字符连接区域热力图），热力图的形式可自定义调试，满足中间置信度高，四周低就行了。
    判断热力图制作是否合理的粗略标准是：经过特定的后处理能还原回文本框。
    2.目前开源的字符级标准数据有2个：ICDAR的ReCTs 和人工合成数据SynthText，该类即针对这两个数据集准备三元组，且SynthText太多，训练时不必都用。
    3.验证流程时，可以只用ReCTs，SynthText太大了。
    """
    
    def __init__(self, Synth_gt, norm=True, have_Synth_img=False, ReCTS_img_root= '/data3/fansen/CVProject/text-detection/DataSets/ReCTs/img', random_rote_rate=None,
                 image_size=(3, 640, 640), down_rate=2, transform=None,img_num=20000):
        
        self.have_Synth_img=have_Synth_img
        if self.have_Synth_img:
            Synth_img_root = "/data3/fansen/CVProject/text-detection/DataSets/SynthText80W/SynthText"
            self.Synth_img_root = Synth_img_root
            start_index = random.randint(1,200000)
            self.gt = {}
            self.gt["txt"] = Synth_gt["txt"][0][:-1][start_index:start_index+img_num]
            self.gt["imnames"] = Synth_gt["imnames"][0][start_index:start_index+img_num]
            self.gt["charBB"] = Synth_gt["charBB"][0][start_index:start_index+img_num]
            self.gt["wordBB"] = Synth_gt["wordBB"][0][start_index:start_index+img_num]
            
            

        self.image_list = [os.path.join(ReCTS_img_root, i) for i in os.listdir(ReCTS_img_root)]
        self.label_list = [(i.replace('jpg', 'json')).replace('img', 'gt') for i in self.image_list]

        self.norm = norm
        self.image_size = image_size
        self.down_rate = down_rate
        self.transform = transform
        self.random_rote_rate = random_rote_rate

        ky = cv2.getGaussianKernel(40, int(40 * 0.33))
        kx = cv2.getGaussianKernel(40, int(40 * 0.33))
        MAP = np.multiply(ky, np.transpose(kx))
        norMap = np.multiply(MAP, 1 / np.max(MAP))
        box = np.float32([[0, 0], [39, 0], [39, 39], [0, 39]])
        self.heatmap_loc = box
        self.heatmap = norMap

    def __len__(self):
        if self.have_Synth_img:
            return len(self.label_list) + self.gt["txt"].shape[0]
        else:
            return len(self.label_list)

    def normalizeMeanVariance(self, in_img, mean=(0.406, 0.456, 0.485), variance=(0.225, 0.224, 0.229)):
        # RGB order mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)
        # BGR order mean=(0.406, 0.456, 0.485), variance=(0.225, 0.224, 0.229)
        img = in_img.copy().astype(np.float32)
  

        img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
        img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)

        return img

    def find_min_rectangle(self, points):

        x1, y1, x2, y2, x3, y3, x4, y4 = points
        lt_x = min(x1, x2, x3, x4)
        lt_y = min(y1, y2, y3, y4)
        rd_x = max(x1, x2, x3, x4)
        rd_y = max(y1, y2, y3, y4)

        return np.float32([[lt_x, lt_y], [rd_x, lt_y], [rd_x, rd_y], [lt_x, rd_y]]), int(rd_x - lt_x), int(rd_y - lt_y)

    def syn_resize(self, image, char_label, word_laebl):
        h, w, c = image.shape
        img = np.zeros(self.image_size)
        rate = self.image_size[2] / self.image_size[1]
        rate_pic = w / h

        if rate_pic > rate:
            resize_h = int(self.image_size[2] / rate_pic)
            image = cv2.resize(image, (self.image_size[2], resize_h))
            image = image.transpose((2, 0, 1))
            img[:, :resize_h, :] = image
            char_label = char_label * (resize_h / h)
            word_laebl = word_laebl * (resize_h / h)
        else:
            resize_w = int(rate_pic * self.image_size[1])
            image = cv2.resize(image, (resize_w, self.image_size[1]))
            image = image.transpose((2, 0, 1))
            img[:, :, :resize_w] = np.array(image)
            char_label = char_label * (resize_w / w)
            word_laebl = word_laebl * (resize_w / w)

        return img, char_label, word_laebl

    def rects_resize(self, image, lines_loc, line_boxes):
        h, w, c = image.shape
        img = np.zeros(self.image_size)
        rate = self.image_size[2] / self.image_size[1]
        rate_pic = w / h

        if rate_pic > rate:
            resize_h = int(self.image_size[2] / rate_pic)
            image = cv2.resize(image, (self.image_size[2], resize_h))
            image = image.transpose((2, 0, 1))
            img[:, :resize_h, :] = image

            lines_loc = lines_loc * (resize_h / h)  # / self.down_rate
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_h / h)  # / self.down_rate

        else:

            resize_w = int(rate_pic * self.image_size[1])
            image = cv2.resize(image, (resize_w, self.image_size[1]))
            image = image.transpose((2, 0, 1))
            img[:, :, :resize_w] = image
            lines_loc = lines_loc * (resize_w / w)  # / self.down_rate
            for i in range(len(line_boxes)):
                for j in range(len(line_boxes[i])):
                    for k in range(len(line_boxes[i][j])):
                        line_boxes[i][j][k] = line_boxes[i][j][k] * (resize_w / w)  # / self.down_rate

        return img, lines_loc, line_boxes

    def get_SynthText_item(self, idx):
        img_name = self.gt["imnames"][idx][0]
        image = cv2.imread(os.path.join(self.Synth_img_root, img_name))

        if self.norm:
            image = self.normalizeMeanVariance(image)

        char_label = self.gt["charBB"][idx].transpose(2, 1, 0)

        if len(self.gt["wordBB"][idx].shape) == 3:
            word_laebl = self.gt["wordBB"][idx].transpose(2, 1, 0)
        else:
            word_laebl = self.gt["wordBB"][idx].transpose(1, 0)[np.newaxis, :]
        txt_label = self.gt["txt"][idx]

        debug_size = image.shape
        img, char_label, word_laebl = self.syn_resize(image, char_label, word_laebl)

        if self.random_rote_rate:
            angel = random.randint(0 - self.random_rote_rate, self.random_rote_rate)
            img, M = datautils.rotate(angel, img)

        char_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))
        aff_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))

        line_boxes = []
        char_index = 0
        word_index = 0
        for txt in txt_label:
            for strings in txt.split("\n"):
                for string in strings.split(" "):
                    if string == "":
                        continue
                    char_boxes = []
                    for char in string:
                        x0, y0 = char_label[char_index][0]
                        x1, y1 = char_label[char_index][1]
                        x2, y2 = char_label[char_index][2]
                        x3, y3 = char_label[char_index][3]

                        #                         if self.random_rote_rate:
                        #                             x0, y0 = datautils.rotate_point(M, x0, y0)
                        #                             x1, y1 = datautils.rotate_point(M, x1, y1)
                        #                             x2, y2 = datautils.rotate_point(M, x2, y2)
                        #                             x3, y3 = datautils.rotate_point(M, x3, y3)

                        x0, y0, x1, y1, x2, y2, x3, y3 = int(round(x0)), int(round(y0)), int(round(x1)), int(
                            round(y1)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3))
                        char_boxes.append([x0, y0, x1, y1, x2, y2, x3, y3])
                        min_rect, min_rect_w, min_rect_h = self.find_min_rectangle([x0, y0, x1, y1, x2, y2, x3, y3])

                        pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                        top_l_x, top_l_y = min_rect[0]
                        pts = pts - [top_l_x, top_l_y]
                        M = cv2.getPerspectiveTransform(self.heatmap_loc, pts)
                        res = cv2.warpPerspective(self.heatmap, M, (min_rect_w, min_rect_h))

                        min_x = min(x0, x1, x2, x3)
                        min_y = min(y0, y1, y2, y3)

                        if min_x >= self.image_size[2]:
                            # nima gt框标注存有些整个在图像只有
                            continue
                        gh, gw = res.shape
                        for th in range(gh):
                            for tw in range(gw):
                                try:
                                    char_gt[min_y + th, min_x + tw] = max(char_gt[min_y + th, min_x + tw], res[th, tw])
                                except:
                                    ## 本来标注就有问题，gt框存在越界情况，直接舍弃越界部分
                                    pass

                        char_index += 1
                    word_index += 1
                    line_boxes.append(char_boxes)
        affine_boxes = []

        for char_boxes in line_boxes:
            affine_boxes.extend(datautils.create_affine_boxes(char_boxes))
            for points in affine_boxes:
                x0, y0, x1, y1, x2, y2, x3, y3 = points[0], points[1], points[2], points[3], points[4], points[5], \
                                                 points[6], points[7]
                #                 box, deta_x, deta_y = self.find_min_rectangle(points)
                min_rect, min_rect_w, min_rect_h = self.find_min_rectangle(points)

                pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                top_l_x, top_l_y = min_rect[0]
                pts = pts - [top_l_x, top_l_y]
                M = cv2.getPerspectiveTransform(self.heatmap_loc, pts)
                res = cv2.warpPerspective(self.heatmap, M, (min_rect_w, min_rect_h))

                min_x = min(x0, x1, x2, x3)
                min_y = min(y0, y1, y2, y3)

                if min_x >= self.image_size[2]:
                    #                     print(char_boxes)
                    continue
                gh, gw = res.shape
                for th in range(gh):
                    for tw in range(gw):
                        try:
                            aff_gt[min_y + th, min_x + tw] = max(aff_gt[min_y + th, min_x + tw], res[th, tw])
                        ## 按理来说根据link框的定义，不会存在越界的情况，但是TM的GT有的框整个都在图片外，导致link框也会越界
                        except:
                            pass

        char_gt = cv2.resize(char_gt,
                             (int(self.image_size[2] / self.down_rate), int(self.image_size[1] / self.down_rate)))
        aff_gt = cv2.resize(aff_gt,
                            (int(self.image_size[2] / self.down_rate), int(self.image_size[1] / self.down_rate)))

        img = nd.array(img)
        char_gt = nd.array(char_gt)
        aff_gt = nd.array(aff_gt)
        sample = (img, char_gt, aff_gt)
        return sample

    def get_ReCTS_item(self, idx):
        image_path = self.image_list[idx]
        label_path = self.label_list[idx]
        image = cv2.imread(image_path)

        if self.norm:
            image = self.normalizeMeanVariance(image)

        with open(label_path, "r") as f:
            label_json = json.load(f)

        lines_loc = []
        chars_loc = []
        lines_boxes = []
        char_num = 0
        total_char_num = len(label_json["chars"])
        char_index = list(range(total_char_num))
        for line in label_json["lines"]:
            line_label = line["transcription"]
            #             if len(line_label) == 1:
            #                 lines_boxes.append([line["points"]])
            if line_label == "###":
                #                 neg_lines.append(line["points"])
                #                 print(line["points"])
                contours = np.array(line["points"])
                contours = contours.astype('int32')
                contours = contours.reshape((4, 2))
                cv2.drawContours(image, [contours], -1, (0, 0, 0), -1)
                continue
            if line_label != "###":
                lines_loc.append(line["points"])
            line_boxes = []

            for c in line_label:
                for char_num in char_index:
                    c_label = label_json["chars"][char_num]["transcription"]
                    if c_label == c:
                        if label_json["chars"][char_num]["points"] != [-1, -1, -1, -1, -1, -1, -1, -1]:
                            line_boxes.append(label_json["chars"][char_num]["points"])

                        char_index.remove(char_num)
                        break

            lines_boxes.append(line_boxes)

        lines_loc = np.array(lines_loc)
        img, lines_loc, lines_boxes = self.rects_resize(image, lines_loc, lines_boxes)

        if self.random_rote_rate:
            angel = random.randint(0 - self.random_rote_rate, self.random_rote_rate)
            img, M = datautils.rotate(angel, img)

        char_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))
        aff_gt = np.zeros((int(self.image_size[1]), int(self.image_size[2])))

        for line_box in lines_boxes:
            for box in line_box:
                x0, y0, x1, y1, x2, y2, x3, y3 = box
                if self.random_rote_rate:
                    x0, y0 = datautils.rotate_point(M, x0, y0)
                    x1, y1 = datautils.rotate_point(M, x1, y1)
                    x2, y2 = datautils.rotate_point(M, x2, y2)
                    x3, y3 = datautils.rotate_point(M, x3, y3)
                x0, y0, x1, y1, x2, y2, x3, y3 = int(round(x0)), int(round(y0)), int(round(x1)), int(
                    round(y1)), int(round(x2)), int(round(y2)), int(round(x3)), int(round(y3))
                pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                M = cv2.getPerspectiveTransform(self.heatmap_loc, pts)
                res = cv2.warpPerspective(self.heatmap, M, (int(self.image_size[2]), int(self.image_size[1])))
                sum_ = np.concatenate((char_gt[np.newaxis, :], res[np.newaxis, :]), axis=0)
                char_gt = np.max(sum_, axis=0)  ##避免标注字符框的重叠

            affine_boxes = datautils.create_affine_boxes(line_box)
            for points in affine_boxes:
                x0, y0, x1, y1, x2, y2, x3, y3 = points[0], points[1], points[2], points[3], points[4], points[5], \
                                                 points[6], points[7]
                pts = np.float32([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])
                M = cv2.getPerspectiveTransform(self.heatmap_loc, pts)
                res = cv2.warpPerspective(self.heatmap, M, (int(self.image_size[2]), int(self.image_size[1])))
                sum_ = np.concatenate((aff_gt[np.newaxis, :], res[np.newaxis, :]), axis=0)
                aff_gt = np.max(sum_, axis=0)  ##避免标注字符框的重叠

        char_gt = cv2.resize(char_gt,
                             (int(self.image_size[2] / self.down_rate), int(self.image_size[1] / self.down_rate)))
        aff_gt = cv2.resize(aff_gt,
                            (int(self.image_size[2] / self.down_rate), int(self.image_size[1] / self.down_rate)))

        img = nd.array(img)
        char_gt = nd.array(char_gt)
        aff_gt = nd.array(aff_gt)
        sample = (img, char_gt, aff_gt)
        return sample

    def __getitem__(self, idx):
        if self.have_Synth_img:
            if idx < len(self.label_list):
                sample = self.get_ReCTS_item(idx)
            else:
                syn_idx = idx - len(self.label_list)
                sample = self.get_SynthText_item(syn_idx)

            return sample
        else:
            return self.get_ReCTS_item(idx)

        
        
class Train_Char_Det():
    def __init__(self,Synth_gt,train=True,pretrained ='Models/ReCTS.params', ReCTS_root='/data3/ml/fansen/162_CV/Train_ocr_det/ReCTs/img',max_update=200000,batch=3):
        self.ctx=mx.gpu()
        self.pretrained = pretrained
        self.batch = batch
        self.max_update=max_update
        self.train=train
        self.ReCTS_root=ReCTS_root
        self._build_model()
        if self.train:
            assert(self.ctx==mx.gpu())
            try:
                self.gt_file = scio.loadmat(Synth_gt)
            except:
                self.gt_file=None
            
            self._build_data()
            self._build_opt()
        
    def _build_data(self):

        H = random.randint(8,12)
        self.data_set = CharDet_Data(Synth_gt=self.gt_file,image_size=(3, 48*H, 64*H),ReCTS_img_root=self.ReCTS_root)
        self.data_iter = data.DataLoader(self.data_set , batch_size=self.batch, shuffle=True,last_batch="discard", num_workers = 3)
    
    def _build_model(self):
        
        self.net = CharDet(train_init=False)
        self.net.load_parameters(self.pretrained, allow_missing=False, ignore_extra=False,ctx=self.ctx)
        
    def _build_opt(self):
        self.epochos = self.max_update//len(self.data_iter )
        
        schedule = mx.lr_scheduler.CosineScheduler(base_lr=0.03,final_lr=0.0001,max_update=self.max_update,warmup_steps=500)
        sgd_optimizer = mx.optimizer.SGD(learning_rate=0.03,momentum=0.9,wd=0.0005 ,lr_scheduler=schedule)
        
        self.trainer = gluon.Trainer(self.net.collect_params(),optimizer=sgd_optimizer)
        self.loss_MSE = gluon.loss.L2Loss()

    def check_iter(self):
        assert(self.train)
        if not os.path.isdir("check_iter"):
            os.mkdir('check_iter')
        for i,batch_data in enumerate(self.data_iter):

            batch_npimg=batch_data[0].asnumpy()
            batch_char=batch_data[1].asnumpy()
            batch_link=batch_data[2].asnumpy()

            img =batch_npimg[0]
            img=np.transpose(img,(1,2,0))

#             plt.imshow(img)
#             plt.show()
#            
            
            mean=(0.406, 0.456, 0.485)
            variance=(0.225, 0.224, 0.229)
            img =  img*np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
            img = img + np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
#             img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
            img=img.astype('uint8')
            cv2.imwrite('check_iter/src.jpg',img)

            char_gt_score =  batch_char[0]
            link_gt_score = batch_link[0]


            T_score=cv2.resize(char_gt_score,(img.shape[1],img.shape[0]))
            L_score=cv2.resize(link_gt_score,(img.shape[1],img.shape[0]))

#             plt.imshow(T_score)
#             plt.show()

            mask = cvt2HeatmapImg(T_score)
            cv2.imwrite('check_iter/T_scoremask.jpg',mask)


            T_score_mask=np.multiply(T_score,255/np.max(T_score))
            T_score_mask=T_score_mask.astype('uint8')
        #     T_score_mask=cv2.cvtColor(T_score_mask,cv2.COLOR_GRAY2BGR)
        #     cv2.imwrite('T_score.jpg',T_score)
#             plt.imshow(L_score)
#             plt.show()


            img=cv2.imread('check_iter/src.jpg')
            word_img=img.copy()
            char_img=img.copy()

            det, labels, mapper = getDetBoxes_core(T_score,L_score,0.7,0.4,0.4)
            for d in det:
                counter=np.array(d,dtype='int32')
                cv2.drawContours(word_img,[counter],-1,(255,255,255),2)
            cv2.imwrite('check_iter/word_imgout.jpg',word_img)

            det, labels, mapper = getDetBoxes_core(T_score,L_score,0.7,10,0.4)
            for d in det:
                counter=np.array(d,dtype='int32')
                cv2.drawContours(char_img,[counter],-1,(255,255,255),2)


            char_img[:,:,2]=T_score_mask #np.multiply(char_img,0.7)+    np.multiply(T_score_mask,0.3) 
            cv2.imwrite('check_iter/char_imgout.jpg',char_img)

            break
            
    def training(self):
        assert(self.train)
        step=0
        train_loss=0
        for epoch in range(self.epochos):
            self._build_data()
            
            print('lr:',self.trainer.learning_rate)
            
            pbar = tqdm(range(len(self.data_iter)))
            for i,batch_data in zip(pbar,self.data_iter):
                batch_img=batch_data[0].as_in_context(mx.gpu())
                batch_char_gt = batch_data[1].as_in_context(mx.gpu())
                batch_link_gt = batch_data[2].as_in_context(mx.gpu())

                with ag.record():
                    out=self.net(batch_img)
                    batch_char_pred=out[:,0,:,:]
                    batch_link_pred=out[:,1,:,:]

                    char_loss = self.loss_MSE(batch_char_pred,batch_char_gt)
                    link_loss = self.loss_MSE(batch_link_pred,batch_link_gt)
                    Loss = char_loss+ link_loss

                Loss.backward()
                self.trainer.step(self.batch)
                step=step+1

                train_loss += Loss.sum().asscalar()
                pbar.set_description(f"epo:{epoch},")
                
                if step%1000==999:
                    net.save_parameters('Models/result.params')

            self.data_iter.__del__()
        print(f"epoch loss :{train_loss}")
        
    def infer(self, Inputname='demo.jpg'):
        
        Max_w_h=2000


        Out_img=Inputname.replace('.jpg','_out.jpg')
        output_mask_name=Inputname.replace('.jpg','_outmask.jpg')
        npimg=cv2.imread(Inputname)
        r =  npimg.shape[0]/npimg.shape[1]
        if max(npimg.shape)>Max_w_h and r>1:
            npimg =cv2.resize(npimg,(int(Max_w_h*(1/r)),Max_w_h))
        if max(npimg.shape)>Max_w_h and r<1:
            npimg =cv2.resize(npimg,(Max_w_h,int(Max_w_h*r)))
            
            
        w=int(npimg.shape[1]/16)
        h=int(npimg.shape[0]/16)
        src=cv2.resize(npimg,(w*16,h*16))
        srccopy=src.copy()
        
        src = normalizeMeanVariance(src)
        src=np.transpose(src,(2,0,1))
        src=np.expand_dims(src,0)
        mxnet_input=nd.array(src,ctx=self.ctx)
        print(mxnet_input.shape)
        out= self.net(mxnet_input).asnumpy()
        
        char_score=out[0][0]
        link_score=out[0][1]

        char_score=cv2.resize(char_score,(char_score.shape[1]*2,char_score.shape[0]*2))
        img = cvt2HeatmapImg(char_score)
        cv2.imwrite(output_mask_name,img)


        img2 = srccopy.copy()
        T_score=cv2.resize(char_score,(w*16,h*16))
        L_score=cv2.resize(link_score,(w*16,h*16))
        det, labels, mapper = getDetBoxes_core(T_score,L_score,0.7,0.4,0.4)



        Countours=[]
        print(len(det))
        for d in det:
            counter=np.array(d,dtype='int32')
            cv2.drawContours(img2,[counter],-1,(0,0,255),2)
            Countours.append(counter)
        cv2.imwrite(Out_img,img2)
