from flask import Flask, render_template, request, redirect, url_for, send_file, send_from_directory
from flask_restful import Resource, Api, reqparse
from werkzeug.utils import secure_filename

import cv2
import base64
import os
import sys
import time
import math
import json
import numpy as np

from utils.prepare_images import *
from models.Models import *
from torchvision.utils import save_image
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model_cran_v2 = network_to_half(model_cran_v2)

"""
config_path = "Config.json"
with open(config_path,'r') as file:
    config = json.load(file)
"""
# checkpoint = config["checkpoint"]
checkpoint = "CARN_model_checkpoint.pt"

model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
model_cran_v2 = model_cran_v2.cuda(0)
print('upsize load finished')
# target_w = config["size"]
# RGB_Inpaint = RGB_Inpaint(config)
# Alpha_Inpaint = Alpha_Inpaint(config)

def upsizeimg(oriimg):
    # oriimg = Image.open(img_url+'.png').convert("RGB")
    img = oriimg
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model_cran_v2(i.cuda(3)) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    # save_image(img_upscale.cpu(), img_url+'_resize.png', nrow=2)
    print('upsized')
    return img_upscale.cpu()
app = Flask(__name__, static_folder='static')
project_folder = 'projects'
api = Api(app)


class CreateUpsize(Resource):

    def post(self):

        ctime = time.time()
        image_str = base64.b64decode(request.form['img'])
        # mask_str = base64.b64decode(request.form['mask'])
        image = self.decode_image(image_str)
        # mask = self.decode_image(mask_str)

        h, w, c = image.shape
        print(image.shape)
        if c == 4:
            image = Image.fromarray(cv2.cvtColor(image[:,:,0:3],cv2.COLOR_BGR2RGB))
            alpha = image[:,:,3]
        else:
            image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        result = upsizeimg(image)
        return {"result":self.encode_image_png(result)}
        # image = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        # image = cv2.resize(image, (target_w, target_h))
        # image_rgb = image[:,:,:3]
        # if c>3:
        #     image_alpha = np.expand_dims(image[:,:,3],2)
        # mask = cv2.resize(mask, (target_w, target_h))
        # mask = np.expand_dims(mask,2)

        # rgb_input_image = np.concatenate(
        #     [np.expand_dims(image_rgb, 0), np.expand_dims(np.concatenate([mask,mask,mask],axis=2), 0)], axis=2)

        # print('pre process time {}'.format(time.time() - ctime))
        # ctime = time.time()

        # #generative inpaint
        # gen_result_1 = RGB_Inpaint.rgb_inpaint1(rgb_input_image)
        # gen_result_2 = RGB_Inpaint.rgb_inpaint2(rgb_input_image)
        # # opencv inpaint
        # opencv_result = cv2.inpaint(
        #     image_rgb, mask[:, :, 0], 3, cv2.INPAINT_TELEA)
        # results = [gen_result_1,gen_result_2,opencv_result]
        # print('inpaint time {}'.format(time.time() - ctime))
        # ctime = time.time()

        # if c>3:
        #     #alpha inpaint
        #     for i in range(len(results)):
        #         result = np.concatenate([results[i],image_alpha],2)
        #         # print(result.shape)
        #         # print(mask.shape)
        #         results[i] = Alpha_Inpaint.add_alpha_func(result,mask[:,:,0])
        #     print('alpha inpaint time {}'.format(time.time() - ctime))
        #     ctime = time.time()

        # result_texts=[]
        # for result in results:
        #     result_resized = cv2.resize(result,(w,h))
        #     result_texts.append(self.encode_image_png(result_resized))

        # print('post process time {}'.format(time.time() - ctime))
        # ctime = time.time()

        # return {'res1': result_texts[0], 'res2': result_texts[1], 'res3': result_texts[2]}

    def decode_image(self, img_string):
        image_np = np.fromstring(img_string, np.uint8)
        image = cv2.imdecode(image_np, cv2.IMREAD_UNCHANGED)
        return image

    def encode_image_png(self, img):
        _, buf = cv2.imencode('.png', img)
        buf = buf.tobytes()
        result_text = base64.b64encode(buf)
        result_text = result_text.decode('utf-8')
        return result_text



api.add_resource(CreateUpsize, '/upsize/create_upsize')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3028)