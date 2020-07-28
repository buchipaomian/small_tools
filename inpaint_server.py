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

from net.alpha_inpaint import Alpha_Inpaint

from net.rgb_inpaint import RGB_Inpaint

config_path = "Config.json"
with open(config_path,'r') as file:
    config = json.load(file)
target_h = config["size"]
target_w = config["size"]
RGB_Inpaint = RGB_Inpaint(config)
Alpha_Inpaint = Alpha_Inpaint(config)

app = Flask(__name__, static_folder='static')
project_folder = 'projects'
api = Api(app)


class CreateInpaint(Resource):

    def post(self):

        ctime = time.time()
        image_str = base64.b64decode(request.form['img'])
        mask_str = base64.b64decode(request.form['mask'])
        image = self.decode_image(image_str)
        mask = self.decode_image(mask_str)

        h, w, c = image.shape
        print(image.shape)
        image = cv2.resize(image, (target_w, target_h))
        image_rgb = image[:,:,:3]
        if c>3:
            image_alpha = np.expand_dims(image[:,:,3],2)
        mask = cv2.resize(mask, (target_w, target_h))
        mask = np.expand_dims(mask,2)

        rgb_input_image = np.concatenate(
            [np.expand_dims(image_rgb, 0), np.expand_dims(np.concatenate([mask,mask,mask],axis=2), 0)], axis=2)

        print('pre process time {}'.format(time.time() - ctime))
        ctime = time.time()

        #generative inpaint
        gen_result_1 = RGB_Inpaint.rgb_inpaint1(rgb_input_image)
        gen_result_2 = RGB_Inpaint.rgb_inpaint2(rgb_input_image)
        # opencv inpaint
        opencv_result = cv2.inpaint(
            image_rgb, mask[:, :, 0], 3, cv2.INPAINT_TELEA)
        results = [gen_result_1,gen_result_2,opencv_result]
        print('inpaint time {}'.format(time.time() - ctime))
        ctime = time.time()

        if c>3:
            #alpha inpaint
            for i in range(len(results)):
                result = np.concatenate([results[i],image_alpha],2)
                # print(result.shape)
                # print(mask.shape)
                results[i] = Alpha_Inpaint.add_alpha_func(result,mask[:,:,0])
            print('alpha inpaint time {}'.format(time.time() - ctime))
            ctime = time.time()

        result_texts=[]
        for result in results:
            result_resized = cv2.resize(result,(w,h))
            result_texts.append(self.encode_image_png(result_resized))

        print('post process time {}'.format(time.time() - ctime))
        ctime = time.time()

        return {'res1': result_texts[0], 'res2': result_texts[1], 'res3': result_texts[2]}

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



api.add_resource(CreateInpaint, '/inpaint/create_inpaint')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3003)
