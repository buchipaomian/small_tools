from PIL import Image
import numpy as np
import colorsys
import copy

img_ori = Image.open("leg3.jpg")
img = img_ori.convert('RGBA')
img1 = img_ori.convert('RGBA')

def get_main_color(image):
    image.thumbnail((200, 200))
    max_score = 0#原来的代码此处为None
    dominant_color = 0#原来的代码此处为None，但运行出错，改为0以后 运行成功，原因在于在下面的 score > max_score的比较中，max_score的初始格式不定
    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # 跳过纯黑色
        if a == 0:
            continue
        if (r,g,b) == (255,255,255):
            continue
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        y = (y - 16.0) / (235 - 16)
        # 忽略高亮色
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color = [r, g, b]
    return dominant_color
def seperate_shadow(image, color):
    # image = image.convert('RGB')
    image_target = np.array(image)#this is the array of image
    background_img = copy.deepcopy(image_target)
    # print(image_target)
    for x,line in enumerate(image_target):
        for y,point in enumerate(line):
            if (point[0],point[1],point[2]) == (255,255,255):
                continue
            if [point[0],point[1],point[2]] == color:
                image_target[x][y] = [255,255,255,point[3]]
                #在这之后需要处理一批在范围内的
            else:
                differ = (abs(point[0]-color[0])+abs(point[1]-color[1])+abs(point[2]-color[2]))/3
                if differ <= 15:
                    image_target[x][y] = [255,255,255,point[3]]
                else:
                    background_img[x][y] = [color[0],color[1],color[2],point[3]]
            # else:
            #     print(point)
    # print(image_target)
    result = Image.fromarray(image_target)
    background = Image.fromarray(background_img)
    return result,background
def shadow_generate(imagename):
    img_resource = Image.open(imagename)
    img_temp = img_resource.convert('RGBA')
    img_temp1 = img_resource.convert('RGBA')
    result = seperate_shadow(img_temp1,get_main_color(img_temp))
    return result
    #this function is trying to seperate the shadow from ori image

# print(get_main_color(img))
final,background = seperate_shadow(img1,get_main_color(img))
final.save("shadow.png")