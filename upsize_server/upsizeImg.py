from prepare_images import *
from Models import *
from torchvision.utils import save_image
model_cran_v2 = CARN_V2(color_channels=3, mid_channels=64, conv=nn.Conv2d,
                        single_conv_size=3, single_conv_group=1,
                        scale=2, activation=nn.LeakyReLU(0.1),
                        SEBlock=True, repeat_blocks=3, atrous=(1, 1, 1))
                        
model_cran_v2 = network_to_half(model_cran_v2)
checkpoint = "CARN_model_checkpoint.pt"
model_cran_v2.load_state_dict(torch.load(checkpoint, 'cpu'))
model_cran_v2 = model_cran_v2.cuda(3)
print('upsize load finished')
# if use GPU, then comment out the next line so it can use fp16. 
# model_cran_v2 = model_cran_v2.float() 

# used to compare the origin
# ori_img = "composition.png"
# oriimg = Image.open(ori_img).convert("RGB")
# img = oriimg.resize((img.size[0] // 2, img.size[1] // 2), Image.BICUBIC) 
# img = oriimg
# overlapping split
# if input image is too large, then split it into overlapped patches 
# details can be found at [here](https://github.com/nagadomi/waifu2x/issues/238)
# img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
# img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
# with torch.no_grad():
#     out = [model_cran_v2(i) for i in img_patches]
# img_upscale = img_splitter.merge_img_tensor(out)

# final = torch.cat([img_t, img_upscale])
# save_image(img_upscale, 'out.png', nrow=2)
def upsizeimg(img_url):
    oriimg = Image.open(img_url+'.png').convert("RGB")
    img = oriimg
    img_splitter = ImageSplitter(seg_size=64, scale_factor=2, boarder_pad_size=3)
    img_patches = img_splitter.split_img_tensor(img, scale_method=None, img_pad=0)
    with torch.no_grad():
        out = [model_cran_v2(i.cuda(3)) for i in img_patches]
    img_upscale = img_splitter.merge_img_tensor(out)
    save_image(img_upscale.cpu(), img_url+'_resize.png', nrow=2)
    print('upsized')
if __name__ == "__main__":
    upsizeimg("res230.png")
