
import PIL.Image as pilImage
import PIL.ImageDraw as pilImageDraw
import os
import shutil
import PIL
import cv2
import numpy as np

import preprocess

controlnet_script = None

DENOISING_STRENGTH = 0.4
CONTROLNET_WEIGHT = 0.9
PROMPT = "(masterpiece:1.2), (best quality:1.2), photorealistic, 1girl, (nude), (nsfw), no clothing"
NE_PROMPT = "paintings, cartoon, rendered, anime, sketches, (worst quality:2), (low quality:2), error,ugly,morbid,mutilated, clothing, easynegative, ng_deepnegative_v1_75t, bad anatomy, bad hands"

import modules.img2img

def img2img_controlnet(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, controlnet_unit):
    img2img=modules.img2img.img2img
    return img2img(id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces, tiling, n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, override_settings_texts, 0, controlnet_unit)

# 1. 加载人头检测模型（Haar级联分类器）
#front_face = cv2.CascadeClassifier('workdir/haarcascade_frontalface_default.xml')
# 2. 加载人头检测模型（Haar级联分类器）
#profile_face = cv2.CascadeClassifier('workdir/haarcascade_profileface.xml')

class OneProcess(preprocess.PreProcess):
    def __init__(self, singleName):
        super().__init__(singleName)
        self._drawArr = None
        self._drawImage = None
        self._drawPath = "workdir/draw/"+singleName+".jpg"
        self._finalPath = "workdir/final/"+singleName+".jpg"

    def sourceheadraw2final(self):
        final = self._sourceArr.copy()
        line = self._headLine
        draw = (self._drawArr & self._inpaintMaskArr) + (final & ~self._inpaintMaskArr)
        final[line:,:] = draw[line:,:]
        cv2.imwrite(self._finalPath, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    def pixeldepth2draw(self):
        pixelImage = self._pixelImage
        depthArr = self._depthArr
        if pixelImage is None:
            print("WARNING: pixel image is not calculated")
            pixelImage = self._sourceImage
        # controlnet args
        controlnet_unit = preprocess.controlnet_script.get_default_ui_unit()
        controlnet_unit.enabled = True
        controlnet_unit.weight = CONTROLNET_WEIGHT
        controlnet_unit.processor_res = self._width
        controlnet_unit.module = 'none'
        controlnet_unit.model = 'control_sd15_depth [fef5e48e]'
        controlnet_unit.guess_mode = False
        input_img_arr = np.asarray(depthArr)
        mask_arr = np.zeros((input_img_arr.shape[0], input_img_arr.shape[1], 4), dtype=np.uint8)
        mask_arr[:,:,3] = 255
        controlnet_unit.image = dict(image=input_img_arr,mask=mask_arr)
        output_imgs,_,_,_ = img2img_controlnet(id_task="fds", mode=0,
                prompt=PROMPT,
                negative_prompt=NE_PROMPT,
                prompt_styles=[],
                init_img=pixelImage,
                sketch=None, init_img_with_mask=None, inpaint_color_sketch=None, inpaint_color_sketch_orig=None,
                init_img_inpaint=None, init_mask_inpaint=None,
                steps=20, sampler_index=16, mask_blur=4, mask_alpha=0, inpainting_fill=0, restore_faces=False, tiling=False,
                n_iter=1, batch_size=1, cfg_scale=7, image_cfg_scale=1.5, denoising_strength=DENOISING_STRENGTH,
                seed=1, subseed=-1, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0, seed_enable_extras=False,
                height=self._height, width=self._width, resize_mode=0, inpaint_full_res=False, inpaint_full_res_padding=0,
                inpainting_mask_invert=0, img2img_batch_input_dir="", img2img_batch_output_dir="", img2img_batch_inpaint_mask_dir="", override_settings_texts=[], controlnet_unit=controlnet_unit)
        self._drawImage = output_imgs[0]
        self._drawArr = np.asarray(self._drawImage)
        self._drawImage.save(self._drawPath)

    def main(self):
        if not self.premain():
            return False
        self.pixeldepth2draw()
        self.sourceheadraw2final()
        return True

class TwoProcess(OneProcess):
    def __init__(self, a, preProc):
        super().__init__(a)
        self._preProc = preProc

    def pixeldepth2draw(self):
        pixelImage = self._pixelImage
        if pixelImage is None:
            print("WARNING: pixel image is not calculated")
            pixelImage = self._sourceImage
        # controlnet args
        controlnet_unit = preprocess.controlnet_script.get_default_ui_unit()
        controlnet_unit.enabled = True
        controlnet_unit.weight = CONTROLNET_WEIGHT
        controlnet_unit.processor_res = self._width
        controlnet_unit.module = 'none'
        controlnet_unit.model = 'control_sd15_depth [fef5e48e]'
        controlnet_unit.guess_mode = False
        depth_arr = np.zeros((self._height, 2*self._width, 3), dtype=np.uint8)
        depth_arr[:,:self._width,:] = np.asarray(self._preProc._depthArr)
        depth_arr[:,self._width:,:] = np.asarray(self._depthArr)
        mask_arr = np.zeros((self._height, 2*self._width, 4), dtype=np.uint8)
        mask_arr[:,:,3] = 255
        controlnet_unit.image = dict(image=depth_arr,mask=mask_arr)
        # pixel args
        pixel_img = pilImage.new("RGB", (self._width*2, self._height))
        pixel_img.paste(self._preProc._drawImage, (0, 0))
        pixel_img.paste(pixelImage, (self._width, 0))
        # mask
        latent_mask = pilImage.new("RGB", (self._width*2, self._height), "black")
        #latent_mask.paste(pilImage.fromarray(self._inpaintMaskArr), (self._width, 0))
        latent_draw = pilImageDraw.Draw(latent_mask)
        latent_draw.rectangle((self._width,self._rawHeadLine,self._width*2, self._height), fill="white")
        output_imgs,_,_,_ = img2img_controlnet(id_task="fds", mode=4,
                                               prompt=PROMPT,
                                               negative_prompt=NE_PROMPT,
                                               prompt_styles=[],
                                               init_img=None,
                                               sketch=None, init_img_with_mask=None, inpaint_color_sketch=None, inpaint_color_sketch_orig=None,
                                               init_img_inpaint=pixel_img, init_mask_inpaint=latent_mask,
                                               steps=20, sampler_index=16, mask_blur=4, mask_alpha=0, inpainting_fill=1, restore_faces=False, tiling=False,
                                               n_iter=1, batch_size=1, cfg_scale=7, image_cfg_scale=1.5, denoising_strength=DENOISING_STRENGTH,
                                               seed=1, subseed=-1, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0, seed_enable_extras=False,
                                               height=self._height, width=self._width*2, resize_mode=0, inpaint_full_res=0, inpaint_full_res_padding=32,
                                               inpainting_mask_invert=0, img2img_batch_input_dir="", img2img_batch_output_dir="", img2img_batch_inpaint_mask_dir="", override_settings_texts=[], controlnet_unit=controlnet_unit)
        self._drawImage = output_imgs[0].crop((self._width, 0, self._width*2, self._height))
        self._drawArr = np.asarray(self._drawImage)
        self._drawImage.save(self._drawPath)

def clearPath(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.mkdir(path)
if __name__=="__main__":
    from p2v import Video
    import os
    preprocess.initialize()
    for videoName in ["zhoukeyi.mp4"]: #, "cyn.mp4", "buerwanji1.mp4", "dujixiao.mp4"]:
        clearPath("workdir/input")
        clearPath("workdir/final")
        video = Video(videoName)
        video.source2input()
        video.input2image()
        firstProc = None
        l = os.listdir("workdir/input")
        l.sort()
        for fileName in l:
            simpleName = fileName.split(".")[0]
            if firstProc is None:
                curProc = OneProcess(simpleName)
                if curProc.main():
                    firstProc = curProc
                    print(videoName, fileName, "finish")
                else:
                    print(videoName, fileName, "ignore")
            else:
                curProc = TwoProcess(simpleName, firstProc)
                #curProc = OneProcess(simpleName)
                if curProc.main():
                    firstProc = curProc
                    print(videoName, fileName, "finish")
                else:
                    print(videoName, fileName, "ignore")
        video.image2immute()
