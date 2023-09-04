
from yolo8_hd.head import getHeadxywh
from yolo8_hd.body import getMask

import PIL.Image as pilImage
import PIL.ImageDraw as pilImageDraw
import os
import PIL
import cv2
import numpy as np
import re
import warnings
from modules import modelloader

from modules import paths, timer, import_hook, errors
import modules.sd_models
import modules.scripts
from modules import shared, devices, sd_samplers, upscaler, extensions, localization, ui_tempdir, ui_extra_networks

startup_timer = timer.Timer()

import torch
import pytorch_lightning # pytorch_lightning should be imported after torch, but it re-enables warnings on import so import once to disable them
warnings.filterwarnings(action="ignore", category=DeprecationWarning, module="pytorch_lightning")
startup_timer.record("import torch")

import ldm.modules.encoders.modules
startup_timer.record("import ldm")

controlnet_script = None

def initialize():
    global controlnet_script
    extensions.list_extensions()
    startup_timer.record("list extensions")

    modelloader.cleanup_models()
    modules.sd_models.setup_model()
    startup_timer.record("list SD models")

    modules.scripts.load_scripts()
    startup_timer.record("load scripts")

    modules.sd_models.load_model()
    startup_timer.record("load SD checkpoint")

    modules.scripts.scripts_img2img.initialize_scripts(True)

    # controlnet script
    controlnet_script = modules.scripts.scripts_img2img.alwayson_scripts[0]
    controlnet_script.args_from = 1
    controlnet_script.args_to = 2

class PreProcess(object):
    def __init__(self, singleName):
        self._sourceImage = PIL.Image.open("workdir/input/"+singleName+".jpg").convert("RGB")
        self._sourceArr = np.asarray(self._sourceImage)
        self._shape = self._sourceArr.shape
        self._height, self._width, _ = self._shape
        self._headLine:int = 0
        self._headPath = "workdir/head/"+singleName+".jpg"
        self._pixelImage = None
        self._pixelPath = "workdir/pixel/"+singleName+".jpg"
        self._depthArr = None
        self._depthPath = "workdir/depth/"+singleName+".jpg"
        self._segmentArr = None
        self._segmentPath = "workdir/segment/"+singleName+".jpg"
        self._inpaintMaskArr = None
        self._inpaintMaskPath = "workdir/mask/"+singleName+".jpg"

    def pixeldepth2draw(self):
        pixelImage = self._pixelImage
        depthArr = self._depthArr
        if pixelImage is None:
            print("WARNING: pixel image is not calculated")
            pixelImage = self._sourceImage
        # controlnet args
        controlnet_unit = controlnet_script.get_default_ui_unit()
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
                steps=20, sampler_index=0, mask_blur=4, mask_alpha=0, inpainting_fill=0, restore_faces=False, tiling=False,
                n_iter=1, batch_size=1, cfg_scale=7, image_cfg_scale=1.5, denoising_strength=DENOISING_STRENGTH,
                seed=1, subseed=-1, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0, seed_enable_extras=False,
                height=self._height, width=self._width, resize_mode=0, inpaint_full_res=False, inpaint_full_res_padding=0,
                inpainting_mask_invert=0, img2img_batch_input_dir="", img2img_batch_output_dir="", img2img_batch_inpaint_mask_dir="", override_settings_texts=[], controlnet_unit=controlnet_unit)
        self._drawImage = output_imgs[0]
        self._drawArr = np.asarray(self._drawImage)
        self._drawImage.save(self._drawPath)

    def source2depth(self):
        input_img_arr = np.asarray(self._sourceImage.convert("RGB"))
        def HWC3(x):
            assert x.dtype == np.uint8
            if x.ndim == 2:
                x = x[:, :, None]
            assert x.ndim == 3
            H, W, C = x.shape
            assert C == 1 or C == 3 or C == 4
            if C == 3:
                return x
            if C == 1:
                return np.concatenate([x, x, x], axis=2)
            if C == 4:
                color = x[:, :, 0:3].astype(np.float32)
                alpha = x[:, :, 3:4].astype(np.float32) / 255.0
                y = color * alpha + 255.0 * (1.0 - alpha)
                y = y.clip(0, 255).astype(np.uint8)
                return y
        pre = controlnet_script.preprocessor["depth"]
        result, is_image = pre(HWC3(input_img_arr), res=self._width, thr_a=64, thr_b=64)
        self._depthArr = np.empty(result.shape + (3,), dtype=np.uint8)
        self._depthArr[:,:,:] = result.reshape(result.shape + (1,))
        cv2.imwrite(self._depthPath, result)

    def source2seg(self):
        input_img_arr = np.asarray(self._sourceImage.convert("RGB"))
        self._segmentArr = getMask(input_img_arr)
        cv2.imwrite(self._segmentPath, self._segmentArr)

    def depthseg2mask(self):
        fatMask = self._segmentArr.copy()
        fatWidth = self._width//20
        fatHeight = self._height//20
        fatMask = np.zeros((self._height+fatHeight, self._width+2*fatWidth, 3), dtype=np.int32)
        for y in range(2*fatWidth+1):
            for x in range(fatHeight+1):
                fatMask[x:self._height+x,y:self._width+y] += self._segmentArr
        fatMask = fatMask[:self._height,fatWidth:self._width+fatWidth]
        finalMask = fatMask * (self._depthArr > 128) * (self._sourceArr != [255,255,255])
        finalMask += self._segmentArr
        finalMask = ((finalMask > 0) * 255).astype(np.uint8)
        self._inpaintMaskArr = finalMask
        cv2.imwrite(self._inpaintMaskPath, self._inpaintMaskArr)

    def source2head(self):
        image = self._sourceArr.copy()
        x, y, w, h = getHeadxywh(image)
        if x is None:
            return
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite(self._headPath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        self._headLine = y + h

    def sourcedepthhead2pixel(self):
        line = self._headLine
        depth = self._depthArr
        pixel = self._sourceArr.copy()
        source32 = self._sourceArr.astype(np.int32)
        pixel[line:,:] = (source32[line:,:] + 9 * (source32[line:,:] * (255-depth[line:,:]) + (189, 169, 162) * depth[line:,:]) // 255)//10
        cv2.imwrite(self._pixelPath, cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR))
        self._pixelImage = PIL.Image.open(self._pixelPath)

    def getSkinHairMask(self):
        # calc skin
        yuv = cv2.cvtColor(self._sourceArr, cv2.COLOR_RGB2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(yuv, lower_skin, upper_skin).reshape((self._height, self._width, 1))
        # calc hair
        gray = cv2.cvtColor(self._sourceArr, cv2.COLOR_RGB2GRAY)
        hair_mask = ((gray < 50) * 255).astype(np.uint8).reshape((self._height, self._width, 1))
        # calc average skin color
        #skin = self._sourceArr & skin_mask
        #skin[:self._headLine] = 0
        #gray = cv2.cvtColor(skin, cv2.COLOR_RGB2GRAY)
        #calc_mask = gray > 100
        #calc = calc_mask.reshape(calc_mask.shape + (1,)) * skin
        #color = np.sum(calc, axis=(0,1)) /np.sum(calc_mask)
        return skin_mask, hair_mask

    def sourceseghead2pixel(self):
        line = self._headLine
        skin_mask, hair_mask = self.getSkinHairMask()
        mask = (self._segmentArr & ~skin_mask & ~hair_mask).astype(np.int32)
        pixel = self._sourceArr.copy()
        source32 = self._sourceArr.astype(np.int32)
        source32Out = source32 * (255-mask)//255
        source32In = source32 * mask//255
        source32In = (source32In + 9*np.array([189, 169, 162])*mask//255)//10
        pixel[line:,:] = source32Out[line:,:] + source32In[line:,:]
        cv2.imwrite(self._pixelPath, cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR))
        self._pixelImage = PIL.Image.open(self._pixelPath)

    def premain(self):
        self.source2head()
        if self._headLine == 0:
            return False
        self.source2depth()
        self.source2seg()
        self.depthseg2mask()
        #self.sourcedepthhead2pixel()
        self.sourceseghead2pixel()
        return True

    def main(self):
        self.premain()

if __name__=="__main__":
    initialize()
    l = os.listdir("workdir/input")
    l.sort()
    for fileName in l:
        simpleName = fileName.split(".")[0]
        curProc = PreProcess(simpleName)
        curProc.main()
        print(fileName, "finish")
