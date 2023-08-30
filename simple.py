import PIL.Image as pilImage
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

import gradio
startup_timer.record("import gradio")

import ldm.modules.encoders.modules
startup_timer.record("import ldm")

from modules import extra_networks, ui_extra_networks_checkpoints
from modules import extra_networks_hypernet, ui_extra_networks_hypernets, ui_extra_networks_textual_inversion
from modules.call_queue import wrap_queued_call, queue_lock, wrap_gradio_gpu_call

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

import modules.img2img
# 1. video 2 image
# 1.1 video split into image, image into right size

# 2. image to nude image
# input : a. src image, b. depth image, c. inpaint mask
# 2.1 create depth image

# 3. nude image

def img2img_controlnet(id_task: str, mode: int, prompt: str, negative_prompt: str, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps: int, sampler_index: int, mask_blur: int, mask_alpha: float, inpainting_fill: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, image_cfg_scale: float, denoising_strength: float, seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, img2img_batch_output_dir: str, img2img_batch_inpaint_mask_dir: str, override_settings_texts, controlnet_unit):
    img2img=modules.img2img.img2img
    return img2img(id_task, mode, prompt, negative_prompt, prompt_styles, init_img, sketch, init_img_with_mask, inpaint_color_sketch, inpaint_color_sketch_orig, init_img_inpaint, init_mask_inpaint, steps, sampler_index, mask_blur, mask_alpha, inpainting_fill, restore_faces, tiling, n_iter, batch_size, cfg_scale, image_cfg_scale, denoising_strength, seed, subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras, height, width, resize_mode, inpaint_full_res, inpaint_full_res_padding, inpainting_mask_invert, img2img_batch_input_dir, img2img_batch_output_dir, img2img_batch_inpaint_mask_dir, override_settings_texts, 0, controlnet_unit)

# 1. 加载人头检测模型（Haar级联分类器）
front_face = cv2.CascadeClassifier('workdir/haarcascade_frontalface_default.xml')
# 2. 加载人头检测模型（Haar级联分类器）
profile_face = cv2.CascadeClassifier('workdir/haarcascade_profileface.xml')

class OneProcess(object):
    def __init__(self, singleName):
        self._sourceImage = PIL.Image.open("workdir/input/"+singleName+".jpg").convert("RGB")
        self._sourceArr = np.asarray(self._sourceImage)
        self._headLine:int = 0
        self._headPath = "workdir/head/"+singleName+".jpg"
        self._pixelImage = None
        self._pixelPath = "workdir/pixel/"+singleName+".jpg"
        self._depthArr = None
        self._depthPath = "workdir/depth/"+singleName+".jpg"
        self._drawArr = None
        self._drawPath = "workdir/draw/"+singleName+".jpg"
        self._finalPath = "workdir/final/"+singleName+".jpg"

    def pixeldepth2draw(self):
        pixelImage = self._pixelImage
        depthArr = self._depthArr
        if pixelImage is None:
            print("WARNING: pixel image is not calculated")
            pixelImage = self._sourceImage
        # controlnet args
        controlnet_unit = controlnet_script.get_default_ui_unit()
        controlnet_unit.enabled = True
        controlnet_unit.processor_res = 512
        controlnet_unit.module = 'none'
        controlnet_unit.model = 'control_sd15_depth [fef5e48e]'
        controlnet_unit.guess_mode = False
        input_img_arr = np.asarray(depthArr)
        mask_arr = np.zeros((input_img_arr.shape[0], input_img_arr.shape[1], 4), dtype=np.uint8)
        mask_arr[:,:,3] = 255
        controlnet_unit.image = dict(image=input_img_arr,mask=mask_arr)
        output_imgs,_,_,_ = img2img_controlnet(id_task="fds", mode=0,
                prompt="""(masterpiece:1.2), (best quality:1.2), photorealistic, 1girl, nude""",
                negative_prompt="""
                paintings, cartoon, rendered, anime, sketches, (worst quality:2), (low quality:2), error,ugly,morbid,mutilated, easynegative, ng_deepnegative_v1_75t
            """,
                prompt_styles=[],
                init_img=pixelImage,
                sketch=None, init_img_with_mask=None, inpaint_color_sketch=None, inpaint_color_sketch_orig=None,
                init_img_inpaint=None, init_mask_inpaint=None,
                steps=20, sampler_index=0, mask_blur=4, mask_alpha=0, inpainting_fill=0, restore_faces=False, tiling=False,
                n_iter=1, batch_size=1, cfg_scale=7, image_cfg_scale=1.5, denoising_strength=0.4,
                seed=1, subseed=-1, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0, seed_enable_extras=False,
                height=768, width=512, resize_mode=0, inpaint_full_res=False, inpaint_full_res_padding=0,
                inpainting_mask_invert=0, img2img_batch_input_dir="", img2img_batch_output_dir="", img2img_batch_inpaint_mask_dir="", override_settings_texts=[], controlnet_unit=controlnet_unit)
        self._drawArr = np.asarray(output_imgs[0])
        output_imgs[0].save(self._drawPath)

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
        result, is_image = pre(HWC3(input_img_arr), res=512, thr_a=64, thr_b=64)
        self._depthArr = np.empty(result.shape + (3,), dtype=np.uint8)
        self._depthArr[:,:,:] = result.reshape(result.shape + (1,))
        cv2.imwrite(self._depthPath, result)

    def source2head(self):
        image = self._sourceArr.copy()
        gray = cv2.cvtColor(self._sourceArr, cv2.COLOR_RGB2GRAY)
        faces = front_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_xywh = None
        for (x, y, w, h) in faces:
            face_xywh = (x, y, w, h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if len(faces) == 0:
            # 3. 运行人头检测模型
            faces = profile_face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in faces:
                face_xywh = (x, y, w, h)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        if face_xywh is None:
            return
        cv2.imwrite(self._headPath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        (x,y,w,h) = face_xywh
        self._headLine = y + h

    def sourcedepthhead2pixel(self):
        line = self._headLine
        depth = self._depthArr
        pixel = self._sourceArr.copy()
        source32 = self._sourceArr.astype(np.int32)
        pixel[line:,:] = (source32[line:,:] + 10 * (source32[line:,:] * (255-depth[line:,:]) + (189, 169, 162) * depth[line:,:]) // 255)//10
        cv2.imwrite(self._pixelPath, cv2.cvtColor(pixel, cv2.COLOR_RGB2BGR))
        self._pixelImage = PIL.Image.open(self._pixelPath)

    def sourceheadraw2final(self):
        final = self._sourceArr.copy()
        line = self._headLine
        draw = self._drawArr
        final[line:,:] = draw[line:,:]
        cv2.imwrite(self._finalPath, cv2.cvtColor(final, cv2.COLOR_RGB2BGR))

    def main(self):
        self.source2head()
        if self._headLine == 0:
            return
        self.source2depth()
        self.sourcedepthhead2pixel()
        self.pixeldepth2draw()
        self.sourceheadraw2final()

if __name__=="__main__":
    initialize()
    for fileName in os.listdir("workdir/input"):
        simpleName = fileName.split(".")[0]
        proc = OneProcess(simpleName)
        proc.main()
