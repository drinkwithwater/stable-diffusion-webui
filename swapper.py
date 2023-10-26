import copy
import math
import os
import tempfile
from dataclasses import dataclass
from typing import List, Union, Dict, Set, Tuple

import cv2
import numpy as np
from PIL import Image

import tqdm
import insightface
import onnxruntime

from moviepy.editor import VideoFileClip
providers = ["CUDAExecutionProvider"]
#providers = ["CPUExecutionProvider"]


FS_MODEL = None
CURRENT_FS_MODEL_PATH = None

FACE_ANALYSIS = None


def getFaceSwapModel(model_path: str):
    global FS_MODEL
    global CURRENT_FS_MODEL_PATH
    if CURRENT_FS_MODEL_PATH is None or CURRENT_FS_MODEL_PATH != model_path:
        CURRENT_FS_MODEL_PATH = model_path
        FS_MODEL = insightface.model_zoo.get_model(model_path, providers=providers)
    return FS_MODEL

def get_faces_all(img_data: np.ndarray, det_size=(640, 640)):
    global FACE_ANALYSIS
    if FACE_ANALYSIS is None:
        FACE_ANALYSIS = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
        FACE_ANALYSIS.prepare(ctx_id=0, det_size=det_size)
    face_analyser = FACE_ANALYSIS
    face = face_analyser.get(img_data)

    if len(face) == 0 and det_size[0] > 320 and det_size[1] > 320:
        det_size_half = (det_size[0] // 2, det_size[1] // 2)
        return get_faces_all(img_data, det_size=det_size_half)

    return list(filter(lambda f:f.sex=="F", face))


def get_face_single(img_data: np.ndarray, last_bbox=None):
    faces = get_faces_all(img_data)
    if len(faces) == 1:
        return faces[0]
    elif len(faces) <= 0:
        return None
    else:
        if last_bbox is None:
            max_color = 0
            max_index = 0
            for i, face in enumerate(faces):
                x,y,x1,y1 = np.floor(face.bbox).astype(np.int32)
                wIn4 = (x1-x)//4
                hIn4 = y1-y//4
                arr = img_data[y+hIn4:y1-hIn4,x+wIn4:x1-wIn4]
                cur_color = np.mean(arr)
                if cur_color > max_color:
                    max_color = cur_color
                    max_index = i
            return faces[max_index]
        else:
            min_dis = 0x7FFFFFFF
            min_index = 0
            x,y,x1,y1 = np.floor(last_bbox).astype(np.int32)
            x0 = (x1+x)//2
            y0 = (y1+y)//2
            for i, face in enumerate(faces):
                x,y,x1,y1 = np.floor(face.bbox).astype(np.int32)
                x = (x1+x)//2
                y = (y1+y)//2
                dis = (x-x0)**2 + (y-y0) **2
                if dis < min_dis:
                    min_dis = dis
                    min_index = i
            return faces[min_index]

def swap_face(source_face, target_img: np.ndarray, last_bbox=None)->np.ndarray:
    model = "./models/roop/inswapper_128.onnx"
    model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), model)
    face_swapper = getFaceSwapModel(model_path)

    target_face = get_face_single(target_img, last_bbox)
    if target_face is not None:
        result_arr = face_swapper.get(target_img, target_face, source_face)
        return result_arr, target_face.bbox
    else:
        #print(f"No target face found")
        return target_img, None


class Main(object):
    def __init__(self, face_img, video_file, fps):
        self.fps = fps
        output_file = face_img.split(".")[0] + "_" + video_file.split(".")[0]+".mp4"
        self._faceImagePath = "workspace/face/"+face_img
        directory = str(self.fps) if self.fps in [30 , 25] else "other"
        self._inputVideoPath = f"workdir/video_input/{directory}/{video_file}"
        self._middleVideoPath = "workdir/video_middle/" + output_file
        self._outputVideoPath = "workdir/video_output/" + output_file

    def video(self):
        source_img = cv2.imread(self._faceImagePath)
        source_face = get_face_single(source_img)
        bbox = None
        capture = cv2.VideoCapture(self._inputVideoPath)
        if not capture.isOpened():
            print("Cannot open mp4")
            exit()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = None
        for i in tqdm.tqdm(range(1000000)):
            # Capture frame-by-frame
            ret, frame = capture.read()
            if writer is None:
                height, width = frame.shape[0], frame.shape[1]
                fps = 30 if self.fps == 60 else self.fps
                writer = cv2.VideoWriter(self._middleVideoPath, fourcc, fps, (width, height))
            # if frame is read correctly ret is True
            if not ret:
                break
            if self.fps == 60 and i % 2 == 0:
                continue
            result_arr, bbox = swap_face(source_face, frame, bbox)
            if result_arr is None:
                continue
            writer.write(result_arr)

        writer.release()
        capture.release()

    def audio(self):

        # 加载目标视频（不包含声音）
        middle_clip = VideoFileClip(self._middleVideoPath).without_audio()

        # 从源视频中提取声音
        audio_clip = VideoFileClip(self._inputVideoPath).subclip(0, middle_clip.duration).audio

        # 将声音剪辑应用于目标视频
        output_clip = middle_clip.set_audio(audio_clip)

        # 输出最终视频
        output_clip.write_videofile(self._outputVideoPath)


    def test(self):
        for i, img_path in enumerate(os.listdir("workspace/face")):
            print(i, img_path)
            for target_path in ["target1.png", "target2.png", "target3.png"]:
                inpath = "workspace/face/"+img_path
                #with open("workspace/face/"+img_path, "rb") as fi:
                #    buf = np.asarray(bytearray(fi.read()), np.uint8)
                #    source_img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                source_img = cv2.imread(inpath)
                source_face = get_face_single(source_img)
                target = cv2.imread("workspace/target/"+target_path)
                result_arr = swap_face(source_face, target)
                target_base = os.path.basename(target_path).split(".")[0]
                img_base = img_path.split(".")[0]
                outpath = "workspace/result/"+target_base+"_"+img_base+".jpg"
                cv2.imwrite(outpath, result_arr)
