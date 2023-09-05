
import numpy as np
import os
import cv2

from moviepy.editor import VideoFileClip

class Video(object):
    def __init__(self, fileName):
        self._sourceFileName = "workdir/video_source/"+fileName
        self._inputFileName = "workdir/video_input/"+fileName
        self._immuteFileName = "workdir/video_immute/"+fileName
        self._outputFileName = "workdir/video_output/"+fileName

    def source2input(self):
        clip = VideoFileClip(self._sourceFileName)
        clip = clip.subclip(0, clip.duration-3)
        clip.write_videofile(self._inputFileName)

    def input2image(self):
        cap = cv2.VideoCapture(self._inputFileName)
        if not cap.isOpened():
            print("Cannot open mp4")
            exit()
        NEW_HEIGHT = 512
        for i in range(10000):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # if frame is read correctly ret is True
            if not ret:
                break
            height, width, channel = frame.shape
            newWidth = width*NEW_HEIGHT//height
            frame = cv2.resize(frame, (newWidth, NEW_HEIGHT))
            for fixNewWidth in range(0, 1026, 64):
                if fixNewWidth >= newWidth:
                    break
            padding = np.ones((NEW_HEIGHT,fixNewWidth,3)) * 255
            padding[:,:newWidth,:] = frame
            cv2.imwrite('workdir/input/output_%04d.jpg'%i, padding)

        cap.release()

    def image2immute(self):
        duration = VideoFileClip(self._inputFileName).duration
        image_folder = "workdir/final"
        fps = len(os.listdir(image_folder)) / duration
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort()
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(self._immuteFileName, fourcc, fps, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

    def addAudio(self):

        # 加载目标视频（不包含声音）
        target_video_clip = VideoFileClip(self._immuteFileName).without_audio()

        # 从源视频中提取声音
        audio_clip = VideoFileClip(self._inputFileName).audio

        # 将声音剪辑应用于目标视频
        target_video_clip = target_video_clip.set_audio(audio_clip)

        # 输出最终视频
        target_video_clip.write_videofile(self._outputFileName)

if __name__=="__main__":
    video = Video("xiaoning.mp4")
    video.image2output()
    video.addAudio()
