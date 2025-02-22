import numpy as np
import cv2
import time
# import torch
import sys
#import onnxruntime

sys.path.insert(0, './demo_with_online_ds/training/')

from typing import Any
# from mss import mss
from PIL import Image
# from models import get_pre_trained_net

class FrameCapture:
    def __init__(self, bounding_box : dict[str,int] = 
                 {'top':  100, 'left': 100, 'width': 256//2, 'height': 256//2},
                 run_model : Any = None,
                 onnx = True) -> None:
        self.bbox = bounding_box
        self.frame_counter = 0
        # self.sct = mss()
        self.duration = None
        self.run_model = run_model
        self.isOnnx = onnx
    def start_capture(self, cancel_key = 'q'):
        if self.run_model is None:
            start_time = time.time()
            
            # Change number based on video input you want!
            cam = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
            # cam = cv2.VideoCapture(0)
            

            if not cam.isOpened():
                print("Cannot open camera")
                exit()

            frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

            while True:
                # sct_img = self.sct.grab(self.bbox)
                # cv2.imshow('screen', np.array(sct_img))
                ret, frame = cam.read()

                #out.write(frame)

                cv2.imshow('Camera', frame)
                
                self.frame_counter += 1

                if (cv2.waitKey(1) & 0xFF) == ord(cancel_key):
                    cv2.destroyAllWindows()
                    end_time = time.time()

                    self.duration = end_time - start_time
                    break
        else:
            if self.isOnnx:
                ...
            else: # run with the model
                ...

    def print_stats(self) -> float:
        if (self.duration is None or self.frame_counter == 0):
            #raise Exception("Please run FrameCapture.start_capture() on your object instance")
            ...
        else:
            fps = self.frame_counter / self.duration
            print(f"Duration : {round(self.duration,3)}")
            print(f"FPS : {round(fps,3)}")

            return fps

def test(fps):
    if fps < 27.:
        raise ValueError("Frame rate below target 30 fps")
    
def run_no_model_example():
    frame_capturer = FrameCapture()
    frame_capturer.start_capture()
    fps_ext = frame_capturer.print_stats()    
    test(fps_ext)

def run_model_example_torch(dir_to_state_dict : str):
    model = get_pre_trained_net()
    model.load_state_dict(torch.load(dir_to_state_dict)["state_dict"])
    model.eval()
    model.cpu()

    frame_capturer = FrameCapture(run_model=model,onnx=False)
    frame_capturer.start_capture()
    fps_ext = frame_capturer.print_stats()    
    test(fps_ext)

class OnnxModel:
    def __init__(self, dir_to_onnx_model : str) -> None:
        self.onnx_session = onnxruntime.InferenceSession(dir_to_onnx_model, providers=["CPUExecutionProvider"])
        self.input_names = self.onnx_session.get_inputs()[0].name
    def __call__(self,x):
        inputs = {self.input_names : (x)}
        return self.onnx_session.run(None, inputs)

def run_onnx_model(dir_to_onnx_model):
    model = OnnxModel(dir_to_onnx_model)

    frame_capturer = FrameCapture(run_model=model,onnx=True)
    frame_capturer.start_capture()
    fps_ext = frame_capturer.print_stats()  
    test(fps_ext)


if __name__ == "__main__":
    run_no_model_example()
    # run_model_example_torch("saved/saved/9c28d30c-b2a7-4299-a54b-5292e8f210e4_0.021.torch")
    #run_onnx_model("saved/fde2bf45-de3f-4d9d-a57e-30188c295bc3_0.067.quant.onnx")
    print("EOF")
