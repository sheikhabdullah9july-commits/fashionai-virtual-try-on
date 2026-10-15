import pdb
from pathlib import Path
import sys
import os
import onnxruntime as ort
PROJECT_ROOT = Path(__file__).absolute().parents[0].absolute()
sys.path.insert(0, str(PROJECT_ROOT))
from parsing_api import onnx_inference
import torch


class Parsing:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        # صرف اس وقت CUDA set کریں جب CUDA available ہو اور gpu_id valid ہو
        self.use_cuda = torch.cuda.is_available() and gpu_id is not None and int(gpu_id) >= 0
        if self.use_cuda:
            torch.cuda.set_device(int(gpu_id))

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # CUDA available ہو تو CUDAExecutionProvider استعمال کریں، ورنہ CPU
        if self.use_cuda:
            providers = [('CUDAExecutionProvider', {'device_id': int(gpu_id)}), 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        self.session = ort.InferenceSession(
            os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_atr.onnx'),
            sess_options=session_options,
            providers=providers,
        )
        self.lip_session = ort.InferenceSession(
            os.path.join(Path(__file__).absolute().parents[2].absolute(), 'checkpoints/humanparsing/parsing_lip.onnx'),
            sess_options=session_options,
            providers=providers,
        )

    def __call__(self, input_image):
        if self.use_cuda:
            torch.cuda.set_device(int(self.gpu_id))
        parsed_image, face_mask = onnx_inference(self.session, self.lip_session, input_image)
        return parsed_image, face_mask
