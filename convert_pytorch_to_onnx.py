try:
    import sys

    if '/opt/project/common' not in sys.path:
        sys.path.append('common')

    from config_overrider import override_config_using_model_loading_scheme_config
    override_config_using_model_loading_scheme_config()
except Exception:
    raise

import pandas as pd
import numpy as np
from config import GlobalConfig
import torch
import time
import onnx
import onnxruntime
from transfuser.model import TransFuser
import argparse
import warnings
from transfuser.model_loading_scheme import ModelLoadingScheme
from transfuser.model_loading_scheme import parse_model_loading_scheme
from config import RunConfig
from transfuser_common import bind_input_and_output
from transfuser_common import to_numpy

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


def get_dummy_input():
    x = {
        'speed': torch.rand((1,)).to(device),
        'target_point': torch.rand((1, 2)).to(device),
        'lidar': torch.rand((1, 2, 256, 256)).to(device)
    }

    if GlobalConfig.should_use_3_cams:
        x['img'] = torch.rand((1, 3, 256, 256)).to(device)
        x['img2'] = torch.rand((1, 3, 256, 256)).to(device)
        x['img3'] = torch.rand((1, 3, 256, 256)).to(device)
    else:
        x['img'] = torch.rand((1, 3, 256, 256)).to(device)
        x['img2'] = torch.rand((1,)).to(device)
        x['img3'] = torch.rand((1,)).to(device)

    return x


def export_pytorch_to_onnx(_model, _x):
    # Export the model
    torch.onnx.export(_model,  # model being run
                      _x,  # model input, dummy input
                      args.onnx_model_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=13,  # the ONNX version (pytorch 1.12.0)
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['speed', 'target_point', 'img', 'img2', 'img3', 'lidar'],  # the model's input names
                      output_names=['output'],  # the model's output names
                      dynamic_axes={'speed': {0: 'batch_size'},
                                    'target_point': {0: 'batch_size'},
                                    'img': {0: 'batch_size'},
                                    'img2': {0: 'batch_size'},
                                    'img3': {0: 'batch_size'},
                                    'lidar': {0: 'batch_size'},
                                    'output': {0: 'batch_size'}},  # variable length axes
                      verbose=False)


#########################################################################
# Requirements: torch==1.12.0   torchvision==0.13.0
#########################################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_model_path', type=str, required=True)
    parser.add_argument('--model_loading_scheme', type=str, required=True)
    args = parser.parse_args()

    torch.cuda.empty_cache()

    model = TransFuser(True, GlobalConfig(), device, 4).to(device)

    run_config = RunConfig()
    run_config.batch_size = 1
    run_config.should_shuffle_train = False
    model_loading_scheme = ModelLoadingScheme(parse_model_loading_scheme(args.model_loading_scheme, run_config))
    last_epoch = model_loading_scheme.load_state_dicts(model, None, None)

    model.eval()

    # Input example
    x = get_dummy_input()

    # Measure inference and get output for onnx model evaluation
    time_torch = time.time()
    for i in range(1):
        out = model(**x)
    print(f'Pytorch time: {time.time() - time_torch}')

    export_pytorch_to_onnx(model, x)
    torch.cuda.empty_cache()

    # Load and check onnx model
    onnx_model = onnx.load(args.onnx_model_path)
    onnx.checker.check_model(onnx_model)

    # Give providers (GPU and CPU) some operators will be done by CPU instead of GPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Set optimization
    so = onnxruntime.SessionOptions()
    so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Create session
    ort_session = onnxruntime.InferenceSession(args.onnx_model_path, so, providers=providers)
    options = ort_session.get_provider_options()
    cuda_options = options['CUDAExecutionProvider']
    cuda_options['cudnn_conv_use_max_workspace'] = '1'
    ort_session.set_providers(['CUDAExecutionProvider'], [cuda_options])

    # Generate input output data
    io_binding = bind_input_and_output(x, ort_session)

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x['speed']),
    #               ort_session.get_inputs()[1].name: to_numpy(x['target_point']),
    #               ort_session.get_inputs()[2].name: to_numpy(x['img']),
    #               ort_session.get_inputs()[3].name: to_numpy(x['lidar'])}

    # Warming-up (first run takes 10x more time that other runs)
    ort_session.run_with_iobinding(io_binding)
    ort_outs = io_binding.copy_outputs_to_cpu()

    # Run onnx model with binded input output data and measure time
    time_onnx = time.time()
    for i in range(1):
        io_binding = bind_input_and_output(x, ort_session)
        ort_session.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        # ort_outs = ort_session.run(None, ort_inputs)
    print(f'Onnx time: {time.time() - time_onnx}')

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
