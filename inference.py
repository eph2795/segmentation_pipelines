# -*- coding: utf-8 -*-
import argparse

import torch

from stack_segmentation.stack import Stack
from stack_segmentation.training import make_model


def inference(model_path, input_path, device):
    model_config = {'source': 'basic'}
    model = make_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    stack = Stack.read_from_source(input_path, has_targets=False)
    predicted_stack = stack.apply(model,
                                  model_config,
                                  patch_sizes=(128, 128, 1),
                                  bs=32, num_workers=8,
                                  device=device,
                                  threshold=None)
    predicted_stack.dump(input_path,
                         features=False,
                         targets=False,
                         preds=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default=None,
                        help='Путь к модели, которая будет использоваться для сегментации')
    parser.add_argument('--input',
                        type=str,
                        default=None,
                        help='Путь к стекам, которые требуется сегментировать')
    parser.add_argument('--device',
                        type=str,
                        default='cpu',
                        help='На каком устройстве будет производиться расчет: "cpu" или "cuda"')
    args = parser.parse_args()
    inference(args.model, args.input, args.device)