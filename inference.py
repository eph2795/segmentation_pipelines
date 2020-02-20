# -*- coding: utf-8 -*-
import argparse

import torch

from stack_segmentation.stack import Stack
from stack_segmentation.training import make_model


def inference(model_path, input_path, device, patch_size, batch_size, num_workers):
    model_config = {'source': 'basic'}
    model = make_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    stack = Stack.read_from_source(input_path, has_targets=False)
    predicted_stack = stack.apply(model,
                                  model_config,
                                  patch_sizes=(patch_size, patch_size, 1),
                                  bs=batch_size,
                                  num_workers=num_workers,
                                  device=device,
                                  threshold=None)
    predicted_stack.dump(input_path,
                         features=False,
                         targets=False,
                         preds=True)


def parse_args(defaults):
    if defaults is None:
        defaults = dict()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',
                        type=str,
                        default=defaults.get('model_path'),
                        help='Путь к модели, которая будет использоваться для сегментации')
    parser.add_argument('--input_path',
                        type=str,
                        default=defaults.get('input_path'),
                        help='Путь к стекам, которые требуется сегментировать')
    parser.add_argument('--device',
                        type=str,
                        default=defaults.get('cpu'),
                        help='На каком устройстве будет производиться расчет: "cpu" или "cuda"')
    parser.add_argument('--patch_size',
                        type=int,
                        default=defaults.get('patch_size'))
    parser.add_argument('--batch_size',
                        type=int,
                        default=defaults.get('batch_size'))
    parser.add_argument('--num_workers',
                        type=int,
                        default=defaults.get('num_workers'))
    args = parser.parse_args()
    return {
        'model_path': args.model,
        'input_path': args.input,
        'device': args.device,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers
    }


def main():
    defaults = {
        'batch_size': 4,
        'num_workers': 0,
        'patch_size': 128,
        'input_path': '/opt/input',
        'model_path': 'opt/model/model.torch'
    }
    args = parse_args(defaults)
    inference(**args)


if __name__ == '__main__':
    main()
