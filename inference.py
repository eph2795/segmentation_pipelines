# -*- coding: utf-8 -*-
import argparse

import torch

from stack_segmentation.stack import Stack
from stack_segmentation.training import make_model


def inference(model_path, input_path, device, patch_size, batch_size, num_workers):
    model_config = {'source': 'basic'}
    model = make_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
    parser.add_argument('--patch_size',
                        type=int,
                        default=128)
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--num_workers',
                        type=int,
                        default=2)
    args = parser.parse_args()
    inference(args.model,
              args.input,
              args.device,
              args.patch_size,
              args.batch_size,
              args.num_workers)