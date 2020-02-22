# -*- coding: utf-8 -*-
import os
from itertools import product

from tqdm import tqdm

import imageio
import numpy as np

import torch

from docker_code.args_parse import parse_args
from docker_code.stack_segmentation.stack import Stack, handle_batch
from docker_code.stack_segmentation.training import make_model
from docker_code.stack_segmentation.io import image_process_basic


def predict_on_patches(model, patches, batch_size, device, threshold):
    X = []
    for item in patches:
        x = np.squeeze(item['features'])
        X.append(x[np.newaxis, np.newaxis, :, :])
    X = np.concatenate(X, axis=0)
    batch_num = len(patches) // batch_size + (len(patches) % batch_size != 0)
    offset = 0
    for i in range(batch_num):
        x = X[i * batch_size: (i + 1) * batch_size]
        x = image_process_basic(x).astype(np.float32)
        preds = handle_batch(model=model, item=x, device=device, threshold=threshold)
        for j, pred in enumerate(preds):
            patches[offset + j]['predictions'] = pred
        offset += batch_size
    return patches


def slice_up(data, H, W, D, patch_sizes):
    grids = []
    for dim, patch_size in zip([H, W, D], patch_sizes):
        grids.append(Stack.get_one_dimensional_grid(dim, patch_size))

    patches = []
    for x, y, z in product(*grids):
        patch = {
            'coordinates': [x, y, z],
        }
        selector = tuple(slice(x, x + dx, 1) for x, dx in zip([x, y, z], patch_sizes))
        patch['features'] = data['features'][selector]
        patches.append(patch)
    return patches


def assembly(H, W, D, patches):
    pred = np.zeros((H, W, D), dtype=np.float32)

    for patch in patches:
        selector = tuple(slice(x, x + dx, 1) for x, dx in zip(patch['coordinates'],
                                                              patch['features'].shape))
        pred[selector] = patch['predictions'][:, :, np.newaxis]

    return pred


def inference_loop(
        input_path,
        model,
        patch_size,
        batch_size=1,
        device='cpu',
        threshold=0.5
):
    prefix = os.path.join(input_path, 'NLM')
    files = [os.path.join(prefix, f) for f in sorted(os.listdir(prefix))]
    image = imageio.imread(files[0])[:, :, np.newaxis]
    H, W, D = image.shape

    output_path = os.path.join(input_path, 'preds')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    model.eval()
    with torch.no_grad():
        for i, f in tqdm(enumerate(files), total=len(files), mininterval=10, maxinterval=20):
            image = imageio.imread(f)[:, :, np.newaxis]
            data = {
                'features': image,
                'coordinates': [0, 0, 0]
            }
            patches = slice_up(data=data, H=H, W=W, D=D, patch_sizes=(patch_size, patch_size, 1))
            patches = predict_on_patches(model=model,
                                         patches=patches,
                                         batch_size=batch_size,
                                         device=device,
                                         threshold=threshold)
            pred = assembly(H=H, W=W, D=D, patches=patches)
            output = np.where(pred, 0, 255).astype(np.uint8)
            imageio.imwrite(os.path.join(output_path, 'preds{:04}.bmp'.format(i)), output)

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def inference(model_path, input_path, device, patch_size, batch_size, num_workers):
    model_config = {'source': 'basic'}
    model = make_model(**model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    if num_workers == 0:
        inference_loop(input_path=input_path,
                       model=model,
                       patch_size=patch_size,
                       batch_size=batch_size,
                       device=device)
    else:
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


def main():
    defaults = {
        'batch_size': 4,
        'num_workers': 0,
        'patch_size': 128,
        'input_path': '/mnt/input',
        'model_path': '/mnt/model/model.torch',
        'device': 'cpu'
    }
    args = parse_args(defaults)
    inference(**args)


if __name__ == '__main__':
    main()
