# -*- coding: utf-8 -*-
import argparse
import os

from docker import Client

from inference import parse_args


def main(model_path, input_path, device, patch_size, batch_size, num_workers):
    client = Client()
    optional = dict()
    model_path, model_name = os.path.split(model_path)
    input_path, input_name = os.path.split(input_path)
    if device.startswith('cuda'):
        optional['runtime'] = 'nvidia'

    print('Model volume location: {model_path}'.format(model_path=model_path))
    print('Input volume location: {input_path}'.format(input_path=input_path))
    volumes = [model_path, input_path]
    volume_bindings = {
        model_path: {'bind': '/mnt/model', 'mode': 'ro'},
        input_path: {'bind': '/mnt/input', 'mode': 'rw'}
    }
    host_config = client.create_host_config(
        binds=volume_bindings,
        ipc_mode='host',
    )
    container = client.create_container(
        image='segmentation:basic',
        command=['python', '-u', 'count.py'],
        # command=['python', '-u', 'inference.py',
        #          '--model', '/mnt/model/{model_name}'.format(model_name=model_name),
        #          '--input', '/mnt/input/{input_name}'.format(input_name=input_name),
        #          '--device', device_,
        #          '--patch_size', str(patch_size_),
        #          '--batch_size', str(batch_size_),
        #          '--num_workers', str(num_workers_)],
        volumes=volumes,
        host_config=host_config,
        **optional
    )
    container_id = container.get('Id')
    response = client.start(container=container_id)
    # while True:
    #     t = client.logs(container_id)
    #     if len(t) > 0:
    #         print(t.decode('utf-8'))

    print('Done!')


if __name__ == '__main__':
    defaults = {
        'batch_size': 4,
        'num_workers': 0,
        'patch_size': 128,
        'input_path': '/home/elavrukhin/Study/binary_data/test_stack',
        'model_path': '/home/elavrukhin/Study/segmentation_pipelines/models/model.torch',
        'device': 'cpu'
    }
    args = parse_args(defaults)
    main(**args)
