# -*- coding: utf-8 -*-
import os

import docker

from docker_code.args_parse import parse_args


def main(model_path, input_path, device, patch_size, batch_size, num_workers, test, **kwags):
    optional = dict()
    model_path, model_name = os.path.split(model_path)
    input_path, input_name = os.path.split(input_path)
    # if device.startswith('cuda'):
    #     optional['runtime'] = 'nvidia'
    volume_bindings = {
        model_path: {'bind': '/mnt/model', 'mode': 'ro'},
        input_path: {'bind': '/mnt/input', 'mode': 'rw'}
    }
    if test:
        command = ['python', '-u', 'docker_code/count.py']
        print('Test mode "on"; start counting!')
    else:
        command = ['python', '-u', 'docker_code/inference.py',
                   '--model', '/mnt/model/{model_name}'.format(model_name=model_name),
                   '--input', '/mnt/input/{input_name}'.format(input_name=input_name),
                   '--device', device,
                   '--patch_size', str(patch_size),
                   '--batch_size', str(batch_size),
                   '--num_workers', str(num_workers)]

    print('Model volume location: {model_path}'.format(model_path=model_path))
    print('Input volume location: {input_path}'.format(input_path=input_path))

    client = docker.from_env()
    container = client.containers.run(
        image='segmentation:basic',
        command=command,
        volumes=volume_bindings,
        detach=True,
        **optional
    )
    print('Container id: {id}'.format(id=container.id))
    try:
        for t in container.attach(stdout=True, stream=True, logs=True):
            print(t.decode('utf-8'))
    except KeyboardInterrupt:
        pass
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
