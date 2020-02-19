# -*- coding: utf-8 -*-
import argparse
import os

from docker import Client


def main(model_, input_, device_):
    client = Client()
    optional = dict()
    model_path, model_name = os.path.split(model_)
    input_path, input_name = os.path.split(input_)
    if device_.startswith('cuda'):
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
        # command='/bin/sh -c "while true; do ping 8.8.8.8; done"',
        command=['python', '-u', 'inference.py',
                 '--model', '/mnt/model/{model_name}'.format(model_name=model_name),
                 '--input', '/mnt/input/{input_name}'.format(input_name=input_name),
                 '--device', device_],
        volumes=volumes,
        host_config=host_config,
        **optional
    )
    container_id = container.get('Id')
    response = client.start(container=container_id)
    while True:
        t = client.logs(container_id)
        if len(t) > 0:
            print(t.decode('utf-8'))

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default='/home/elavrukhin/Study/segmentation_pipelines/models/model.torch',
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
    main(args.model, args.input, args.device)
