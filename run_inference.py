# -*- coding: utf-8 -*-
import argparse

import docker


def main(model, input):
    client = docker.from_env()
    result = client.containers.run(
        image='anibali/pytorch:no-cuda',
        volumes={
            model: {'bind': '/mnt/model', 'mode': 'ro'},
            input: {'bind': '/mnt/input', 'mode': 'ro'}}
    )
    print(result)


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

    args = parser.parse_args()
    main(args.model, args.input)
