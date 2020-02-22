import argparse


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
                        default=defaults.get('device'),
                        help='На каком устройстве будет производиться расчет: "cpu" или "cuda"')
    parser.add_argument('--patch_size',
                        type=int,
                        default=defaults.get('patch_size'),
                        help='Размер минимального фрагмента для сегментации')
    parser.add_argument('--batch_size',
                        type=int,
                        default=defaults.get('batch_size'),
                        help='Количество фрагментов, которые единовременно проходят через сеть')
    parser.add_argument('--num_workers',
                        type=int,
                        default=defaults.get('num_workers'),
                        help="Число процессов-worker'ов в dataloader'e")
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()
    return {
        'model_path': args.model_path,
        'input_path': args.input_path,
        'device': args.device,
        'patch_size': args.patch_size,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'test': args.test
    }
