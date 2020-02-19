import torch

from stack_segmentation.stack import Stack
from stack_segmentation.training import make_model


def inference(model_path, input_path, model_config):
    device = 'cuda:0'
    model, criterion, optimizer, scheduler = make_model(**model_config)

    model.load_state_dict(torch.load(model_path))
    stack = Stack.read_from_source(input_path)
    predicted_stack = stack.apply(model,
                                  model_config,
                                  patch_sizes=(128, 128, 1),
                                  bs=32, num_workers=8, device='cuda:0',
                                  threshold=None)
    predicted_stack.dump(input_path,
                         features=False,
                         targets=False,
                         preds=True)


if __name__ == '__main__':
    inference(model_path, input_path)