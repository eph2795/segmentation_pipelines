import torch


def main():
    device = torch.cuda.is_available()
    print(device)
    t = torch.rand(10).to(device)
    print(t)


if __name__ == '__main__':
    main()