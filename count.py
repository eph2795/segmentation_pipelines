from time import sleep


def count():
    counter = 0
    while True:
        print('Counting: {}'.format(counter))
        counter += 1
        sleep(2)


if __name__ == '__main__':
    count()
