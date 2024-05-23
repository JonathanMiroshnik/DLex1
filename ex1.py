import torch
import numpy


def main():
    with open("ex1 data", 'rb') as dataFile:
        for line in dataFile:
            try:
                # 'latin1'
                print(torch.Tensor(numpy.frombuffer(line, dtype=numpy.int32)))
                print(line.decode('utf-8'))
            except UnicodeDecodeError:
                print(line.decode('utf-8', errors='ignore'))



if __name__ == '__main__':
    main()
