import zipfile
import torch
import numpy


def main():
    with zipfile.ZipFile("ex1 data", 'r') as zip_ref:
        print("NEGATIVE VALUES")
        with zip_ref.open("ex1 data/neg_A0201.txt", 'r') as neg:
            for line in neg:
                print(line.decode())
        print("POSITIVE VALUES")
        with zip_ref.open("ex1 data/pos_A0201.txt", 'r') as pos:
            for line in pos:
                print(line.decode())

if __name__ == '__main__':
    main()
