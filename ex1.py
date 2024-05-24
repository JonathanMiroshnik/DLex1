import zipfile
import random
import torch
import numpy


def split_list(input_list, split_ratio=0.9):
    # Shuffle the list to ensure randomness
    shuffled_list = input_list[:]
    random.shuffle(shuffled_list)

    # Determine the split index
    split_index = int(len(shuffled_list) * split_ratio)

    # Split the list
    list_90 = shuffled_list[:split_index]
    list_10 = shuffled_list[split_index:]

    return list_90, list_10


# Part e
def split_string_to_consecutive_sequences(whole_sequence: str, sub_seq_len: int):
    if sub_seq_len > len(whole_sequence):
        return
    ret_list = list()
    for i in range(len(whole_sequence)-sub_seq_len-1):
        ret_list.append(whole_sequence[i:(i+sub_seq_len)])
    return ret_list


def main():
    neg_list, pos_list = list(), list()
    with zipfile.ZipFile("ex1 data", 'r') as zip_ref:
        with zip_ref.open("ex1 data/neg_A0201.txt", 'r') as neg:
            for line in neg:
                neg_list.append(line.decode())
        with zip_ref.open("ex1 data/pos_A0201.txt", 'r') as pos:
            for line in pos:
                pos_list.append(line.decode())

    data_to_split = neg_list + pos_list

    # Getting all the letter representations of the amino acids
    all_letters = set()
    for dpoint in data_to_split:
        for l in dpoint:
            all_letters.add(l)
    # print(len(all_letters), all_letters)



    # train_data, test_data = split_list(data_to_split)
    # print(train_data, test_data)

    whole_sent = ""
    with open("spike.txt", 'r') as spike_txt:
        whole_sent = spike_txt.read()
        # for line in spike_txt:
        #     whole_sent += line

    print(len(whole_sent))
    print(whole_sent[-1])
    print(len(split_string_to_consecutive_sequences(whole_sent, 9)))



if __name__ == '__main__':
    main()
