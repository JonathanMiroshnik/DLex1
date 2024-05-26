import zipfile
import random
import torch
import heapq
import numpy
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def string_to_one_hot(s, char_to_index):
    """
    :param s: string to make one-hot embedding of
    :param char_to_index: dictionary with char key to one-hot index position
    :return: One hot embedding of string of one-hot character representations
    """
    if len(char_to_index.keys()) <= 0:
        return
    one_hot = torch.zeros(len(s), len(char_to_index.keys()))
    for i, char in enumerate(s):
        one_hot[i, char_to_index[char]] = 1.0
    return one_hot


def split_list(input_list, split_ratio=0.9):
    """
    Shuffle and split list into two lists of percentage size according to split_ratio
    :param input_list: input list to shuffle and split
    :param split_ratio: Percentage size of first list
    :return: two lists of percentage size according to split_ratio
    """
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
    """
    Splits string to all consecutive sub-strings of length sub_seq_len
    :param whole_sequence: string to split
    :param sub_seq_len: length of sub-strings
    :return: list of sub strings of sub_seq_len
    """
    if sub_seq_len > len(whole_sequence):
        return
    ret_list = list()
    for i in range(len(whole_sequence)-sub_seq_len-1):
        ret_list.append(whole_sequence[i:(i+sub_seq_len)])
    return ret_list


def make_train_test_graph(train_loss, test_loss):
    """
    Graphs the train and test loss over the epochs
    :param train_loss: train loss list
    :param test_loss: test loss list
    """
    if len(train_loss) != len(test_loss):
        return

    # Create a list of indices
    indices = list(range(len(train_loss)))

    # Plot the data
    plt.figure(figsize=(10, 5))
    plt.plot(indices, train_loss, label='Train loss')
    plt.plot(indices, test_loss, label='Test loss')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and test loss by epoch')
    plt.legend()

    # Show the plot
    plt.show()


class StringClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StringClassifier, self).__init__()
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


def main():
    neg_list, pos_list = list(), list()
    with zipfile.ZipFile("ex1 data", 'r') as zip_ref:
        with zip_ref.open("ex1 data/neg_A0201.txt", 'r') as neg:
            for line in neg:
                neg_list.append(line.decode())
            neg.close()
        with zip_ref.open("ex1 data/pos_A0201.txt", 'r') as pos:
            for line in pos:
                pos_list.append(line.decode())
            pos.close()
        zip_ref.close()

    # Getting all the letter representations of the amino acids
    all_data = neg_list + pos_list
    all_letters = set()
    for d_point in all_data:
        for label in d_point:
            all_letters.add(label)
    char_to_index = {char: idx for idx, char in enumerate(all_letters)}

    # Creating and shuffling train/test data
    pre_shuffled_data = neg_list + pos_list
    pre_shuffled_labels = [0]*len(neg_list) + [1]*len(pos_list)
    paired_list = list(zip(pre_shuffled_data, pre_shuffled_labels))

    split_train, split_test = split_list(paired_list)
    split_train_data, split_train_labels = zip(*split_train)
    split_test_data, split_test_labels = zip(*split_test)

    split_train_data, split_train_labels = list(split_train_data), list(split_train_labels)
    split_test_data, split_test_labels = list(split_test_data), list(split_test_labels)

    # Neural Network creation and training
    train_inputs = torch.stack([string_to_one_hot(s.replace('\n', ''), char_to_index) for s in split_train_data])
    train_targets = torch.tensor(split_train_labels, dtype=torch.float32)

    test_inputs = torch.stack([string_to_one_hot(s.replace('\n', ''), char_to_index) for s in split_test_data])
    test_targets = torch.tensor(split_test_labels, dtype=torch.float32)

    input_size = 9 * len(char_to_index.keys())  # 9 characters each with a one-hot encoding of length num_chars
    hidden_size = 180  # Number of hidden units
    output_size = 1  # Binary output

    model = StringClassifier(input_size, hidden_size, output_size)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("-------------------------------------------------------- Training outputs:")

    train_loss, test_loss = list(), list()

    # Training loop
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_inputs)
        loss = criterion(outputs.squeeze(), train_targets)

        train_loss.append(float(loss))

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        outputs = model(test_inputs)
        loss = criterion(outputs.squeeze(), test_targets)

        test_loss.append(float(loss))

    print("Training complete.")

    # Trained Neural Network on test data
    outputs = model(test_inputs)
    loss = criterion(outputs.squeeze(), test_targets)

    print("-------------------------------------------------------- Test outputs:\n" + f'Test Loss: {loss.item():.4f}')

    # Getting results for the given spike protein
    with open("spike.txt", 'r') as spike_txt:
        whole_sent = spike_txt.read()
        spike_txt.close()
    whole_sent = whole_sent.replace('\n', '')

    spike_data = split_string_to_consecutive_sequences(whole_sent, 9)
    spike_inputs = torch.stack([string_to_one_hot(s, char_to_index) for s in spike_data])

    outputs = model(spike_inputs)

    print("-------------------------------------------------------- Spike outputs:")
    # Spike dictionary takes the peptide as the key and gives a tuple of type (number_of_occurrences, average)
    spike_dict = dict()
    for i, s in enumerate(spike_data):
        if s not in spike_dict.keys():
            spike_dict[s] = (1, outputs[i])
        else:
            cur_num_occur, cur_avg = spike_dict[s]
            spike_dict[s] = (cur_num_occur+1, (cur_avg * cur_num_occur + outputs[i]) / (cur_num_occur+1))

    # Finding the 3 most correlated peptides from the spike protein(the 3 highest averages from the dictionary)
    best_peptides = heapq.nlargest(3, spike_dict.items(), key=lambda item: item[1][1])
    best_peptides = [item[0] for item in best_peptides]

    print("The best peptides for prediction are:\n" + str(best_peptides))

    make_train_test_graph(train_loss, test_loss)


if __name__ == '__main__':
    main()
