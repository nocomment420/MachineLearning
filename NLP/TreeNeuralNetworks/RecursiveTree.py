import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


class RecursiveTree:
    def __init__(self):
        self.left = None
        self.right = None
        self.word = None
        self.score = None

    def forwards(self, We, Wl, Wr, b, Wo, bo, logits, targets):

        if self.left is None:

            # Leaf node
            if self.right is None:
                h = tf.nn.embedding_lookup(We, [self.word])  # 1 x D

            # Right node only
            else:
                h = tf.nn.relu(tf.matmul(Wr, tf.transpose(
                    self.right.forwards(We, Wl, Wr, b, Wo, bo, logits, targets)) + b))  # 1 x D
        else:

            # Left node only
            if self.right is None:
                h = tf.nn.relu(tf.matmul(Wl, tf.transpose(
                    self.left.forwards(We, Wl, Wr, b, Wo, bo, logits, targets)) + b))  # 1 x D
            # Both nodes
            else:
                left = tf.matmul(self.left.forwards(We, Wl, Wr, b, Wo, bo, logits, targets), Wl)
                right = tf.matmul(self.right.forwards(We, Wl, Wr, b, Wo, bo, logits, targets), Wr)
                h = tf.nn.relu(left + right + b)

        logit = tf.matmul(h, Wo) + bo
        logits.append(logit)

        targets.append(self.score)

        return h

    def validate(self):
        if self.left is not None or self.right is not None:
            if self.word is not None:
                return "Root node with word"
        if self.left is None and self.right is None:
            if self.word is None:
                return "Leaf node with no word"
        if self.score is None:
            return "Score is none"

        return None


def is_number(word):
    try:
        return True, int(word)
    except:
        return False, None


def parse_line(i, line):
    tree = RecursiveTree()
    while i <= len(line) - 1:
        char = line[i]
        if char != " ":
            # new node
            if char == "(":
                if tree.left is None:
                    tree.left, i = parse_line(i + 1, line)
                    errors = tree.left.validate()
                elif tree.right is None:
                    tree.right, i = parse_line(i + 1, line)
                    errors = tree.right.validate()
                else:
                    errors = "New node but left and right nodes are taken"
                if errors is not None:
                    raise Exception("Error: {} - \"{}\"".format(errors, line[i - 10:i]))


            # end of node
            elif char == ")":
                return tree, i

            else:
                is_num, number = is_number(char)

                # rating
                if is_num and i < len(line) - 1 and line[i + 1] == " ":
                    tree.score = number

                # word
                else:
                    if tree.word is None:
                        tree.word = char
                    else:
                        tree.word += char
        i += 1
    return tree


def build_trees(tree_strings):
    tree_roots = []
    for tree_string in tree_strings:
        try:
            root, i = parse_line(1, tree_string)
            assert i == len(tree_string) - 1
            tree_roots.append(root)
        except Exception as e:
            print("{}\n".format(str(e)))

    return tree_roots


def load_trees_from_file(filename):
    tree_strings = []

    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            if line != "\n":
                tree_string = line.rstrip()
                tree_strings.append(tree_string)

    return build_trees(tree_strings)


def convert_trees_to_word2idx(tree, word2idx, word_idx, can_add):
    if tree.left is not None:
        word2idx, word_idx = convert_trees_to_word2idx(tree.left, word2idx, word_idx, can_add)

    if tree.right is not None:
        word2idx, word_idx = convert_trees_to_word2idx(tree.right, word2idx, word_idx, can_add)

    if tree.word is not None:
        if tree.word not in word2idx:
            if can_add:
                word2idx[tree.word] = word_idx
                word_idx += 1
                tree.word = word2idx[tree.word]
            else:
                tree.word = word_idx
        else:
            tree.word = word2idx[tree.word]

    return word2idx, word_idx


def get_train_test_trees(train_filename="trees/train.txt", test_filename="trees/test.txt"):
    print("Parsing Training trees")

    start = datetime.datetime.now()
    train_roots = load_trees_from_file(train_filename)
    time_taken = (datetime.datetime.now() - start).microseconds
    print("Found {} trees in {} ms".format(len(train_roots), time_taken / 1000))

    word2idx = {}
    word_idx = 0
    for root in train_roots:
        word2idx, word_idx = convert_trees_to_word2idx(root, word2idx, word_idx, can_add=True)

    print("Tokenized train trees, found {} words\n".format(len(word2idx)))

    print("Parsing Test trees")

    start = datetime.datetime.now()
    test_roots = load_trees_from_file(test_filename)
    time_taken = (datetime.datetime.now() - start).microseconds
    print("Found {} trees in {} ms\n".format(len(test_roots), time_taken / 1000))

    for root in test_roots:
        word2idx, word_idx = convert_trees_to_word2idx(root, word2idx, word_idx, can_add=False)

    return train_roots, test_roots, word2idx

