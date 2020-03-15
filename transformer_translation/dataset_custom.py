import pickle
import random
import numpy as np

from torch.utils.data import Dataset
from  tqdm import tqdm
import os
from collections import Counter

def map_words(sentence, freq_list):
    return [freq_list[word] for word in sentence if word in freq_list]

def preprocess_data(path, vocab_folder, lang, split = 0.8):

    punct = ['(', ')', ':', '"', ' ']

    data = []
    with open(path, 'r') as fp:
        for line in fp:
            data.append(line.strip())

    proc_data = []
    for sentence in data:
        sentence = sentence.lower()
        sentence = [elem for elem in sentence.split(" ") if elem not in punct]
        proc_data.append(sentence)
        # sentence = [tok.text for tok in lang_model.tokenizer(sentence) if tok.text not in punctuation]
    # lang_data = load_data(data_path)
    # lang_model = spacy.load(lang, disable=['tagger', 'parser', 'ner'])

    indices = [i for i in range(len(data))]
    random.shuffle(indices)

    # 80:20:0 train validation test split
    train_idx = int(len(data) * split)

    train_indices = indices[:train_idx]
    test_indices = indices[train_idx:]
    # processed_sentences = [process_sentences(lang_model, sentence, punctuation) for sentence in lang_data]

    train = [proc_data[i] for i in train_indices]

    freq_list = Counter()
    for sentence in train:
        freq_list.update(sentence)
    freq_list = freq_list.most_common(10000)

    # Map words in the dictionary to indices but reserve 0 for padding,
    # 1 for out of vocabulary words, 2 for start-of-sentence and 3 for end-of-sentence
    freq_list = {freq[0]: i + 4 for i, freq in enumerate(freq_list)}
    freq_list['[PAD]'] = 0
    freq_list['[OOV]'] = 1
    freq_list['[SOS]'] = 2
    freq_list['[EOS]'] = 3
    proc_data = [map_words(sentence, freq_list) for sentence in tqdm(proc_data)]

    train = [proc_data[i] for i in train_indices]
    test = [proc_data[i] for i in test_indices]
    # test = [processed_sentences[i] for i in test_indices]

    os.makedirs(f'{vocab_folder}/processed/{lang}', exist_ok=True)
    with open(f'{vocab_folder}/processed/{lang}/train.pkl', 'wb') as f:
        pickle.dump(train, f)
    with open(f'{vocab_folder}/processed/{lang}/val.pkl', 'wb') as f:
        pickle.dump(test, f)
    # with open(f'data/processed/{lang}/test.pkl', 'wb') as f:
    #     pickle.dump(test, f)
    with open(f'{vocab_folder}/processed/{lang}/freq_list.pkl', 'wb') as f:
        pickle.dump(freq_list, f)


class TranslationDataset(Dataset):

    def __init__(self, data_path_source, data_path_target, num_tokens, max_seq_length):
        self.num_tokens = num_tokens

        with open(data_path_source, 'rb') as f:
            self.data_1 = pickle.load(f)
        with open(data_path_target, 'rb') as f:
            self.data_2 = pickle.load(f)

        self.data_lengths = {}
        for i, (str_1, str_2) in enumerate(zip(self.data_1, self.data_2)):
            if 0 < len(str_1) <= max_seq_length and 0 < len(str_2) <= max_seq_length - 2:
                if (len(str_1), len(str_2)) in self.data_lengths:
                    self.data_lengths[(len(str_1), len(str_2))].append(i)
                else:
                    self.data_lengths[(len(str_1), len(str_2))] = [i]

        dl = self.data_lengths.copy()
        for k, v in dl.items():
            random.shuffle(v)

        batches = []
        prev_tokens_in_batch = 1e10
        for k in sorted(dl):
            v = dl[k]
            total_tokens = (k[0] + k[1]) * len(v)

            while total_tokens > 0:
                tokens_in_batch = min(total_tokens, num_tokens) - min(total_tokens, num_tokens) % (k[0] + k[1])
                sentences_in_batch = tokens_in_batch // (k[0] + k[1])

                # Combine with previous batch?
                if tokens_in_batch + prev_tokens_in_batch <= num_tokens:
                    batches[-1].extend(v[:sentences_in_batch])
                    prev_tokens_in_batch += tokens_in_batch
                else:
                    batches.append(v[:sentences_in_batch])
                    prev_tokens_in_batch = tokens_in_batch
                v = v[sentences_in_batch:]

                total_tokens = (k[0] + k[1]) * len(v)
        self.batches = batches
        # self.batches = gen_batches(num_tokens, self.data_lengths)

    def __proc_data_mask(self, idx, src = True):
        sentence_indices = self.batches[idx]
        if src:
            batch = [self.data_1[i] for i in sentence_indices]
        else:
            batch = [[2] + self.data_2[i] + [3] for i in sentence_indices]

        seq_length = 0
        for sentence in batch:
            if len(sentence) > seq_length:
                seq_length = len(sentence)

        masks = []
        for i, sentence in enumerate(batch):
            masks.append([False for _ in range(len(sentence))] + [True for _ in range(seq_length - len(sentence))])
            batch[i] = sentence + [0 for _ in range(seq_length - len(sentence))]

        return np.array(batch), np.array(masks)
    def __getitem__(self, item):
        src, src_mask = self.__proc_data_mask(item, True)
        tgt, tgt_mask = self.__proc_data_mask(item, False)

        return src, src_mask, tgt, tgt_mask

    def __len__(self):
        return len(self.batches)

    def shuffle_batches(self):

        dl = self.data_lengths.copy()
        for k, v in dl.items():
            random.shuffle(v)

        batches = []
        prev_tokens_in_batch = 1e10
        for k in sorted(dl):
            v = dl[k]
            total_tokens = (k[0] + k[1]) * len(v)

            while total_tokens > 0:
                tokens_in_batch = min(total_tokens, self.num_tokens) - min(total_tokens, self.num_tokens) % (
                            k[0] + k[1])
                sentences_in_batch = tokens_in_batch // (k[0] + k[1])

                # Combine with previous batch?
                if tokens_in_batch + prev_tokens_in_batch <= self.num_tokens:
                    batches[-1].extend(v[:sentences_in_batch])
                    prev_tokens_in_batch += tokens_in_batch
                else:
                    batches.append(v[:sentences_in_batch])
                    prev_tokens_in_batch = tokens_in_batch
                v = v[sentences_in_batch:]

                total_tokens = (k[0] + k[1]) * len(v)
        self.batches = batches

if __name__ == "__main__":
    preprocess_data("data_enru/raw/corpus.en_ru.1m.en", "data_enru", "en")
    # preprocess_data("data_enru/raw/corpus.en_ru.1m.ru", "data_enru", "ru")