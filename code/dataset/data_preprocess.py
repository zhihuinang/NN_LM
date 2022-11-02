import gzip
import time
import jieba


from collections import Counter, OrderedDict
from torchtext.vocab import vocab

def data_preprocess():
    f = gzip.open('./dataset/news.2018.zh.shuffled.deduped.gz','rb')
    sentences = []
    for line in f.readlines():
        s = line.decode()
        sentences.append(s)
    f.close()

    word_list = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        word_list.extend(jieba.lcut(sentence.strip('\n').strip()))

    data_len = len(word_list)
    ordered_dict = OrderedDict(sorted(Counter(word_list).items(), key=lambda x: x[1], reverse=True))
    v = vocab(ordered_dict,specials=['<unk>'],min_freq=5)
    v.set_default_index(v['<unk>'])
    train_sentences = word_list[:int(0.8*data_len)]
    eval_sentences = word_list[int(0.8*data_len):]
    
    return train_sentences, eval_sentences, v
    
