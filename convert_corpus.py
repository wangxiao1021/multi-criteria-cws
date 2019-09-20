# -*- coding:utf-8 -*-
# Filename: convert_corpus.py
# Author：hankcs
# Date: 2017-08-08 AM10:45

"""
Convert and preprocess original space separated corpus to bmes tagged corpus
"""
import os
import re

from utils import make_sure_path_exists, append_tags

def normalize(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif 65281 <= inside_code <= 65374:  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def preprocess(text):  # 文本预处理
    rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
    rENG = '[A-Za-z_.]+'
    sent = normalize(text.strip()).split()
    new_sent = []
    for word in sent:
        word = re.sub('\s+', '', word, flags=re.U)   # flags?
        word = re.sub(rNUM, '0', word, flags=re.U)   #数字统一用0替换
        word = re.sub(rENG, 'X', word)  # 字母用X替换
        new_sent.append(word)
    return new_sent


def to_sentence_list(text, split_long_sentence=False):  # 句子切分
    text = preprocess(text) # 先做预处理
    delimiter = set()
    delimiter.update('。！？：；…、，（）”’,;!?、,')
    delimiter.add('……')
    sent_list = []
    sent = []
    for word in text:
        sent.append(word)
        if word in delimiter or (split_long_sentence and len(sent) >= 50):
            sent_list.append(sent)
            sent = []

    if len(sent) > 0:
        sent_list.append(sent)

    return sent_list


def convert_file(src, des, split_long_sentence=False, encode='UTF-8'):  # 转换成text格式的
    with open(src, encoding=encode) as src, open(des, 'w', encoding=encode) as des:
        for line in src:
            for sent in to_sentence_list(line, split_long_sentence):
                des.write(' '.join(sent) + '\n')
                # if len(''.join(sent)) > 200:
                #     print(' '.join(sent))


def split_train_dev(dataset):  # 划分训练集和验证集
    root = 'data/' + dataset + '/raw/'
    with open(root + 'train-all.txt', encoding='UTF-8') as src, open(root + 'train.txt', 'w', encoding='UTF-8') as train, open(root + 'dev.txt',
                                                                                           'w', encoding='UTF-8') as dev:
        lines = src.readlines()
        idx = int(len(lines) * 0.9)
        for line in lines[: idx]:
            train.write(line)
        for line in lines[idx:]:
            dev.write(line)


def combine_files(one, two, out):  #合并文件
    if os.path.exists(out):  # 如果已经有out 删除
        os.remove(out)
    with open(one, encoding='utf-8') as one, open(two, encoding='utf-8') as two, open(out, 'a', encoding='utf-8') as out:
        for line in one:
            out.write(line)
        for line in two:
            out.write(line)


def bmes_tag(input_file, output_file):   # 字符级别标注
    with open(input_file, encoding='utf-8') as input_data, open(output_file, 'w', encoding='utf-8') as output_data:
        for line in input_data:
            word_list = line.strip().split()  #Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。
            for word in word_list:
                if len(word) == 1 or (len(word) > 2 and word[0] == '<' and word[-1] == '>'):
                    output_data.write(word + "\tS\n")  # 标注为句子开头
                else:
                    output_data.write(word[0] + "\tB\n")  # 标注为句子开始词
                    for w in word[1:len(word) - 1]:  # 不包括最后一个
                        output_data.write(w + "\tM\n")  # 句子中间词
                    output_data.write(word[len(word) - 1] + "\tE\n")
            output_data.write("\n")


def make_bmes(dataset='pku'):   # + tag
    path = 'data/' + dataset + '/'
    make_sure_path_exists(path + 'bmes')
    bmes_tag(path + 'raw/train.txt', path + 'bmes/train.txt')
    bmes_tag(path + 'raw/train-all.txt', path + 'bmes/train-all.txt')
    bmes_tag(path + 'raw/dev.txt', path + 'bmes/dev.txt')
    bmes_tag(path + 'raw/test.txt', path + 'bmes/test.txt')


def convert_sighan2005_dataset(dataset):  # conver2005
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/sighan2005/{}_training.utf8'.format(dataset), 'data/{}/raw/train-all.txt'.format(dataset), True)
    convert_file('data/sighan2005/{}_test_gold.utf8'.format(dataset), 'data/{}/raw/test.txt'.format(dataset), False)
    split_train_dev(dataset)


def convert_sighan2008_dataset(dataset, utf=16): # convert 2008
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/sighan2008/{}_train_seg/{}_train_utf{}.seg'.format(dataset, dataset, utf),
                 'data/{}/raw/train-all.txt'.format(dataset), True, 'utf-{}'.format(utf))
    convert_file('data/sighan2008/{}_seg_truth&resource/{}_truth_utf{}.seg'.format(dataset, dataset, utf),
                 'data/{}/raw/test.txt'.format(dataset), False, 'utf-{}'.format(utf))
    split_train_dev(dataset)


def convert_sxu():
    dataset = 'sxu'
    print(('Converting corpus {}'.format(dataset)))
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/other/{}/train.txt'.format(dataset), 'data/{}/raw/train-all.txt'.format(dataset), True)
    convert_file('data/other/{}/test.txt'.format(dataset), 'data/{}/raw/test.txt'.format(dataset), False)
    split_train_dev(dataset)
    make_bmes(dataset)


def convert_ctb():
    dataset = 'ctb'
    print(('Converting corpus {}'.format(dataset)))
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    convert_file('data/other/ctb/ctb6.train.seg', 'data/{}/raw/train.txt'.format(dataset), True)
    convert_file('data/other/ctb/ctb6.dev.seg', 'data/{}/raw/dev.txt'.format(dataset), True)
    convert_file('data/other/ctb/ctb6.test.seg', 'data/{}/raw/test.txt'.format(dataset), False)
    combine_files('data/{}/raw/train.txt'.format(dataset), 'data/{}/raw/dev.txt'.format(dataset),
                  'data/{}/raw/train-all.txt'.format(dataset))
    make_bmes(dataset)


def remove_pos(src, out, delimiter='/'):
    # print(src)
    with open(src, encoding='utf-8') as src, open(out, 'w', encoding='utf-8') as out:
        for line in src:
            words = []
            for word_pos in line.split(' '):
                # if len(word_pos.split(delimiter)) != 2:
                #     print(line)
                word, pos = word_pos.split(delimiter)
                words.append(word)
            out.write(' '.join(words) + '\n')


def convert_zhuxian():
    dataset = 'zx'
    print(('Converting corpus {}'.format(dataset)))
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    remove_pos('data/other/zx/dev.zhuxian.wordpos', 'data/zx/dev.txt', '_')
    remove_pos('data/other/zx/train.zhuxian.wordpos', 'data/zx/train.txt', '_')
    remove_pos('data/other/zx/test.zhuxian.wordpos', 'data/zx/test.txt', '_')

    convert_file('data/zx/train.txt', 'data/{}/raw/train.txt'.format(dataset), True)
    convert_file('data/zx/dev.txt', 'data/{}/raw/dev.txt'.format(dataset), True)
    convert_file('data/zx/test.txt', 'data/{}/raw/test.txt'.format(dataset), False)
    combine_files('data/{}/raw/train.txt'.format(dataset), 'data/{}/raw/dev.txt'.format(dataset),
                  'data/{}/raw/train-all.txt'.format(dataset))
    make_bmes(dataset)


def convert_cncorpus():
    dataset = 'cnc'
    print(('Converting corpus {}'.format(dataset)))
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')
    remove_pos('data/other/cnc/train.txt', 'data/cnc/train-no-pos.txt')
    remove_pos('data/other/cnc/dev.txt', 'data/cnc/dev-no-pos.txt')
    remove_pos('data/other/cnc/test.txt', 'data/cnc/test-no-pos.txt')

    convert_file('data/cnc/train-no-pos.txt', 'data/{}/raw/train.txt'.format(dataset), True)
    convert_file('data/cnc/dev-no-pos.txt', 'data/{}/raw/dev.txt'.format(dataset), True)
    convert_file('data/cnc/test-no-pos.txt', 'data/{}/raw/test.txt'.format(dataset), False)
    combine_files('data/{}/raw/train.txt'.format(dataset), 'data/{}/raw/dev.txt'.format(dataset),
                  'data/{}/raw/train-all.txt'.format(dataset))
    make_bmes(dataset)


def extract_conll(src, out):
    words = []
    with open(src, encoding='utf-8') as src, open(out, 'w', encoding='utf-8') as out:
        for line in src:
            line = line.strip()
            if len(line) == 0:
                out.write(' '.join(words) + '\n')
                words = []
                continue
            cells = line.split()
            words.append(cells[1])


def convert_conll(dataset):
    print(('Converting corpus {}'.format(dataset)))
    root = 'data/' + dataset
    make_sure_path_exists(root)
    make_sure_path_exists(root + '/raw')

    extract_conll('data/other/{}/dev.conll'.format(dataset), 'data/{}/dev.txt'.format(dataset))
    extract_conll('data/other/{}/test.conll'.format(dataset), 'data/{}/test.txt'.format(dataset))
    extract_conll('data/other/{}/train.conll'.format(dataset), 'data/{}/train.txt'.format(dataset))

    convert_file('data/{}/train.txt'.format(dataset), 'data/{}/raw/train.txt'.format(dataset), True)
    convert_file('data/{}/dev.txt'.format(dataset), 'data/{}/raw/dev.txt'.format(dataset), True)
    convert_file('data/{}/test.txt'.format(dataset), 'data/{}/raw/test.txt'.format(dataset), False)
    combine_files('data/{}/raw/train.txt'.format(dataset), 'data/{}/raw/dev.txt'.format(dataset),
                  'data/{}/raw/train-all.txt'.format(dataset))
    make_bmes(dataset)


def make_joint_corpus(datasets, joint):
    parts = ['dev', 'test', 'train', 'train-all']
    for part in parts:
        old_file = 'data/{}/raw/{}.txt'.format(joint, part)
        if os.path.exists(old_file):
            os.remove(old_file)
        elif not os.path.exists(os.path.dirname(old_file)):
            os.makedirs(os.path.dirname(old_file))
        for name in datasets:
            append_tags(name, joint, part) #？


def convert_all_sighan2005(datasets):
    for dataset in datasets:
        print(('Converting sighan bakeoff 2005 corpus: {}'.format(dataset)))
        convert_sighan2005_dataset(dataset)
        make_bmes(dataset)


def convert_all_sighan2008(datasets):
    for dataset in datasets:
        print(('Converting sighan bakeoff 2008 corpus: {}'.format(dataset)))
        convert_sighan2008_dataset(dataset, 8 if dataset == 'ckip' or dataset == 'cityu' else 16)
        make_bmes(dataset)

#主要是一些格式的转换，字符的预处理，tagging什么的
if __name__ == '__main__':
    print('Converting sighan2005 Simplified Chinese corpus')   # 都转换成简体的
    datasets = 'pku', 'msr', 'as', 'cityu'
    convert_all_sighan2005(datasets)

    print('Combining sighan2005 corpus to one joint Simplified Chinese corpus')
    datasets = 'pku', 'msr', 'as', 'cityu'
    make_joint_corpus(datasets, 'joint-sighan2005')    # 生成联合的训练数据
    make_bmes('joint-sighan2005')   

    # For researchers who doesn't have access to sighan2008 corpus, use following freely available corpora please.
    print('Converting extra 6 corpora')
    convert_sxu()
    convert_ctb()
    convert_zhuxian()
    convert_cncorpus()
    convert_conll('udc')
    convert_conll('wtb')
    # make a large joint corpus
    print('Combining those 10 corpora to one joint corpus')
    datasets = 'pku', 'msr', 'as', 'cityu', 'sxu', 'ctb', 'zx', 'cnc', 'udc', 'wtb'
    make_joint_corpus(datasets, 'joint-10in1')
    make_bmes('joint-10in1')

    # For researchers who have access to sighan2008 corpus, use official corpora please.
    # print('Converting sighan2008 Simplified Chinese corpus')
    # datasets = 'ctb', 'ckip', 'cityu', 'ncc', 'sxu'
    # convert_all_sighan2008(datasets)
    # print('Combining those 8 sighan corpora to one joint corpus')
    # datasets = 'pku', 'msr', 'as', 'ctb', 'ckip', 'cityu', 'ncc', 'sxu'
    # make_joint_corpus(datasets, 'joint-sighan2008')
    # make_bmes('joint-sighan2008')
