import os
import sys

"""
Reads in tab separated files to make the dataset
Output a cPickle file of a dict with the following elements
training_instances: List of (sentence, tags) for training data
dev_instances
test_instances
w2i: Dict mapping words to indices
t2i: Dict mapping tags to indices
c2i: Dict mapping characters to indices
"""

import codecs
import argparse
import pickle
import collections
from utils import get_processing_word, read_pretrained_embeddings, is_dataset_tag, make_sure_path_exists

Instance = collections.namedtuple("Instance", ["sentence", "tags"])

# 这里面应该就是生成 词 字符 tag的index表示

UNK_TAG = "<UNK>"
NONE_TAG = "<NONE>"
START_TAG = "<START>"
END_TAG = "<STOP>"
PADDING_CHAR = "<*>"
# 处理完语料库之后，要制作这个实验需要的数据集

# 读数据文件的函数。把一个文件变成单个样本的list，这里t2i是啥？
# 小整数最大值为sys.maxsize  在这知道是个很大的整数就成
# 关于processing https://www.jianshu.com/p/045255cefe94
def read_file(filename, w2i, t2i, c2i, max_iter=sys.maxsize, processing_word=get_processing_word(lowercase=False)):
    """
    Read in a dataset and turn it into a list of instances.
    Modifies the w2i, t2is and c2i dicts, adding new words/attributes/tags/chars 
    as it sees them.
    """
    instances = []
    vocab_counter = collections.Counter()
    niter = 0
    with codecs.open(filename, "r", "utf-8") as f:  # 只读
        words, tags = [], []
        for line in f:
            line = line.strip()  # 好像是默认按空格划分每一行
            if len(line) == 0 or line.startswith("-DOCSTART-"):
                if len(words) != 0:
                    niter += 1
                    if max_iter is not None and niter > max_iter: # 这里是做了一下截断  看要多少数据
                        break
                    instances.append(Instance(words, tags))  # 把之前已经取到的word和tag 加入到instances中
                    words, tags = [], []
            else:
                word, tag = line.split()
                word = processing_word(word)   # 预处理一下word
                vocab_counter[word] += 1 # 计算这个word的数量
                if word not in w2i:  # 生成word的index表示
                    w2i[word] = len(w2i)
                if tag not in t2i:  # tag的index表示
                    t2i[tag] = len(t2i)
                if is_dataset_tag(word):  # 这个是干啥？
                    if word not in c2i:
                        c2i[word] = len(c2i)
                else:
                    for c in word:  # 好像是生成每个字符的index表示
                        if c not in c2i:
                            c2i[c] = len(c2i)
                words.append(w2i[word])
                tags.append(t2i[tag])
    return instances, vocab_counter


parser = argparse.ArgumentParser() # argparse是python的命令行解析工具，或者说可以在python代码中调用shell的一些命令，从而简化和系统命令之间的交互。
parser.add_argument("--training-data", required=True, dest="training_data", help="Training data .txt file")
parser.add_argument("--dev-data", required=True, dest="dev_data", help="Development data .txt file")
parser.add_argument("--test-data", required=True, dest="test_data", help="Test data .txt file")
parser.add_argument("-o", required=True, dest="output", help="Output filename (.pkl)")
parser.add_argument("--word-embeddings", dest="word_embeddings", help="File from which to read in pretrained embeds")
parser.add_argument("--vocab-file", dest="vocab_file", default="vocab.txt", help="Text file containing all of the words in \
                    the train/dev/test data to use in outputting embeddings")
options = parser.parse_args()

# type为参数类型，例如int。
# choices用来选择输入参数的范围，例如上面choices=【1,5,10】表示输入参数只能为1或5或10
# required用来设置在命令中显示参数，当required为True时，在输入命令时需要显示该参数
# help用来描述这个选项的作用
# action表示该选项要执行的操作
#
# dest用来指定参数的位置
# metavar用在help信息的输出中


w2i = {}  # mapping from word to index
t2i = {}  # mapping from tag to index
c2i = {} # ?
output = {}
print('Making training dataset')
output["training_instances"], output["training_vocab"] = read_file(options.training_data, w2i, t2i, c2i)
print('Making dev dataset')
output["dev_instances"], output["dev_vocab"] = read_file(options.dev_data, w2i, t2i, c2i)
print('Making test dataset')
output["test_instances"], output["test_vocab"] = read_file(options.test_data, w2i, t2i, c2i)

# Add special tokens / tags / chars to dicts
w2i[UNK_TAG] = len(w2i)
t2i[START_TAG] = len(t2i)
t2i[END_TAG] = len(t2i)
c2i[UNK_TAG] = len(c2i)

output["w2i"] = w2i
output["t2i"] = t2i
output["c2i"] = c2i

# Read embedding
if options.word_embeddings:   # 读取embedding
    output["word_embeddings"] = read_pretrained_embeddings(options.word_embeddings, w2i)

make_sure_path_exists(os.path.dirname(options.output))

print('Saving dataset to {}'.format(options.output))
with open(options.output, "wb") as outfile:
    pickle.dump(output, outfile)

#Python中的Pickle模块实现了基本的数据序列与反序列化。

# 一、dump()方法
#
# pickle.dump(obj, file, [,protocol])
#
# 注释：序列化对象，将对象obj保存到文件file中去。参数protocol是序列化模式，默认是0（ASCII协议，表示以文本的形式进行序列化），protocol的值还可以是1和2（1和2表示以二进制的形式进行序列化。其中，1是老式的二进制协议；2是新二进制协议）。file表示保存到的类文件对象，file必须有write()接口，file可以是一个以'w'打开的文件或者是一个StringIO对象，也可以是任何可以实现write()接口的对象。
#
#
#
# 二、load()方法
#
# pickle.load(file)
#
# 注释：反序列化对象，将文件中的数据解析为一个python对象。file中有read()接口和readline()接口


with codecs.open(os.path.dirname(options.output) + "/words.txt", "w", "utf-8") as vocabfile:
    for word in w2i.keys():
        vocabfile.write(word + "\n")   # 一行一个word index表示

with codecs.open(os.path.dirname(options.output) + "/chars.txt", "w", "utf-8") as vocabfile:
    for char in c2i.keys():
        vocabfile.write(char + "\n")   # 每一行是字符的inex表示

