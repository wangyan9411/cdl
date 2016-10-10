# pylint:skip-file
import sys, random, time, math
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *
from operator import itemgetter
from optparse import OptionParser
from numpy import linalg as la


np.set_printoptions(threshold=np.nan)

def get_net(vocab_size, num_input, num_label):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_V = mx.sym.Variable('label_V')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    # input_dim don't mean the input data.shape[1]
    data_embed = mx.sym.Embedding(data = data, input_dim = vocab_size,
                                  weight = embed_weight,
                                  output_dim = 50, name = 'data_embed')
    datavec = mx.sym.SliceChannel(data = data_embed,
                                     num_outputs = num_input,
                                     squeeze_axis = 1, name = 'data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]
    # datavec[0] is the learned embedding of the location
    f1_loss = mx.symbol.LinearRegressionOutput(data=datavec[0],
                                               label=label_V)
    f2_loss = nce_loss(data = pred,
                    label = label,
                    label_weight = label_weight,
                    embed_weight = embed_weight,
                    vocab_size = vocab_size,
                    num_hidden = 50,
                    num_label = num_label)
    return mx.symbol.Group([f1_loss, f2_loss])


# return data: dict(zip(itemID, itemdata)) Or a list indexed by itemID, loaction: dict(zip(itemID, locationID)),
# negative: sampling according to frequency.
# fre[wordID] = freq
# name1: raw text concated, name2: item text per line
def load_data(name):
    # sum
    fin = open(name, 'r')
    tks = []
    for buf in fin.readlines():
        tks += buf.split(' ')
    vocab = {}
    # word index start from 1
    freq = [0]
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab) + 1
            freq.append(0)
        wid = vocab[tk]
        #data.append(wid)
        freq[wid] += 1
    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]

    print 'vocabularies size'
    print len(vocab)
    # construct data
    data = []
    fin = open(name, 'r')
    item_size = 0
    for id, line in enumerate(fin.readlines()):
        tks = line.split(' ')
        data.append([vocab[tk] for tk in tks])
        item_size += 1
    #assign faked word id to locations
    location = range(len(vocab)+1, len(vocab)+1+item_size)

    return data, location, negative, vocab, freq

class DataMatrix():
    # file(name) store the corpus, each line stands for a item.
    def __init__(self, name, batch_size, num_label, sample_num = 4):
        # super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.sample_num = sample_num
        # self.data: a dict storing the key itemID to value the context sequence.
        # self.negative stores the sampling pool for the NCE
        self.data, self.location, self.negative, self.vocab, self.freq = load_data(name)
        # Some mistake here? !! the word 0 and word_last is never updated
        self.vocab_size = 1 + len(self.vocab) + len(self.data) # plus the location size
        print 'vocab size'
        print self.vocab_size
        self.num_label = num_label
        self.item_size = len(self.data)
        # Dimension num_label-1 context words and one location,
        self.provide_data = [('data', (batch_size, num_label))]
        # predict the label
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]

    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def get_matrix(self, V_theta1):
        # location[i]: the location id of the item i; imitate a word and is added to the input.
        print 'begin'
        batch_data = []
        batch_label = []
        batch_label_weight = []
        for i in range(0, len(self.data)):
            # for each item, sample subset of 4 to train out the location embedding.
            for j in range(0, self.sample_num):
                start = random.randint(0, len(self.data[i]) - self.num_label - 1)
                context = [self.location[i]] + self.data[i][start: start + self.num_label / 2] \
                    + self.data[i][start + 1 + self.num_label / 2: start + self.num_label]
                target_word = self.data[i][start + self.num_label / 2]
                if self.freq[target_word] < 5:
                    continue
                target = [target_word] \
                    + [self.sample_ne() for _ in range(self.num_label - 1)]
                target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
                # print (context)
                batch_data.append(context)
                batch_label.append(target)
                batch_label_weight.append(target_weight)

        # construct label_V-theta1:
        assert len(self.data) == len(V_theta1)
        batch_label_V = [l for l in V_theta1 for _ in range(0, self.sample_num)]
        assert len(batch_label_V) == len(batch_data)

        matrix = np.array(batch_data)
        matrix_label = np.array(batch_label)
        matrix_label_weight = np.array(batch_label_weight)
        matrix_label_V = np.array(batch_label_V)
        return len(self.data), mx.nd.array(matrix), mx.nd.array(matrix_label), \
                mx.nd.array(matrix_label_weight), mx.nd.array(matrix_label_V)


class LocationEmbedding(object):
    def __init__(self, optimizer, **kwargs):
        if isinstance(optimizer, str):
            self.optimizer = mx.optimizer.create(optimizer, **kwargs)
        else:
            self.optimizer = optimizer
        # updater is SGD
        self.updater = mx.optimizer.get_updater(self.optimizer)
        self.monitor = None
        self.metric = None
        self.iter_end_callback = None
        self.iter_start_callback = None

    def set_metric(self, metric):
        self.metric = metric

    def set_monitor(self, monitor):
        self.monitor = monitor

    def set_iter_end_callback(self, callback):
        self.iter_end_callback = callback

    def set_iter_start_callback(self, callback):
        self.iter_start_callback = callback

    def perform_one_epoch(self, data_iter):
        print 'perform one epoch'
        for batch in data_iter:
            # input_buffs to load the batch_data. input_buff is bind to the executor.
            for data, buff in zip(batch.data+batch.label, self.input_buffs):
                data.copyto(buff)
            self.exe.forward(is_train=True)
            self.exe.backward()
            for key, arr in self.update_dict.items():
                # print str(arr.asnumpy())
                self.updater(key, arr, self.args[key])

            # exe.outputs is a list
            self.exe.outputs[0].wait_to_read()


    def construct(self, xpu, sym,
            data_iter, vocab_size, data_size, data_matrix, auxs = None, begin_iter = 0, end_iter = 2000, args_lrmult={}, debug = False):
        self.xpu = xpu
        self.vocab_size = vocab_size
        self.data_size = data_size
        self.datamatrix = data_matrix
        # 50 is consistent with K in Joint learning
        self.args = {'embed_weight': mx.nd.empty((vocab_size, 50), self.xpu),}
        self.args_grad = {'embed_weight': mx.nd.empty((vocab_size, 50), self.xpu),}
        # initialize the params
        init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        for k,v in self.args.items():
            init(k,v)

#print 'original'
#print str(self.args['embed_weight'].asnumpy())


        input_desc = data_iter.provide_data + data_iter.provide_label
        input_names = [k for k, shape in input_desc]
        self.input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in input_desc]
        # add to the arguments of the sym. merge two dictionary.
        self.args = dict(self.args, **dict(zip(input_names, self.input_buffs)))

        output_names = sym.list_outputs()
        self.exe = sym.bind(xpu, args=self.args, args_grad=self.args_grad, aux_states=auxs)

        assert len(sym.list_arguments()) == len(self.exe.grad_arrays)

        # update arguments except the data
        self.update_dict = {name: nd for name, nd in zip(sym.list_arguments(), self.exe.grad_arrays) if nd}
        batch_size = self.input_buffs[0].shape[0]
        # set the scale.
        self.optimizer.rescale_grad = 1.0/batch_size
        self.optimizer.set_lr_mult(args_lrmult)

# output diff used for the metric, not used so far
        output_dict = {}
        output_buff = {}
        internal_dict = dict(zip(input_names, self.input_buffs))
        for key, arr in zip(sym.list_outputs(), self.exe.outputs):
            if key in output_names:
                output_dict[key] = arr
                output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
            else:
                internal_dict[key] = arr

        data_iter.reset()

    def get_params(self):
        return self.args['embed_weight'].asnumpy()[self.vocab_size - self.data_size:, :]


def getLocationEmbedding(batch_size, item_size):
    num_label = 6
    datamatrix = DataMatrix("./data/text8", batch_size, num_label)
    # (TODO: wangyan) the initialization is very important
    V_theta1 = np.random.random([item_size,50])
    print str(V_theta1)
    data_size, data, label, label_weight, label_V = datamatrix.get_matrix(V_theta1)
    data_iter = mx.io.NDArrayIter({'data': data, 'label': label, 'label_weight': label_weight, 'label_V': label_V},
                                  batch_size=batch_size, shuffle=False, # shuffle should be True?
                                  last_batch_handle='pad')

    network = get_net(datamatrix.vocab_size, num_label, num_label)
    metric = NceAuc()
    solver = LocationEmbedding('sgd', momentum=0.9, wd=0.0000, learning_rate=0.3, lr_scheduler
                    = None #mx.misc.FactorScheduler(20000,0.1)
                    )
    solver.set_metric(metric)
# solver.set_monitor(Monitor(1000))

#   logging.info('Fine tuning...')
    solver.construct(xpu = mx.cpu(), sym = network, data_iter = data_iter, vocab_size = datamatrix.vocab_size, data_size = data_size, data_matrix = datamatrix)
    return solver, data_iter

if __name__ == '__main__':
    LE, data_iter = getLocationEmbedding()
    #for i in range(begin_iter, end_iter):
    epoch = 20
    for i in range(0, epoch):
        LE.perform_one_epoch(data_iter)
        data_iter.reset()

#print str(args['embed_weight'].asnumpy())


# a = model.get_params()
    args = LE.get_params()
    print len(args)

    A = args[0,:]
    for i in range(0, len(args)):
        inA = np.mat(A)
        inB = np.mat(args[i, :])
        print i
        print 0.5+0.5*(float(inA*inB.T)/(la.norm(inA)*la.norm(inB)))
    open('save2', 'w').write(str(args))

