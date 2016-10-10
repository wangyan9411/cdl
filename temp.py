# pylint:skip-file
import sys, random, time, math
sys.path.insert(0, "../../python")
import mxnet as mx
import numpy as np
from collections import namedtuple
from nce import *
from operator import itemgetter
from optparse import OptionParser
np.set_printoptions(threshold=np.nan)

def get_net(vocab_size, num_input, num_label):
    data = mx.sym.Variable('data')
    label = mx.sym.Variable('label')
    label_weight = mx.sym.Variable('label_weight')
    embed_weight = mx.sym.Variable('embed_weight')
    data_embed = mx.sym.Embedding(data = data, input_dim = vocab_size,
                                  weight = embed_weight,
                                  output_dim = 50, name = 'data_embed')
    datavec = mx.sym.SliceChannel(data = data_embed,
                                     num_outputs = num_input,
                                     squeeze_axis = 1, name = 'data_slice')
    pred = datavec[0]
    for i in range(1, num_input):
        pred = pred + datavec[i]
    return nce_loss(data = pred,
                    label = label,
                    label_weight = label_weight,
                    embed_weight = embed_weight,
                    vocab_size = vocab_size,
                    num_hidden = 50,
                    num_label = num_label)    

def load_data(name):
    buf = open(name).read()
    tks = buf.split(' ')
    vocab = {}
    freq = [0]
    data = []
    for tk in tks:
        if len(tk) == 0:
            continue
        if tk not in vocab:
            vocab[tk] = len(vocab) + 1
            freq.append(0)
        wid = vocab[tk]
        data.append(wid)
        freq[wid] += 1
    negative = []
    for i, v in enumerate(freq):
        if i == 0 or v < 5:
            continue
        v = int(math.pow(v * 1.0, 0.75))
        negative += [i for _ in range(v)]
    return data, negative, vocab, freq

class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, name, batch_size, num_label):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.data, self.negative, self.vocab, self.freq = load_data(name)
        self.vocab_size = 1 + len(self.vocab)
        print 'vocab_size'
        print self.vocab_size
        self.num_label = num_label
        self.provide_data = [('data', (batch_size, num_label - 1))]
        self.provide_label = [('label', (self.batch_size, num_label)),
                              ('label_weight', (self.batch_size, num_label))]
        
    def sample_ne(self):
        return self.negative[random.randint(0, len(self.negative) - 1)]

    def __iter__(self):
        print 'begin'
        batch_data = []
        batch_label = []
        batch_label_weight = []
        start = random.randint(0, self.num_label - 1)
        for i in range(start, len(self.data) - self.num_label - start, self.num_label):
            context = self.data[i: i + self.num_label / 2] \
                      + self.data[i + 1 + self.num_label / 2: i + self.num_label]
            target_word = self.data[i + self.num_label / 2]
            if self.freq[target_word] < 5:
                continue
            target = [target_word] \
                     + [self.sample_ne() for _ in range(self.num_label - 1)]
            target_weight = [1.0] + [0.0 for _ in range(self.num_label - 1)]
            batch_data.append(context)
            batch_label.append(target)
            batch_label_weight.append(target_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)]
                label_all = [mx.nd.array(batch_label), mx.nd.array(batch_label_weight)]
                data_names = ['data']
                label_names = ['label', 'label_weight']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass

class Solver():
    def solve(self, xpu, sym, args, args_grad, auxs,
            data_iter, begin_iter = 0, end_iter = 2000, args_lrmult={}, debug = False):
        input_desc = data_iter.provide_data + data_iter.provide_label
        input_names = [k for k, shape in input_desc]
        input_buffs = [mx.nd.empty(shape, ctx=xpu) for k, shape in input_desc]
        args = dict(args, **dict(zip(input_names, input_buffs)))

        output_names = sym.list_outputs()
        if debug:
            sym = sym.get_internals()
            blob_names = sym.list_outputs()
            sym_group = []
            for i in range(len(blob_names)):
                if blob_names[i] not in args:
                    x = sym[i]
                    if blob_names[i] not in output_names:
                        x = mx.symbol.BlockGrad(x, name=blob_names[i])
                    sym_group.append(x)
            sym = mx.symbol.Group(sym_group)
        exe = sym.bind(xpu, args=args, args_grad=args_grad, aux_states=auxs)

        assert len(sym.list_arguments()) == len(exe.grad_arrays)
        update_dict = {name: nd for name, nd in zip(sym.list_arguments(), exe.grad_arrays) if nd}
        batch_size = input_buffs[0].shape[0]
        self.optimizer.rescale_grad = 1.0/batch_size
        self.optimizer.set_lr_mult(args_lrmult)

        output_dict = {}
        output_buff = {}
        internal_dict = dict(zip(input_names, input_buffs))
        for key, arr in zip(sym.list_outputs(), exe.outputs):
            if key in output_names:
                output_dict[key] = arr
                output_buff[key] = mx.nd.empty(arr.shape, ctx=mx.cpu())
            else:
                internal_dict[key] = arr

        data_iter.reset()
        for i in range(begin_iter, end_iter):
            if self.iter_start_callback is not None:
                if self.iter_start_callback(i):
                    return
            try:
                batch = data_iter.next()
            except:
                data_iter.reset()
                batch = data_iter.next()
            for data, buff in zip(batch.data+batch.label, input_buffs):
                data.copyto(buff)
            exe.forward(is_train=True)
            if self.monitor is not None:
                self.monitor.forward_end(i, internal_dict)
            for key in output_dict:
                output_dict[key].copyto(output_buff[key])

            exe.backward()
            for key, arr in update_dict.items():
                self.updater(key, arr, args[key])

            if self.metric is not None:
                self.metric.update([input_buffs[-1]],
                                   [output_buff[output_names[0]]])

            if self.monitor is not None:
                self.monitor.backward_end(i, args, update_dict, self.metric)

            if self.iter_end_callback is not None:
                if self.iter_end_callback(i):
                    return
            exe.outputs[0].wait_to_read()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-g", "--gpu", action = "store_true", dest = "gpu", default = False,
                      help = "use gpu")
    batch_size = 256
    num_label = 5
    
    data_train = DataIter("./data/text8", batch_size, num_label)
    
    network = get_net(data_train.vocab_size, num_label - 1, num_label)
    
    options, args = parser.parse_args()
    devs = mx.cpu()
    if options.gpu == True:
        devs = mx.gpu()

    
    model = mx.model.FeedForward(ctx = devs,
                                 symbol = network,
                                 num_epoch = 20,
                                 learning_rate = 0.3,
                                 momentum = 0.9,
                                 wd = 0.0000,
                                 initializer=mx.init.Xavier(factor_type="in", magnitude=2.34))

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    
    metric = NceAuc()
    model.fit(X = data_train,
              eval_metric = metric,)
    a = model.get_params()
    print a['arg_params']['embed_weight'].shape
    open('matrix_embedding','w').write((str(a['arg_params']['embed_weight'].asnumpy())))

