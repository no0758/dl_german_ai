import torch
from collections import OrderedDict
from torch import nn
from collections import defaultdict
from torch.utils.trainer.plugins.plugin import Plugin
from datetime import datetime


### load model
def load(model,train_mode,pretrained_path=None):
    # train_mode: 0:from scratch, 1:finetuning, 2:update
    # if not update all parameters:
    # for param in list(model.parameters())[:-1]:    # only update parameters of last layer
    #    param.requires_grad = False
    if train_mode == 'fromscratch':
        is_available(model)
        is_parallel(model)
        print('from scratch............................')

    elif train_mode == 'finetune':
        _load(model,pretrained_path)
        is_available(model)
        is_parallel(model)
        print('finetuning...............................')

    elif train_mode == 'update':
        _load(model,pretrained_path)
        print('updating...............................')
    else:
        ValueError('train_mode is error...')


def _load(model,pretrained_path):
    _state_dict = torch.load(pretrained_path,map_location=None) if torch.cuda.is_available() else torch.load(pretrained_path,map_location='cpu')
    # for multi-gpus
    state_dict = OrderedDict()
    for item, value in _state_dict.items():
        if 'module' in item.split('.')[0]:
            name = '.'.join(item.split('.')[1:])
        else:
            name = item
        state_dict[name] = value
    # for handling in case of different models compared to the saved pretrain-weight
    model_dict = model.state_dict()
    diff = {k: v for k, v in model_dict.items() if \
            k in state_dict and state_dict[k].size() != v.size()}
    print('diff: ', [i for i, v in diff.items()])
    state_dict.update(diff)
    model.load_state_dict(_state_dict)

def is_parallel(obeject):
    print(obeject.__module__,'data parallel.......')
    if torch.cuda.device_count() > 1:
        obeject = nn.DataParallel(obeject,device_ids=range(torch.cuda.device_count()))
    # return obeject

def is_available(obeject):
    print(obeject.__module__, 'cuda.......')
    if torch.cuda.is_available():
        obeject.cuda()

def is_adaptive(model,fc_num,num_classes,num_channels):
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(fc_num, num_classes)


def channels_conv(model,fc_num,num_classes):
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    model.fc = nn.Linear(fc_num, num_classes)


class Logger(Plugin):
    alignment = 4
    separator = '#' * 80

    def __init__(self, fields, interval=None, file_path=None):
        if interval is None:
            interval = [(1, 'iteration'), (1, 'epoch')]
        super(Logger, self).__init__(interval)
        self.field_widths = defaultdict(lambda: defaultdict(int))
        self.fields = list(map(lambda f: f.split('.'), fields))
        self.file_path = file_path

    def _join_results(self, results):
        joined_out = map(lambda i: (i[0], ' '.join(i[1])), results)
        joined_fields = map(lambda i: '{}: {}'.format(i[0], i[1]), joined_out)
        return '\t'.join(joined_fields)

    def log(self, msg):
        print(msg)

    def register(self, trainer):
        self.trainer = trainer

    def gather_stats(self):
        result = {}
        return result

    def _align_output(self, field_idx, output):
        for output_idx, o in enumerate(output):
            if len(o) < self.field_widths[field_idx][output_idx]:
                num_spaces = self.field_widths[field_idx][output_idx] - len(o)
                output[output_idx] += ' ' * num_spaces
            else:
                self.field_widths[field_idx][output_idx] = len(o)

    def _gather_outputs(self, field, log_fields, stat_parent, stat, require_dict=False):
        output = []
        name = ''
        if isinstance(stat, dict):
            log_fields = stat.get(log_fields, [])
            name = stat.get('log_name', '.'.join(field))
            for f in log_fields:
                output.append(f.format(**stat))
        elif not require_dict:
            name = '.'.join(field)
            number_format = stat_parent.get('log_format', '')
            unit = stat_parent.get('log_unit', '')
            fmt = '{' + number_format + '}' + unit
            output.append(fmt.format(stat))
        return name, output

    def _log_all(self, log_fields, prefix=None, suffix=None, require_dict=False):
        results = []
        for field_idx, field in enumerate(self.fields):
            parent, stat = None, self.trainer.stats
            for f in field:
                parent, stat = stat, stat[f]
            name, output = self._gather_outputs(field, log_fields,
                                                parent, stat, require_dict)
            if not output:
                continue
            self._align_output(field_idx, output)
            results.append((name, output))
        if not results:
            return
        output = self._join_results(results)
        if prefix is not None:
            self.log(prefix)
        self.log(output)
        if suffix is not None:
            self.log(suffix)
        return output

    def iteration(self, *args):
        self._log_all('log_iter_fields')

    def epoch(self, epoch_idx):
        output = self._log_all('log_epoch_fields',
                               prefix=self.separator + '\nEpoch summary:',
                               suffix=self.separator,
                               require_dict=True)
        if self.file_path is not None:
            with open(self.file_path, 'a+') as f:
                f.write('\n' + str(datetime.now())[0:19] + ' ' + 'epoch' + str(epoch_idx) + ':' + output)