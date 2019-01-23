from datetime import datetime
import os
from collections import OrderedDict
import torch
import heapq
from torch import nn
from torch.autograd import Variable
import sys
from torch.optim.lr_scheduler import StepLR
from utils import _load

class Trainer(object):

    def __init__(self, model=None, criterion=None, optimizer=None, dataset=None,
                 valid_dataset=None, file_path=None, save_path='val_model'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.valid_dataset = valid_dataset
        self.iterations = 0
        self.valid_dataset_nums = self.valid_dataset.dataset.__len__()
        self.dataset_nums = self.dataset.dataset.__len__() if dataset is not None else 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }
        self.file_path = file_path
        self.accuracy = 0
        self.save_path = save_path
        self.step_size = 20
        self.scheduler = StepLR(optimizer, step_size=self.step_size, gamma=0.1)

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for duration, unit in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        # evaluate
        self.log('=' * 5 + '*'*10 + '=' * 5)
        self.evaluate(0)
        for i in range(1, epochs + 1):
            # self.scheduler.step()
            # if i%self.step_size ==0:
            #     lr_msg = '\nlr updating........lr = '+str(self.scheduler.get_lr())
            #     print(lr_msg)
            #     self.log(lr_msg)
            self.train(i)
            self.call_plugins('epoch', i)
            self.evaluate(i)


    def train(self,epoch_idx):
        # Set the model to be in training mode (for dropout and batchnorm)
        self.model.train()
        for i, data in enumerate(self.dataset, self.iterations + 1):
            batch_input, batch_target1, batch_target = data
            if torch.cuda.is_available():
                batch_input, batch_target1, batch_target = batch_input.cuda(async=True), batch_target1.cuda(async=True), batch_target.cuda(async=True)
            batch_input, batch_target1, batch_target = Variable(batch_input), Variable(batch_target1), Variable(batch_target)
            self.call_plugins('batch', i, batch_input, batch_target)
            input_var = batch_input
            target_var = batch_target
            target_var1 = batch_target1
            plugin_data = [None, None]

            def closure():
                batch_output1,batch_output = self.model(input_var)
                loss = self.criterion(batch_output, target_var) + self.criterion(batch_output1, target_var1)
                loss.backward()
                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data
                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins('iteration', i, batch_input, batch_target,
                              *plugin_data)
            self.call_plugins('update', i, self.model)
            if i%500==0:
                self.evaluate(epoch_idx)
            # self.evaluate(epoch_idx)
        self.iterations += i


    def evaluate(self, epoch_idx):
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.eval() #####
        e_loss = 0
        accu_num = 0
        batch_num = len(self.valid_dataset)
        for i, data in enumerate(self.valid_dataset):
            batch_input, batch_target1, batch_target = data
            if torch.cuda.is_available():
                batch_input, batch_target1, batch_target = batch_input.cuda(async=True), batch_target1.cuda(async=True), batch_target.cuda(async=True)
            batch_input, batch_target1, batch_target = Variable(batch_input), Variable(batch_target1), Variable(batch_target)
            input_var = batch_input
            target_var = batch_target
            target_var1 = batch_target1
            batch_output1, batch_output = self.model(input_var)
            loss = self.criterion(batch_output, target_var) + self.criterion(batch_output1, target_var1)
            e_loss += loss.item() * input_var.shape[0] #####
            accu_num += torch.sum(torch.argmax(batch_output.data, 1) == batch_target.data) #####
            percent = float(i) * 100 / float(batch_num)
            sys.stdout.write("%.4f" % percent);
            sys.stdout.write("%\r")
            sys.stdout.flush()
        sys.stdout.write("100%!finish!\r")
        sys.stdout.flush()
        msg = 'evalute_epoch{} ,evaluate_loss:{} ,evaluate_accuracy:{:.3f}.'
        accu = 100 * accu_num.float() / self.valid_dataset_nums
        loss = e_loss / self.valid_dataset_nums
        log_msg = msg.format(epoch_idx, loss, accu)
        print(log_msg)
        self.log(log_msg)
        if accu > self.accuracy:
            self.save_model(self.model, self.save_path)
            self.accuracy = accu
        else:
            pass
            # _load(self.model,os.path.join(self.save_path, 'eval_best'+'.model'))


    def save_model(self, model, save_path):
        print("Saving model...")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_name = os.path.join(save_path, 'eval_best'+'.model') #str(int(time.time()))
        state_dict = OrderedDict()
        for item, value in model.state_dict().items():
            if 'module' in item.split('.')[0]:
                name = '.'.join(item.split('.')[1:])
            else:
                name = item
            state_dict[name] = value
        torch.save(state_dict, save_name)
        print("Model saved: ", save_name)
        self.log("\nModel saved: " + save_name)

    def log(self, msg):
        if self.file_path is not None:
            with open(self.file_path, 'a+') as f:
                f.write('\n' + str(datetime.now())[0:19] + ' ' + msg)

    def get_loss(self, pred, label):
        """
        compute loss
        :param pred:
        :param label:
        :return:
        """
        #        criterion=F.cross_entropy(pred, label,reduction='none')
        criterion = nn.CrossEntropyLoss()  # nn.MSELoss()
        if torch.cuda.is_available():
            criterion.cuda()
        return criterion(pred, label)



    def test(self):
        # Set the model to be in testing mode (for dropout and batchnorm)
        self.model.eval()  #####
        e_loss = 0
        accu_num = 0
        batch_num = len(self.valid_dataset)
        for i, data in enumerate(self.valid_dataset):
            batch_input, batch_target = data
            if torch.cuda.is_available():  #####
                batch_input, batch_target = batch_input.cuda(async=True), batch_target.cuda(async=True)
            batch_input, batch_target = Variable(batch_input), Variable(batch_target)  #####
            input_var = batch_input
            target_var = batch_target
            batch_output = self.model(input_var)
            loss = self.get_loss(batch_output, target_var)
            e_loss += loss.item() * input_var.shape[0]  #####
            accu_num += torch.sum(torch.argmax(batch_output.data, 1) == batch_target.data)  #####
            percent = float(i) * 100 / float(batch_num)
            sys.stdout.write("%.4f" % percent);
            sys.stdout.write("%\r")
            sys.stdout.flush()
        sys.stdout.write("100%!finish!\r")
        sys.stdout.flush()
        msg = 'test_loss:{} ,test_accuracy:{:.3f}.'
        accu = 100 * accu_num.float() / self.valid_dataset_nums
        loss = e_loss / self.valid_dataset_nums
        log_msg = msg.format(loss, accu)
        print(log_msg)
        self.log(log_msg)