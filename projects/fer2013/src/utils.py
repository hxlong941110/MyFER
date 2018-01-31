# -*- coding: utf-8 -*-
from __future__ import division
import torch
import shutil
import os

class AverageMeter():
	def __init__(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
	maxk = max(topk)
	batch_size = target.size(0)

	# torch.Tensor.topk(k, dim=None, largest=True, sorted=True, out=None)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t() # 转置

	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		# correct_k = correct[:k].view(-1)
		# correct_k = correct.float()
		# correct_k = correct_k.sum()
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def save_checkpoint(state, is_best, model_savepath, model_name='model_best.pth.tar', checkpoint_name='checkpoint.pth.tar'):
	torch.save(state, checkpoint_name)
	if is_best:
		shutil.copyfile(checkpoint_name, os.path.join(model_savepath, model_name))
		shutil.copyfile

def save_log(info, log_file):
	with open(log_file, 'a+') as f:
		f.writelines(info+'\n')


if __name__ == '__main__':
	# # -------------------- test AverageMeter ---------------------------------
	# average = AverageMeter()
	# average.update(1)
	# print('val: {} sum: {} count: {} avg: {}'.format(average.val, average.sum, average.count, average.avg))
	# average.update(2)
	# print('val: {} sum: {} count: {} avg: {}'.format(average.val, average.sum, average.count, average.avg))
	# average.update(3)
	# print('val: {} sum: {} count: {} avg: {}'.format(average.val, average.sum, average.count, average.avg))
	# average.update(4)
	# print('val: {} sum: {} count: {} avg: {}'.format(average.val, average.sum, average.count, average.avg))

	# test accuracy
	output = torch.LongTensor([[1,2,3,4],
		[4,2,1,3],
		[1,4,3,2]])
	target = torch.LongTensor([3, 0, 2])
	

	res = accuracy(output, target, (1, 2))
	print(res)
