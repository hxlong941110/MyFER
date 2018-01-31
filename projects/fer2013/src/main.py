# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time

import mydataloader
from mynets.alexnet import AlexNet
from mynets.mynet import MyNet
from utils import AverageMeter, accuracy, save_checkpoint, adjust_learning_rate, save_log


## -------------------------- 变量 ----------------------------------

data_dir = r'/home/zwx/Datasets/fer2013/'  # 数据集根路径
resume = r'checkpoint.pth.tar' # 从checkpoint开始

checkpoint_name = 'checkpoint.pth.tar'
model_savepath = 'models'  # 模型存放路径
model_savename = 'model_best.pth.tar' # 保存模型名称

multi_crops = True
batch_size = 512  # mini batch size
print_freq = 20  # 打印间隔

start_epoch = 0 # 迭代开始位置
epochs = 300  # 迭代次数

log_file = 'logs/train_fivecrop.log'
## ------------------------------------------------------------------

use_gpu = torch.cuda.is_available()
# if use_gpu:
# 	gpuid = 0
# 	torch.cuda.set_device(gpuid)  # 指定GPU ID
gpu_count = torch.cuda.device_count() # GPU 个数


def main():
	global start_epoch, epochs, log_file
	best_prec1 = 0.0
	best_epoch = 0
	# ---------------------------------- test dataloader ------------------------------------
	# train_loader1 = mydataloader.trainloader1(os.path.join(data_dir, 'train'), batch_size)
	# print('train loader mini-batch counts: {}'.format(len(train_loader1)))
	train_loader2 = mydataloader.trainloader2(os.path.join(data_dir, 'train'), batch_size)
	print('train loader mini-batch counts: {}'.format(len(train_loader2)))
	# train_loader3 = mydataloader.trainloader3(os.path.join(data_dir, 'train'), batch_size)
	# print('train loader mini-batch counts: {}'.format(len(train_loader3)))
	# ---------------------------------------------------------------------------------------

	# model = AlexNet(num_classes=7)
	model = MyNet(num_classes=7)

	criterion = nn.CrossEntropyLoss()  # 损失函数
	optimizer = optim.Adam(model.parameters())  # 优化函数

	# 加载checkpoint,如果存在的话
	if os.path.isfile(resume):
		print("=> loading checkpoint '{}'".format(resume))
		checkpoint = torch.load(resume)
		start_epoch = checkpoint['epoch']  # 训练开始位置
		best_prec1 = checkpoint['best_prec1']
		model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print("=> loaded checkpoint '{}' (epoch {}) best prec@1 {:.4f}"
			.format(resume, checkpoint['epoch'], best_prec1))
	else:
		print("=> no checkpoint found at '{}'".format(resume))

	# 验证集数据读取器	
	val_loader = mydataloader.valloader(os.path.join(data_dir, 'val'), batch_size)

	# # 使用多GPU
	# if gpu_count > 1:
	# 	model = nn.DataParallel(model)

	if use_gpu:
		model = model.cuda()
		criterion = criterion.cuda()

	# 打开日志文件记录
	for epoch in range(start_epoch, epochs):
		# train for one epoch
		train(train_loader2, model, criterion, optimizer, epoch, multi_crops)  # 选择需要的trainloader

		# evaludate on validation set
		prec1 = validate(val_loader, model, criterion)

		# remember best prec@1 and save checkpoint
		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		if is_best:
			best_epoch = epoch
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.state_dict(),
			'best_prec1': best_prec1,
			'optimizer': optimizer.state_dict(),
			}, is_best, model_savepath, model_savename)
		print('**************** best prec@1 {:.4f} at epoch[{}] ********************'
			.format(best_prec1, best_epoch))



def train(trainloader, model, criterion, optimizer, epoch, multi_crops):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top2 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input_data, target) in enumerate(trainloader): # 训练集:裁剪42， 水平翻转
		data_time.update(time.time() - end)

		if use_gpu:
			input_data = input_data.cuda()
			target =target.cuda()
		input_var = torch.autograd.Variable(input_data)
		target_var = torch.autograd.Variable(target)

		if multi_crops:
			## ----------------- 训练集FiveCrop --------------------
			# compute output
			bs, ncrops, c , h, w = input_data.size()
			output = model(input_var.view(-1, c, h, w))
			output = output.view(bs, ncrops, -1).mean(1)  # output average
		else:
			## ----------------- 训练集只有水平翻转 ------------------
			# compute output
			output = model(input_var)

		# compute loss
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
		losses.update(loss.data[0], input_data.size(0))
		top1.update(prec1[0], input_data.size(0))
		top2.update(prec2[0], input_data.size(0))

		# compute gradient and do optimizer step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			info = 'Epoch: [{0}/{1}] [{2}/{3}]\t'\
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
				'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'\
				'Loss {loss.val:4f} ({loss.avg:.4f})\t'\
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
				'Prec@2 {top2.val:.3f} ({top2.avg:.3f})\t'.format(
				epoch, epochs, i, len(trainloader), batch_time=batch_time,
				data_time=data_time, loss=losses, top1=top1, top2=top2)
			print(info)
			save_log(info, log_file)
	info = ' * Training [{0}/{1}]: Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f}'.format(epoch, epochs, top1=top1, top2=top2)
	save_log(info, log_file)

def validate(valloader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top2 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	end = time.time()
	for i, (input_data, target) in enumerate(valloader):
		if use_gpu:
			input_data = input_data.cuda(async=True)
			target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input_data, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec2 = accuracy(output.data, target, topk=(1, 2))
		losses.update(loss.data[0], input_data.size(0))
		top1.update(prec1[0], input_data.size(0))
		top2.update(prec2[0], input_data.size(0))

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % print_freq == 0:
			info = 'Testing: [{0}/{1}]\t'\
				'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'\
				'Loss {loss.val:.4f} ({loss.avg:.4f})\t'\
				'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'\
				'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
					i, len(valloader), batch_time=batch_time, loss=losses,
					top1=top1, top2=top2)
			print(info)
			save_log(info, log_file)

	info = '* Test:  Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f}'.format(top1=top1, top2=top2)
	print(info)
	save_log(info, log_file)
	return top1.avg

if __name__ == '__main__':
	main()