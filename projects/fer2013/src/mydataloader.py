# -*- coding: utf-8 -*-
import torch
from torchvision import datasets, transforms

def trainloader1(data_dir, batch_size, num_workers=4):
	data_transforms = transforms.Compose([
		transforms.Grayscale(),
		transforms.RandomResizedCrop(42),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
	])

	mydatasets = datasets.ImageFolder(data_dir, transform=data_transforms)
	myloader = torch.utils.data.DataLoader(mydatasets, batch_size, shuffle=True, num_workers=num_workers)
	return myloader

def trainloader2(data_dir, batch_size, num_workers=4):
	"""
		FiveCrop 
	"""
	data_transforms = transforms.Compose([
		transforms.Grayscale(),
		transforms.FiveCrop(42),
		transforms.Lambda(
			lambda crops: torch.stack([transforms.ToTensor()(crop)
				for crop in crops]))
	])

	mydatasets = datasets.ImageFolder(data_dir, transform=data_transforms)
	myloader = torch.utils.data.DataLoader(mydatasets, batch_size, shuffle=True, num_workers=num_workers)
	return myloader

def trainloader3(data_dir, batch_size, num_workers=4):
	"""
		FiveCrop, RandomHorizontalFlip
	"""
	data_transforms = transforms.Compose([
		transforms.Grayscale(),
		transforms.FiveCrop(42),
		transforms.Lambda(
			lambda crops: torch.stack([transforms.ToTensor()(transforms.RandomHorizontalFlip()(crop)) 
				for crop in crops]))
	])

	mydatasets = datasets.ImageFolder(data_dir, transform=data_transforms)
	myloader = torch.utils.data.DataLoader(mydatasets, batch_size, shuffle=True, num_workers=num_workers)
	return myloader


def valloader(data_dir, batch_size, num_workers=4):
	data_transforms = transforms.Compose([
		transforms.Grayscale(),
		transforms.CenterCrop(42),
		transforms.ToTensor()
	])

	mydatasets = datasets.ImageFolder(data_dir, transform=data_transforms)
	myloader = torch.utils.data.DataLoader(mydatasets, batch_size, shuffle=True, num_workers=num_workers)
	return myloader


def testloader(data_dir, batch_size, num_workers=4):
	data_transforms = transforms.Compose([
		transforms.Grayscale(),
		transforms.CenterCrop(42),
		transforms.ToTensor()
	])

	mydatasets = datasets.ImageFolder(data_dir, transform=data_transforms)
	myloader = torch.utils.data.DataLoader(mydatasets, batch_size, shuffle=True, num_workers=num_workers)
	return myloader