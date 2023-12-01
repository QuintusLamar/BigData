import re, os, sys
import argparse
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models

from support.VocabDictionary import VocabDict
from support.VQAModel import ImgEncoder, QstEncoder, VqaModel, ImgAttentionEncoder, Attention, SANModel
from support.VQADataset import VqaDataset, Args, get_loader
from resources.helper_functions import tokenize, load_str_list, resize_image

# Works without function if we want
# args = Args()

# data_loader = get_loader(
# 	input_dir=args.input_dir,
# 	input_vqa_train='train.npy',
# 	input_vqa_valid='valid.npy',
# 	input_vqa_test='test-dev.npy',
# 	max_qst_length=args.max_qst_length,
# 	max_num_ans=args.max_num_ans,
# 	batch_size=args.batch_size,
# 	num_workers=args.num_workers
# )

# path = "./resources/resizedTest.jpg"
# resizedImg = cv2.imread(path)

# transform = {
# 	phase: transforms.Compose([transforms.ToTensor(),
# 		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
# 		for phase in ['train', 'valid', 'test']
# }

# transformed_img = transform['test'](resizedImg)

# q_vocab = VocabDict('./resources/vocab_questions.txt')
# max_q_length = 30
# tokens = tokenize("What is he doing?")
# q2idc = np.array([q_vocab.word2idx('<pad>')] * max_q_length) 
# q2idc[:len(tokens)] = [q_vocab.word2idx(w) for w in tokens]

# model = torch.load('./resources/best_model.pt', map_location=torch.device('cpu'))
# test_q = torch.from_numpy(q2idc)

# test_q = torch.from_numpy(q2idc)

# new_imgs = transformed_img.unsqueeze(0)
# new_qs = test_q.unsqueeze(0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# output = model(new_imgs.to(device), new_qs.to(device))

# _, pred_exp1 = torch.max(output, 1)  
# _, pred_exp2 = torch.max(output, 1)

# print(data_loader['test'].dataset.ans_vocab.idx2word(41))

def generateAnswer(img, question):
	transform = {
		phase: transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
			for phase in ['train', 'valid', 'test']
	}

	resizedImg = resize_image(img, (224, 224))
	transformed_img = transform['test'](resizedImg)

	args = Args()

	data_loader = get_loader(
		input_dir=args.input_dir,
		input_vqa_train='train.npy',
		input_vqa_valid='valid.npy',
		input_vqa_test='test-dev.npy',
		max_qst_length=args.max_qst_length,
		max_num_ans=args.max_num_ans,
		batch_size=args.batch_size,
		num_workers=args.num_workers
	)

	q_vocab = VocabDict('./resources/vocab_questions.txt')
	max_q_length = 30
	tokens = tokenize(question)
	q2idc = np.array([q_vocab.word2idx('<pad>')] * max_q_length) 
	q2idc[:len(tokens)] = [q_vocab.word2idx(w) for w in tokens]

	model = torch.load('./resources/best_model.pt', map_location=torch.device('cpu'))
	test_q = torch.from_numpy(q2idc)

	test_q = torch.from_numpy(q2idc)

	new_imgs = transformed_img.unsqueeze(0)
	new_qs = test_q.unsqueeze(0)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	output = model(new_imgs.to(device), new_qs.to(device))

	_, pred_exp1 = torch.max(output, 1)  # [batch_size]
	_, pred_exp2 = torch.max(output, 1)  # [batch_size]

	ans = data_loader['test'].dataset.ans_vocab.idx2word(pred_exp1.item())
	return ans


# Test works
# test_img = cv2.imread('./resources/resizedTest.jpg')
# answer = generateAnswer(test_img, "What is the floor made of?")
# print("Answer: ", answer)