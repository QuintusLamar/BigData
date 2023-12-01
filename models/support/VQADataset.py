import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
from support.VocabDictionary import VocabDict
import torch
import re


SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')


class Args:
	model_name = 'best_model'
	input_dir = './resources'
	log_dir = 'logs'
	model_dir = 'models'
	max_qst_length = 30
	max_num_ans = 10
	embed_size = 1024
	word_embed_size = 300
	num_layers = 2
	hidden_size = 512
	learning_rate = 0.001
	step_size = 10
	gamma = 0.1
	num_epochs = 16 #30
	batch_size = 256
	num_workers = 8
	save_step = 1
	resume_epoch = 15
	saved_model = './resources/best_model.pt'


# Data Loader
class VqaDataset(data.Dataset):
	def __init__(self, input_dir, input_vqa, max_qst_length=30, max_num_ans=10, transform=None):
		self.input_dir = input_dir
		self.vqa = np.load(input_dir+'/'+input_vqa, allow_pickle=True)
		self.qst_vocab = VocabDict('./resources/vocab_questions.txt')
		self.ans_vocab = VocabDict('./resources/vocab_answers.txt')
		self.max_qst_length = max_qst_length
		self.max_num_ans = max_num_ans
		self.load_ans = ('valid_answers' in self.vqa[0]) and (self.vqa[0]['valid_answers'] is not None)
		self.transform = transform

	def __getitem__(self, idx):
		vqa = self.vqa
		qst_vocab = self.qst_vocab
		ans_vocab = self.ans_vocab
		max_qst_length = self.max_qst_length
		max_num_ans = self.max_num_ans
		transform = self.transform
		load_ans = self.load_ans

		image = vqa[idx]['image_path']

		if "COCO_val2014_000000380348" in image or "COCO_val2014_000000262148" in image:
			image = Image.open(image).convert('RGB')
			qst2idc = np.array([qst_vocab.word2idx('<pad>')] * max_qst_length)  # padded with '<pad>' in 'ans_vocab'
			qst2idc[:len(vqa[idx]['question_tokens'])] = [qst_vocab.word2idx(w) for w in vqa[idx]['question_tokens']]
			sample = {'image': image, 'question': qst2idc}

			if load_ans:
				ans2idc = [ans_vocab.word2idx(w) for w in vqa[idx]['valid_answers']]
				ans2idx = np.random.choice(ans2idc)
				sample['answer_label'] = ans2idx         # for training

				mul2idc = list([-1] * max_num_ans)       # padded with -1 (no meaning) not used in 'ans_vocab'
				mul2idc[:len(ans2idc)] = ans2idc         # our model should not predict -1
				sample['answer_multi_choice'] = mul2idc  # for evaluation metric of 'multiple choice'

			if transform:
				sample['image'] = transform(sample['image'])

			return sample

		else: 
			return ['None']


	def __len__(self):
		return len(self.vqa)


def get_loader(input_dir, input_vqa_train, input_vqa_valid, input_vqa_test, max_qst_length, max_num_ans, batch_size, num_workers):
	transform = {
		phase: transforms.Compose([transforms.ToTensor(),
			transforms.Normalize((0.485, 0.456, 0.406),
			(0.229, 0.224, 0.225))])
		for phase in ['train', 'valid', 'test']
	}

	vqa_dataset = {
		'train': VqaDataset(
			input_dir=args.input_dir,
			input_vqa=input_vqa_train,
			max_qst_length=max_qst_length,
			max_num_ans=max_num_ans,
			transform=transform['train']),
		'valid': VqaDataset(
			input_dir=args.input_dir,
			input_vqa=input_vqa_valid,
			max_qst_length=max_qst_length,
			max_num_ans=max_num_ans,
			transform=transform['valid']),
		'test': VqaDataset(
			input_dir=args.input_dir,
			input_vqa=input_vqa_test,
			max_qst_length=max_qst_length,
			max_num_ans=max_num_ans,
			transform=transform['test'])
	}

	data_loader = {
		phase: torch.utils.data.DataLoader(
			dataset=vqa_dataset[phase],
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers)
		for phase in ['train', 'valid', 'test']}

	return data_loader

args = Args()