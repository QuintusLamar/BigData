import re
import cv2
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')

def tokenize(sentence):
  tokens = SENTENCE_SPLIT_REGEX.split(sentence.lower())
  tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
  return tokens

def load_str_list(fname):
  with open(fname) as f:
    lines = f.readlines()
  lines = [l.strip() for l in lines]
  return lines

def resize_image(image, size):
  return cv2.resize(image, size, interpolation = cv2.INTER_AREA)
