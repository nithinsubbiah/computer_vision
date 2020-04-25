from torch.utils.data import Dataset
from external.vqa.vqa import VQA

import operator
from itertools import islice
import os
import numpy as np

from PIL import Image
from torchvision import transforms
import torch

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

class VqaDataset(Dataset):
    """
    Load the VQA dataset using the VQA python API. We provide the necessary subset in the External folder, but you may
    want to reference the full repo (https://github.com/GT-Vision-Lab/VQA) for usage examples.
    """

    def __init__(self, image_dir, question_json_file_path, annotation_json_file_path, image_filename_pattern,
                 transform=None, question_word_to_id_map=None, answer_to_id_map=None, question_word_list_length=5746, answer_list_length=5216,
                 pre_encoder=None, cache_location=None):
        """
        Args:
            image_dir (string): Path to the directory with COCO images
            question_json_file_path (string): Path to the json file containing the question data
            annotation_json_file_path (string): Path to the json file containing the annotations mapping images, questions, and
                answers together
            image_filename_pattern (string): The pattern the filenames of images in this dataset use (eg "COCO_train2014_{}.jpg")
        """
        self._vqa = VQA(annotation_file=annotation_json_file_path, question_file=question_json_file_path)
        self._image_dir = image_dir
        self._image_filename_pattern = image_filename_pattern
        self._transform = transform
        self._max_question_length = 26
        self.ques_ids = self._vqa.getQuesIds()

        # Publicly accessible dataset parameters
        self.question_word_list_length = question_word_list_length + 1
        self.unknown_question_word_index = question_word_list_length
        self.answer_list_length = answer_list_length + 1
        self.unknown_answer_index = answer_list_length
        self._pre_encoder = pre_encoder
        self._cache_location = cache_location
        if self._cache_location is not None:
            try:
                os.makedirs(self._cache_location)
            except OSError:
                pass

        # Create the question map if necessary
        if question_word_to_id_map is None:
            questions_list = []
            questions = self._vqa.questions['questions']

            for question in questions:
                questions_list.append(question['question'])

            word_list = self._create_word_list(questions_list)
            self.question_word_to_id_map = self._create_id_map(word_list,self.question_word_list_length)
        else:
            self.question_word_to_id_map = question_word_to_id_map

        # Create the answer map if necessary
        if answer_to_id_map is None:
            answer_list = []
            answers = self._vqa.dataset['annotations']
            for answer in answers:
                all_answers = answer['answers']
                for each_answer in all_answers:
                    answer_list.append(each_answer['answer'])

            self.answer_to_id_map = self._create_id_map(answer_list,self.answer_list_length)

        else:
            self.answer_to_id_map = answer_to_id_map


    def _create_word_list(self, sentences):
        """
        Turn a list of sentences into a list of processed words (no punctuation, lowercase, etc)
        Args:
            sentences: a list of str, sentences to be splitted into words
        Return:
            A list of str, words from the split, order remained.
        """
	
        word_list = []
        # Source: https://www.geeksforgeeks.org/removing-punctuations-given-string/
        for sentence in sentences:
            sentence = sentence.lower()
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for x in sentence: 
                if x in punctuations:
                    sentence = sentence.replace(x, "")
            word_list.extend(sentence.split(" "))
        
        return word_list

    def _create_id_map(self, word_list, max_list_length):
        """
        Find the most common str in a list, then create a map from str to id (its rank in the frequency)
        Args:
            word_list: a list of str, where the most frequent elements are picked out
            max_list_length: the number of strs picked
        Return:
            A map (dict) from str to id (rank)
        """

        freq_words = {}

        for word in word_list:
            if word in freq_words:
                freq_words[word] += 1
            else:
                freq_words[word] = 1

        # Sort dictionary by frequency of words
        freq_words = dict(sorted(freq_words.items(), key=operator.itemgetter(1),reverse=True))
        
        # Update dictionary for the max list length
        freq_words = take(max_list_length,freq_words.items())

        freq_words = [(val[0], idx) for idx, val in enumerate(freq_words)]
        freq_words = dict(freq_words)

        return freq_words

    def __len__(self):
        
        return len(self.ques_ids)

    def __getitem__(self, idx):
        """
        Load an item of the dataset
        Args:
            idx: index of the data item
        Return:
            A dict containing multiple torch tensors for image, question and answers.
        """

        ques_id = self.ques_ids[idx]
        question = self._vqa.loadQA(ques_id)
        img_id = question[0]['image_id']
        img_id = str(img_id)
        img_id = img_id.zfill(12)

        if self._cache_location is not None and self._pre_encoder is not None:
            ############ 3.2 TODO
            # implement your caching and loading logic here

            ############
            raise NotImplementedError()
        else:
            # load the image from disk, apply self._transform (if not None)
            fpath = os.path.join(self._image_dir,self._image_filename_pattern.format(img_id))
            img = Image.open(fpath)
            img = img.convert('RGB')
            if self._transform:
                img = self._transform(img)
            else:
                #TODO: Check if this is right
                img = transforms.functional.to_tensor(img)

        question = self._vqa.questions['questions'][idx]['question']
        question_split = self._create_word_list([question])
        
        total_question_words = np.array(list(self.question_word_to_id_map.keys()))
        
        question_one_hot = np.zeros([self._max_question_length,self.question_word_list_length])

        for idx, word in enumerate(question_split):
            if idx == self._max_question_length:
                break
            contains_word = ((word in total_question_words) == True)
            if contains_word:
                hot_idx = np.where(word==total_question_words)[0][0]
                question_one_hot[idx,hot_idx] = 1
            else:
                question_one_hot[idx,-1] = 1
        question_one_hot = torch.from_numpy(question_one_hot)

        question_one_hot = torch.clamp(torch.sum(question_one_hot,dim=0),max=1)

        answers = self._vqa.dataset['annotations'][idx]['answers']
        answers_one_hot = np.zeros([10,self.answer_list_length])
        total_answer_words = np.array(list(self.answer_to_id_map.keys()))

        for idx, answer_dict in enumerate(answers):
            answer = answer_dict['answer']
            contains_word = ((answer in total_answer_words) == True)
            if contains_word:
                hot_idx = np.where(answer==total_answer_words)[0][0]
                answers_one_hot[idx,hot_idx] = 1
            else:
                answers_one_hot[idx,-1] = 1
        
        answers_one_hot = torch.from_numpy(answers_one_hot)        
        #img = img.cuda()
        #question_one_hot = question_one_hot.cuda()
        #answers_one_hot_list = answers_one_hot_list.cuda()

        #datapoint = {'image':img, 'question_tensor':question_one_hot, 'answers_tensor':answers_one_hot}   

        #return datapoint 
        return img, question_one_hot, answers_one_hot
        
