import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from tensorboardX import SummaryWriter

import random

class ExperimentRunnerBase(object):
    """
    This base class contains the simple train and validation loops for your VQA experiments.
    Anything specific to a particular experiment (Simple or Coattention) should go in the corresponding subclass.
    """

    def __init__(self, train_dataset, val_dataset, model, batch_size, num_epochs, num_data_loader_workers=10, log_validation=False):
        self._model = model
        self._num_epochs = num_epochs
        self._log_freq = 10  # Steps
        self._test_freq = 250  # Steps
        self.batch_size = batch_size

        self._train_dataset_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_data_loader_workers)

        # If you want to, you can shuffle the validation dataset and only use a subset of it to speed up debugging
        self._val_dataset_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_data_loader_workers)

        # Use the GPU if it's available.
        self._cuda = torch.cuda.is_available()

        if self._cuda:
            self._model = self._model.cuda()

        self._log_validation = log_validation
        self.writer = SummaryWriter()

    def _optimize(self, predicted_answers, true_answers):
        """
        This gets implemented in the subclasses. Don't implement this here.
        """
        raise NotImplementedError()

    def validate(self):
        ############ 2.8 TODO
        # Should return your validation accuracy
        no_correct = 0
        no_total = 0

        plt_img = None
        plt_question = None
        plt_gt_answer = None
        plt_pred_answer = None
        chosen = False

        for batch_id, batch_data in enumerate(self._val_dataset_loader):

            input_images, questions, answers, question_word, answer_word = batch_data
            input_images = input_images.cuda()
            questions = questions.cuda()
            answers = answers.cuda()
            
            predicted_answer = self._model(input_images,questions)

            if not chosen:
                rand_idx = random.randint(0,questions.shape[0])
                plt_img = input_images[rand_idx]
                plt_question = question_word[rand_idx]
                plt_gt_answer = answer_word[rand_idx]
                plt_pred_answer = predicted_answer[rand_idx]

            output_idx = torch.argmax(torch.sum(answers,dim=1),dim=1)
            
            _, predicted_idx = torch.max(predicted_answer,dim=1)
            no_correct += (output_idx==predicted_idx).sum().item()
            no_total += predicted_idx.shape[0]

        val_accuracy = no_correct/no_total

        if self._log_validation:
            identifier = random.randint(0,1000)
            self.writer.add_image('val/image_epoch'+str(self._num_epochs)+'_'+str(identifier), plt_img)
            self.writer.text('val/question'+str(self._num_epochs)+'_'+str(identifier), plt_question)
            self.writer.text('val/gt_ans'+str(self._num_epochs)+'_'+str(identifier), plt_gt_answer)
            # self.writer.text('val/pred_ans'+str(self._num_epochs)+'_'+str(identifier), plt_pred_answer)
            
        return val_accuracy

    def train(self):

        for epoch in range(self._num_epochs):
            num_batches = len(self._train_dataset_loader)

            for batch_id, batch_data in enumerate(self._train_dataset_loader):
                self._model.train()  # Set the model to train mode
                current_step = epoch * num_batches + batch_id

                ############ 2.6 TODO
                # Run the model and get the ground truth answers that you'll pass to your optimizer
                # This logic should be generic; not specific to either the Simple Baseline or CoAttention.

                input_images, questions, answers, _, _ = batch_data
                input_images = input_images.cuda()
                questions = questions.cuda()
                answers = answers.cuda()
                
                predicted_answer = self._model(input_images,questions)

                hot_idx = torch.argmax(torch.sum(answers,dim=1),dim=1)
                ground_truth_answer = torch.zeros(answers.shape[0],answers.shape[-1])
                
                for row,col in enumerate(hot_idx):
                        ground_truth_answer[row,col] = 1
                ############
                self.optimizer.zero_grad()

                self._model.WordNet.weight.data.clamp_(max=1500)
                self._model.LinearLayer.weight.data.clamp_(max=20)
                clip_grad_norm_(self._model.parameters(), 20)
                # Optimize the model according to the predictions
                ground_truth_answer = ground_truth_answer.long().cuda()
                loss = self._optimize(predicted_answer, ground_truth_answer)

                if current_step % self._log_freq == 0:
                    print("Epoch: {}, Batch {}/{} has loss {}".format(epoch, batch_id, num_batches, loss))
                    self.writer.add_scalar('train/loss', loss.item(), current_step)

                if current_step % self._test_freq == 0:
                    self._model.eval()
                    val_accuracy = self.validate()
                    print("Epoch: {} has val accuracy {}".format(epoch, val_accuracy))
                    self.writer.add_scalar('val/accuracy', val_accuracy, current_step)
        
        torch.save(self._model.state_dict(), './')
