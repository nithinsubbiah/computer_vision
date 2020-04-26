import torch.nn as nn
import torch
from external.googlenet.googlenet import googlenet

class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """
    def __init__(self):
        super().__init__()
	
        self.ImageNet = googlenet(pretrained=True)
        self.WordNet = nn.Linear(in_features=5747,out_features=1000)
        self.LinearLayer = nn.Linear(in_features=2000,out_features=5217)
        self.activation = nn.Softmax(dim=1)

    def forward(self, image, question_encoding):
	    
        image_embedding = self.ImageNet(image)
        if len(image_embedding) > 1:
            image_embedding = image_embedding[-1]

        question_encoding = question_encoding.float()
        word_embedding = self.WordNet(question_encoding)
        
        feature_embedding = torch.cat([image_embedding,word_embedding],dim=1) 
        feature_embedding = self.LinearLayer(feature_embedding)
        output = self.activation(feature_embedding)

        return output



        
