import torch
import torch.nn as nn

def accuracy_calculate(output,prediction):
	with torch.no_grad():
		preds = torch.argmax(prediction,dim=1)
		total = torch.sum(preds==output)
		return total