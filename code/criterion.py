import torch.nn as nn



def criterion1(preds, targets):
    return nn.CrossEntropyLoss(label_smoothing=0.05)(preds,targets)


def criterion2(preds, labels):
    p1 = 0
    p2 = 0
    mrr = 0
    for i in range(len(preds)):
        j = sorted(list(preds[i]), reverse = True)
        for index,label in enumerate(labels[i]):
            if label == 1:
                if preds[i,index] == j[0]:
                    p1+=1
                    p2+=1
                    mrr += 1 
                    break
                elif preds[i,index] == j[1]:
                    p2+=1
                    mrr += 1/2
                    break
                elif preds[i,index] == j[2]:
                    mrr += 1/3
                elif preds[i,index] == j[3]:
                    mrr += 1/4
    
    return p1 / len(preds), p2 / len(preds), mrr / len(preds)



