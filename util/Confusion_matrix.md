
      |    预测
      |   0    1    
------|------------
    0 |  TN    FP
真    |
实  1 |  FN    TP

TP ： 实际为正例(1),预测为正例(1)
FN ： 实际为正例(1),预测为负例(0)
FP :  实际为负例(0),预测为正例(1)
TN ： 实际为负例(0),预测为负例(0)

Confusion_matrix = [[TN,FP],
                    [FN,TP]]
tp = np.diag(Confusion_matrix) = [TN,TP]
sum_a1 = Confusion_matrix.sum(axis=1)
       = [TN+FP, FN+TP]
sum_a0 = Confusion_matrix.sum(axis=0)
       = [TN+FN, FP+TP]
acc = tp.sum() / (hist.sum()) # Overall Acc
recall = TP / (TP+FN) = tp[1] / (sum_a1[1]) 
precision = TP / (TP+FP) = tp[1] / (sum_a0[1])
mean_acc = mean[TN/(TN+FP) , TP/(FN+TP) ] = np.nanmean(tp / (sum_a1))
FPR = FP / (FP+TN) = FP  / (sum_a1[0])
F1 = 2TP / (2TP+FP+FN) 
   = 2 * tp / (sum_a1 + sum_a0) [1]
   = (2*(recall*precision)) / (recall + precision) 

Iou = [TN/(TN+FN+FP), TP/(TP+FN+FP)] # 每个类别的IoU
    = tp / (sum_a1 + sum_a0 - tp)
Mean_IoU = np.nanmean(Iou) 求各类别IoU的平均

