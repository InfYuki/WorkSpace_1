from torch.utils.data import Dataset
import numpy as np

class Dataset2(Dataset):
    def __init__(self, features1,features2,features3, labels):
        self.features1 = features1
        self.features2 = features2
        self.features3 = features3
        self.labels = labels

    def __len__(self):
        return len(self.features1)

    def __getitem__(self, index):
        feature1 = self.features1[index]
        feature2 = self.features2[index]
        feature3 = self.features3[index]
        label = self.labels[index]
        return feature1,feature2,feature3, label

def Model_Evaluate(true_label, predict_label, pos_label=1):

    pos_num = np.sum(true_label == pos_label)
    print('pos_num=', pos_num)
    neg_num = true_label.shape[0] - pos_num
    print('neg_num=', neg_num)
    tp = np.sum((true_label == pos_label) & (predict_label == pos_label))
    print('tp=', tp)
    tn = np.sum(true_label == predict_label) - tp
    print('tn=', tn)
    sn = tp / pos_num
    sp = tn / neg_num
    acc = (tp + tn) / (pos_num + neg_num)
    fn = pos_num - tp
    fp = neg_num - tn
    print('fn=', fn)
    print('fp=', fp)

    tp = np.array(tp, dtype=np.float64)
    tn = np.array(tn, dtype=np.float64)
    fp = np.array(fp, dtype=np.float64)
    fn = np.array(fn, dtype=np.float64)
    mcc = (tp * tn - fp * fn) / (np.sqrt((tp + fn) * (tp + fp) * (tn + fp) * (tn + fn)))
    print("Model score --- sn:{0:<20}sp:{1:<20}acc:{2:<20}mcc:{3:<20}".format(sn, sp, acc, mcc))
    return sn, sp, acc, mcc
def cal_score(label, pred):
    pred = np.array(pred)
    label = np.array(label)
    sn,sp,acc,mcc=Model_Evaluate(label,pred)

    return acc
