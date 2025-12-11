from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
from scipy.ndimage import gaussian_filter
import numpy as np
import torch.nn.functional as F
import random


def setup_seed(seed): 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size, out_size]) 
    else:
        anomaly_map = np.zeros([out_size, out_size])
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft) 
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True) 
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map 
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list
def evaluation(encoder, decoder, res, dataloader, device, score_num):
    decoder.eval()
    gt_list_sp = []  
    pr_list_sp = []  

    with torch.no_grad():
        for img, label, _, img_path in dataloader:
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(inputs[3], inputs[0:3], res)

            anomaly_map, a_map_list = cal_anomaly_map(inputs[0:3], outputs, img.shape[-1], amap_mode='a')  

            # smooth
            for i in range(len(a_map_list)):
                a_map_list[i] = gaussian_filter(a_map_list[i], sigma=4)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            # store scores
            gt_list_sp.append(label.numpy()[0])
            pre_map = np.flipud(np.sort(anomaly_map.flatten()))
            pre = np.mean(pre_map[:score_num])
            pr_list_sp.append(pre)

    # --- Compute AUROC, AUPR ---
    auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 3)
    aupr_sp  = round(average_precision_score(gt_list_sp, pr_list_sp), 3)

    # ---------- Compute F1-score ----------
    thresholds = np.linspace(min(pr_list_sp), max(pr_list_sp), 100)
    best_f1 = 0
    best_thresh = 0

    for t in thresholds:
        preds = [1 if s >= t else 0 for s in pr_list_sp]
        f1 = f1_score(gt_list_sp, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    f1_sp = round(best_f1, 3)
    print("Best F1 Threshold:", best_thresh)

    return auroc_sp, aupr_sp, f1_sp



