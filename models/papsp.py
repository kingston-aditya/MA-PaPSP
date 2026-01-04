from utils.utils import arrange_data, retrieve_dat, score
from main import reject_algos, reject_base
from utils.run_llm import phi3_gen
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
from scipy.special import softmax, expit
import argparse
import json

def make_dict(txt):
    a = {"1":[txt]}
    return a

def new_classify(scores, pred, gt, lth, th):
    fin_sc = scores
    rsk = 0
    cov = 0
    scorer = Rouge()
    for i in range(len(fin_sc)):
        if fin_sc[i] >= th:
            cov += 1
        else:
            cov += 0
                
        if scorer.compute_score(make_dict(pred[i]),make_dict(gt[i]))>lth and fin_sc[i]>=th:
            rsk += 1
        else:
            rsk += 0

    if cov ==0:
        return 0, 0
    else:
        return cov/len(fin_sc), rsk/cov

def sec_classify(scores, gt_scrs):
    ind_sc = np.argsort(scores)[::-1]
    srt_sc = scores[ind_sc]
    total_samples = srt_sc.shape[0]

    cov = []
    rsk = []
    for i in range(total_samples):
        if i%500==0:
            print("sec class",i)
        covrd = np.where(scores >= srt_sc[i])[0]
        cov.append(len(covrd)/total_samples)
        temp_scrs = gt_scrs[covrd]
        if len(covrd) > 0:
            rsk.append((temp_scrs[temp_scrs < 0.74].shape[0])/len(covrd))
        else:
            rsk.append(0)

    return cov, rsk, srt_sc


class clip_classify(object):
    def __init__(self, Xr, Yr, N):
        super(clip_classify,self).__init__()
        self.Xr = Xr
        self.Yr = Yr
        self.N = N

    def algo_reject_score(self, gamma, alpha, alg, k_r, Xq, Yq, Ytq, Hn):
        scores = []
        # k_r = np.load("/data/aditya/coco_embeds/coco_retrievals_txt_img.npy")

        # get the scores
        for i in range(Xq.shape[0]):
            # scores between image and label
            if alg == 0:
                print(i,"base 1")
                fin_sc = reject_base().base(Xq[i,:],Yq[i,:])
            elif alg == -1:
                print(i,"base 2")
                fin_sc = reject_base().base(Yq[i,:],Ytq[i,:])
                print(fin_sc)
            elif alg == 1:
                print(i, "algo 1")
                temp = np.concatenate((Yq[i,:].reshape(1,512),Hn[10*i:10*(i+1),:]), axis=0)
                c = softmax(Xq[i,:]@temp.T) # 1x11
                fin_sc = c[0]
            elif alg == 2:
                print(i, "algo 2")
                a = []
                for i in range(label.shape[0]):
                    k_r = ret_obj.retrieve_Y(label[i,:])[0]
                    a.append(k_r)
                k_r = np.asarray(a).reshape(11,self.N)
                # accept or reject
                fin_sc = reject_algos(self.Xr, self.Yr).algorithm2(Xq[i,:], k_r, logits)[0][0]
            elif alg == 3:
                print(i, "algo 3")
                a = []
                for i in range(label.shape[0]):
                    k_r = ret_obj.retrieve_X(label[i,:])[0]
                    a.append(k_r)
                k_r = np.asarray(a).reshape(11,self.N)
                # accept or reject
                fin_sc = reject_algos(self.Xr, self.Yr).algorithm3(Xq[i,:], k_r, logits)[0][0]
            elif alg == 4:
                print(i, "algo 4")
                # k = 15 for best performance
                e_r = k_r[i,:].tolist()[:self.N]
                l_r = []
                # accept or reject
                fin_sc = reject_algos(self.Xr, self.Yr).algorithm4(Xq[i,:], Yq[i,:], e_r, l_r, alpha, gamma)
            elif alg == 5:
                print(i, "algo 5")
                e_r = k_r[i,:].tolist()[:self.N]
                l_r = p_r[i,:].tolist()[:self.N]
                # accept or reject
                fin_sc = reject_algos(self.Xr, self.Yr).algorithm5(Xq[i,:], Yq[i,:], e_r, l_r, alpha, gamma)
            elif alg == 6:
                print(i, "algo 6")
                e_r = k_r[i,:].tolist()[:self.N]
                l_r = p_r[i,:].tolist()[:self.N]
                # accept or reject
                fin_sc = reject_algos(self.Xr, self.Yr).algorithm5(Xq[i,:], Yq[i,:], e_r, l_r, alpha, gamma)
            elif alg == 71:
                print(i, "algo 7a")
                e_r = p_r[i,:].tolist()[:self.N]
                scf = 0
                tot = 0
                for i in range(len(e_r)):
                    scf += (gamma**i)*score(Yr_sent[e_r[i],:],Yq_sent[i,:])
                    tot += gamma**i
                fin_sc = scf/tot
            elif alg == 72:
                print(i, "algo 7b")
                e_r = k_r[i,:].tolist()[:self.N]
                scf = 0
                tot = 0
                for i in range(len(e_r)):
                    scf += (gamma**i)*score(Yr_sent[e_r[i],:],Yq_sent[i,:])
                    tot += gamma**i
                fin_sc = scf/tot
            elif alg == 73:
                print(i, "algo 7c")
                e_r = k_r[i,:].tolist()[:self.N]
                scf = []
                for j in range(len(e_r)):
                    scf.append(score(Yr_sent[e_r[j],:],Yq_sent[i,:]))
                fin_sc = np.median(scf)
            elif alg == 81:
                print(i, "algo 8a")
                e_r = k_r[i,:].tolist()[:self.N]
                scr = 0
                tot = 0
                for j in range(len(e_r)):
                    scr += (gamma**j)*score(self.Xr[e_r[j]],Xq[i,:])
                    tot += (gamma**j)
                fin_sc = scr/tot
            elif alg == 82:
                print(i, "algo 8b")
                e_r = p_r[i,:].tolist()[:self.N]
                scr = 0
                tot = 0
                for j in range(len(e_r)):
                    scr += (gamma**j)*score(self.Yr[e_r[j]],Xq[i,:])
                    tot += (gamma**j)
                fin_sc = scr/tot
            else:
                print("Wrong algorithm")
            scores.append(fin_sc)

        return np.asarray(scores).reshape(-1,)

# for coverage threshold plot
def main(gamma, alg, n, alpha, r, thresh = 0):
    # load retrieval and test data
    Xr,Yr,Xq,Yq,Ytq,Hn = arrange_data("nocaps",r)
    print("loaded data")

    # do retrievals
    N = n 
    ret_obj = retrieve_dat(Xr, Yr, N)
    k_r=[]
    for i in range(Xq.shape[0]):
        k_r.append(ret_obj.retrieve_Y(Xq[i,:]))
    k_r = np.asarray(k_r).squeeze()
    print("retrieval done")

    # get scores
    classifier = clip_classify(Xr, Yr, N)

    ## baselines 
    Xq_blip = np.load("/data/aditya/blip_img_emd.npy")
    Yq_blip = np.load("/data/aditya/blip_txt_emd.npy")
    # base1_scrs = classifier.algo_reject_score(gamma, alpha, 0, ret_obj, Xq, Yq, Ytq, Hn)
    # base2_scrs = classifier.algo_reject_score(gamma, alpha, -1, ret_obj, Xq, Yq, Ytq, Hn)
    ## algorithms
    # algo1_scrs = classifier.algo_reject_score(gamma, alpha, 1, ret_obj, Xq, Yq, Ytq, Hn)
    # base2_scrs = base1_scrs

    # Cider scores - 0.06
    # f = open("/data/aditya/IC-rejection/ic_scores/cider-python3/results_out.json")
    # obj = json.load(f)
    # gt_scrs = np.asarray(obj["CIDEr"])/10
    gt_scrs = np.asarray(np.load("/data/aditya/nocaps_embeds/CIDEr_nocaps_scrs.npy"))

    # Spice scores - 0.17
    # gt_scrs = np.asarray(np.load("/data/aditya/coco_embeds/spice_coco_scrs.npy"))

    #Meteor scores - 0.23
    # gt_scrs = np.asarray(np.load("/data/aditya/coco_embeds/meteor_coco_scrs.npy"))

    aurc = 0
    T = 5
    for i in range(T):
        if alg ==2:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 2, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 3:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 3, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 4:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 4, k_r, Xq, Yq, Ytq, Hn)
        elif alg == 5:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 5, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 6:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 6, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 71:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 71, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 72:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 72, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 73:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 73, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 81:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 81, ret_obj, Xq, Yq, Ytq, Hn)
        elif alg == 82:
            algon_scrs = classifier.algo_reject_score(gamma, alpha, 82, ret_obj, Xq, Yq, Ytq, Hn)
        else:
            print("i dont know wtd")

        ## baseline thresholding
        # cov1, rsk1, _ = sec_classify(np.asarray(base1_scrs),gt_scrs)
        # cov2, rsk2, _ = sec_classify(np.asarray(base2_scrs),gt_scrs)
        # cov4, rsk4, _ = sec_classify(np.asarray(algo1_scrs),gt_scrs)
        ## algorithm thresholding
        cov3, rsk3, _ = sec_classify(np.asarray(algon_scrs),gt_scrs)
        # cov4, rsk4, _ = sec_classify(np.asarray(algon_scrs),pred,gt)

        aurc += round(auc(cov3,rsk3),3)

    # print("base", round(auc(cov1,rsk1),3))
    #make plots for coverage vs risk
    plt.figure(figsize=(12,10))
    plt.rcParams['font.size'] = 14
    # plt.plot(cov1, rsk1, linewidth=3.5)
    # plt.plot(cov2, rsk2, linewidth=3.5)
    plt.plot(cov3, rsk3, linewidth=3.5)
    # plt.plot(cov4, rsk4)
    plt.grid()
    plt.xlabel("Coverage", fontsize=25)
    plt.ylabel("Risk", fontsize=25)
    plt.legend(["CLIP","JANe"], fontsize=25)
    plt.xticks(np.linspace(0,1,11), fontsize=20)
    plt.yticks(np.linspace(0,1,11), fontsize=20)
    # plt.ylim([0, 0.5])
    # tit = str(round(auc(cov1,rsk1),3)) + "; " + str(round(auc(cov2,rsk2),3)) + "; " + str(round(auc(cov3,rsk3),3)) + "; " + str(round(auc(cov4,rsk4),3)) + "; N="+str(N)+" "+str(gamma)
    tit = str(round(aurc/T,3)) 
    plt.title(tit)
    nme = "/data/aditya/IC-rejection/cov-rsk-plot-"+str(r)+" "+str(alg)+" "+str(gamma)+" "+str(alpha)+".pdf"
    plt.savefig(nme, format="pdf", bbox_inches="tight")
    print("Title", tit)

    # return np.asarray(base1_scrs), np.asarray(base1_scrs), gt_scrs, np.asarray(algon_scrs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description ='Process some integers.')
    parser.add_argument('gamma', type = float, help ='insert gamma')
    parser.add_argument('alg', type = float, help ='insert algorithm number')
    parser.add_argument('n', type = int, help ='insert k retrievals')
    parser.add_argument('alpha', type = float, help ='insert alpha')
    parser.add_argument('r', type = int, help ='insert retrieval percentage')
    args = parser.parse_args()

    gamma = args.gamma
    n = args.n
    alg = args.alg
    alpha = args.alpha
    r = args.r
    main(gamma, alg, n, alpha, r, thresh=1)