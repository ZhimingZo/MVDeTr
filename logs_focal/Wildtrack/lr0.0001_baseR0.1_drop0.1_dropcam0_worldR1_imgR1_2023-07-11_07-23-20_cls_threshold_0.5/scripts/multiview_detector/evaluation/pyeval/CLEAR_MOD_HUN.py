import numpy as np
from scipy.optimize import linear_sum_assignment
import math


def getDistance(x1, y1, x2, y2):
    return math.sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))


def CLEAR_MOD_HUN(gt, det):
    """
    @param gt: the ground truth result matrix
    @param det: the detection result matrix
    @return: MODA, MODP, recall, precision

    compute CLEAR Detection metrics according to
    PERFORMANCE EVALUATION PROTOCOL FOR FACE, PERSON AND
    VEHICLE DETECTION & TRACKING IN VIDEO ANALYSIS AND
    CONTENT EXTRACTION (VACE-II)
    CLEAR – CLASSIFICATION OF EVENTS, ACTIVITIES AND RELATIONSHIPS
    Submitted to Advanced Research and Development Activity

    metrics contains the following
    [1]   recall	- recall = percentage of detected targets
    [2]   precision	- precision = percentage of correctly detected targets
    [3]	MODA          - N-MODA
    [4]	MODP          - N-MODP
    """
    td = 50 / 2.5  # distance threshold
    #print(gt, gt.shape) 952 4
    # det.shape (942. 4)
    #print(gt.shape, det.shape)
    #print(gt,det)
    #exit()
    F = int(max(gt[:, 0])) + 1 # 40
    N = int(max(det[:, 1])) + 1 # 33
    Fgt = int(max(gt[:, 0])) + 1 # 40
    Ngt = int(max(gt[:, 1])) + 1 # 32

    #print(det, det.shape)
    #print(det[:, 1], det[:, 1].shape) 
    #print(F, N, Fgt, Ngt)
    #exit()
    M = np.zeros((F, Ngt))
    #print(M.shape) # 40 X 32
    #print(M)
    #exit()
    c = np.zeros((1, F))
    fp = np.zeros((1, F))
    m = np.zeros((1, F))
    g = np.zeros((1, F))

    d = np.zeros((F, Ngt))
    distances = np.inf * np.ones((F, Ngt))

   
    for t in range(1, F + 1):
        #print(t)
        GTsInFrames = np.where(gt[:, 0] == t - 1)
        DetsInFrames = np.where(det[:, 0] == t - 1)
        GTsInFrame = GTsInFrames[0]
        DetsInFrame = DetsInFrames[0]
        GTsInFrame = np.reshape(GTsInFrame, (1, GTsInFrame.shape[0]))
        DetsInFrame = np.reshape(DetsInFrame, (1, DetsInFrame.shape[0]))

        Ngtt = GTsInFrame.shape[1]
        Nt = DetsInFrame.shape[1]
        g[0, t - 1] = Ngtt

        if GTsInFrame is not None and DetsInFrame is not None:
            dist = np.inf * np.ones((Ngtt, Nt))
            for o in range(1, Ngtt + 1):
                GT = gt[GTsInFrame[0][o - 1]][2:4]
                for e in range(1, Nt + 1):
                    E = det[DetsInFrame[0][e - 1]][2:4]
                    dist[o - 1, e - 1] = getDistance(GT[0], GT[1], E[0], E[1])
            tmpai = dist
            tmpai = np.array(tmpai)

            # Please notice that the price/distance of are set to 100000 instead of np.inf, since the Hungarian Algorithm implemented in
            # sklearn will suffer from long calculation time if we use np.inf.
            tmpai[tmpai > td] = 1e6
            if not tmpai.all() == 1e6:
                HUN_res = np.array(linear_sum_assignment(tmpai)).T
                HUN_res = HUN_res[tmpai[HUN_res[:, 0], HUN_res[:, 1]] < td]
                u, v = HUN_res[HUN_res[:, 1].argsort()].T
                for mmm in range(1, len(u) + 1):
                    M[t - 1, u[mmm - 1]] = v[mmm - 1] + 1
        curdetected, = np.where(M[t - 1, :])
        
        c[0][t - 1] = curdetected.shape[0]
        for ct in curdetected:
            eid = M[t - 1, ct] - 1
            gtX = gt[GTsInFrame[0][ct], 2]

            gtY = gt[GTsInFrame[0][ct], 3]

            stX = det[DetsInFrame[0][int(eid)], 2]
            stY = det[DetsInFrame[0][int(eid)], 3]

            distances[t - 1, ct] = getDistance(gtX, gtY, stX, stY)
        
        fp[0][t - 1] = Nt - c[0][t - 1]
        m[0][t - 1] = g[0][t - 1] - c[0][t - 1]
        
    #print(np.argmax(m), np.max(m))
    #print(np.argmax(fp), np.max(fp))
    #test = fp+m 
    #print(np.sort(np.add(m, fp)))
    #print(np.argsort(np.add(m, fp))) 
    #exit()
    #print(np.argmax(np.add(m, fp)), m[:, 31], fp[:, 31])
    #[[0. 0. 0. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 2. 2. 2. 2. 2. 2. 2.
     #    2. 2. 2. 3. 3. 3. 3. 3. 3. 3. 3. 3. 4. 4. 4. 6.]]
    #[[14 38 20 21 17 18  6  7  8  4 10 11 12 13  3 27 30 26 24  0 35 28  9  5
    #     2 16 29 34 32 36 19 23 22 15  1 39 33 37 25 31]]
    #print(np.argmax(fp), np.argmax(m))
    #print(c.shape, fp.shape, m.shape, g.shape)    
    #print(np.sum(c), np.sum(fp), np.sum(m), np.sum(g))
    # c: true positive 
    # fp: false positive
    # m: false negative 
    # g: total sample 
    
    MODP = sum(1 - distances[distances < td] / td) / np.sum(c) * 100 if sum(
        1 - distances[distances < td] / td) / np.sum(c) * 100 > 0 else 0
    MODA = (1 - ((np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 if (1 - (
            (np.sum(m) + np.sum(fp)) / np.sum(g))) * 100 > 0 else 0
    recall = np.sum(c) / np.sum(g) * 100 if np.sum(c) / np.sum(g) * 100 > 0 else 0
    precision = np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 if np.sum(c) / (np.sum(fp) + np.sum(c)) * 100 > 0 else 0

    return recall, precision, MODA, MODP
