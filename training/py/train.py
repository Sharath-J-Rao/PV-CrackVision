"""
Train the edge hairline-crack window classifier (logistic regression, portable to TypeScript as plain weights).

Inputs  : windows.json  (written by training/tools/extract.mts: edge-ring windows + 16 features each, using the SAME
                         production code that runs at inference, so train/inference features cannot drift)
          labels.json   ([{img, tag, x, y, label: 1 crack | 0 clean | -1 unsure, ground: 0/1}] from visual review)
Outputs : server/models/crackModel.json  and  training/loto_report.json

Evaluation is leave-one-TABLE-out (all images of a table are held out together), so reported numbers are on tables
the model never saw -- this is the guard against overfitting to the supplied images.
Usage   : python3 train.py <windows.json> <labels.json> <out_model.json> <out_report.json>
"""
import json, sys, collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

NAMES = ['lum_mean', 'lum_std', 'th_mean', 'th_top2', 'th_top5', 'fr_th8', 'fr_th14', 'fr_th22', 'n_comp', 'max_area',
         'sum_area', 'max_diag', 'n_vert', 'vert_frac', 'th_ratio', 'elong_max']
D = 210
MIN_REL = 30      # fewer windows than this in an image => per-image relative part is zero (same rule as inference)
EASY_NEG_FRAC = 0.6  # unreviewed windows below this per-image ridge-score quantile are used as easy negatives


def relative(F, groups):
    """Per-image (x - median) / IQR of every feature; makes the model exposure / zoom / soiling invariant."""
    R = np.zeros_like(F)
    for g in set(groups):
        idx = [i for i, k in enumerate(groups) if k == g]
        if len(idx) < MIN_REL:
            continue
        q = np.percentile(F[idx], [25, 50, 75], axis=0)
        R[idx] = (F[idx] - q[1]) / (q[2] - q[0] + 1e-3)
    return R


def main(wp, lp, mp, rp):
    W = json.load(open(wp)); L = json.load(open(lp))
    F = np.array([w['f'] for w in W], dtype=float)
    imgs = [w['img'] for w in W]; tabs = np.array([w['img'].split('/')[0] for w in W])
    X = np.hstack([F, relative(F, imgs)])
    lab = {(o['tag'], o['x'], o['y']): o for o in L}

    # ridge-style score used only to pick EASY negatives (same recipe that ranked the review candidates)
    Z = relative(F, imgs)
    score = Z[:, 4] + Z[:, 6] + 0.5 * Z[:, 11]
    y = np.full(len(W), -2)  # -2 = unlabeled/ignored, -1 unsure(ignored), 0 neg, 1 pos
    pos_xy = collections.defaultdict(list)
    for i, w in enumerate(W):
        o = lab.get((w['tag'], w['x'], w['y']))
        if o is not None:
            y[i] = o['label'] if o['label'] != 0 or not o['ground'] else -1
            if o['label'] in (1, -1):
                pos_xy[w['tag']].append((w['x'], w['y']))
    for g in set(imgs):
        idx = [i for i, k in enumerate(imgs) if k == g]
        cut = np.quantile(score[idx], EASY_NEG_FRAC)
        for i in idx:
            if y[i] == -2 and score[i] <= cut:
                w = W[i]
                # not overlapping any positive / unsure window (their neighbours are ambiguous)
                if all(abs(w['x'] - px) >= D or abs(w['y'] - py) >= D for px, py in pos_xy[w['tag']]):
                    y[i] = 0
    train_mask = y >= 0
    print('labelled windows:', int(train_mask.sum()), 'pos', int((y == 1).sum()), 'neg', int((y == 0).sum()))

    def fit(Xtr, ytr, C):
        sc = StandardScaler().fit(Xtr)
        m = LogisticRegression(C=C, class_weight='balanced', max_iter=5000).fit(sc.transform(Xtr), ytr)
        return sc, m

    report = {}
    best = None
    for C in [0.03, 0.1, 0.3, 1.0]:
        P = np.full(len(W), np.nan)
        for t in sorted(set(tabs)):
            tr = train_mask & (tabs != t)
            if (y[tr] == 1).sum() == 0:
                continue
            sc, m = fit(X[tr], y[tr], C)
            te = tabs == t
            P[te] = m.predict_proba(sc.transform(X[te]))[:, 1]  # ALL ring windows of the held-out table
        ev = train_mask & ~np.isnan(P)
        auc = roc_auc_score(y[ev], P[ev])
        per = {}
        for t in sorted(set(tabs)):
            e = ev & (tabs == t)
            if (y[e] == 1).sum() and (y[e] == 0).sum():
                per[t] = round(float(roc_auc_score(y[e], P[e])), 3)
        report[f'C={C}'] = dict(pooled_auc=round(float(auc), 3), per_table_auc=per)
        print(f'C={C}: pooled LOTO AUC {auc:.3f}  per-table {per}')
        if best is None or auc > best[0]:
            best = (auc, C, P.copy())
    auc, C, P = best

    # operating point from held-out predictions: per-image view (what a user sees)
    img_list = sorted(set(imgs))
    rows = []
    for thr in np.arange(0.5, 0.96, 0.05):
        tp = fn = fp_rev = 0; fa_per_img = []
        for g in img_list:
            idx = [i for i, k in enumerate(imgs) if k == g and not np.isnan(P[i])]
            hit = [i for i in idx if P[i] >= thr]
            tp += sum(1 for i in hit if y[i] == 1); fn += sum(1 for i in idx if y[i] == 1 and P[i] < thr)
            fp_rev += sum(1 for i in hit if y[i] == 0)
            fa_per_img.append(sum(1 for i in hit if y[i] != 1 and y[i] != -1))  # incl. unreviewed (upper bound)
        rec = tp / max(1, tp + fn); prec = tp / max(1, tp + sum(fa_per_img))
        rows.append(dict(thr=round(float(thr), 2), recall=round(rec, 3), precision_upper_bound_fp=round(prec, 3),
                         mean_flagged_non_crack_windows_per_image=round(float(np.mean(fa_per_img)), 2)))
        print(rows[-1])
    report['operating_points'] = rows
    # threshold: highest recall while flagging <= 12 non-crack windows per image on held-out tables. This count is a
    # pessimistic bound (unreviewed windows that really hold a crack count as false alarms) and the pipeline further
    # merges neighbours and keeps only the best few per image.
    ok = [r for r in rows if r['mean_flagged_non_crack_windows_per_image'] <= 12.0]
    thr = max(ok, key=lambda r: r['recall'])['thr'] if ok else 0.8
    print('chosen threshold', thr, 'C', C)

    sc, m = fit(X[train_mask], y[train_mask], C)
    model = dict(
        version='edge-web-lr-3', window=D, stride=105, margin=8, canonicalWidth=2200, ring=210,
        featureNames=NAMES, inputs='raw+relative', mean=sc.mean_.tolist(), scale=sc.scale_.tolist(),
        coef=m.coef_[0].tolist(), intercept=float(m.intercept_[0]), threshold=float(thr),
        trainedOn=dict(images=len(img_list), tables=sorted(set(tabs.tolist())), positives=int((y == 1).sum()),
                       negatives=int((y == 0).sum()), C=C, loto_pooled_auc=round(float(auc), 3),
                       note='Labels from one visual review of top-ranked edge windows; windows not reviewed may hold unlabelled cracks. Assistive, not authoritative.'))
    json.dump(model, open(mp, 'w'), indent=1)
    report['chosen'] = dict(C=C, threshold=float(thr), pooled_auc=round(float(auc), 3))
    json.dump(report, open(rp, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:5])
