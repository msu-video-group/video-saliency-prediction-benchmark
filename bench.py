from os import path, listdir, mkdir
from tqdm import tqdm
import numpy as np
from glob import glob
import json
from multiprocessing import Pool
from itertools import chain
from ss_robust_metric2 import apply_transforms, ss_robust_metric2
import pandas as pd
from utils import read_sm, padding, padding_fixation
import argparse
import warnings
warnings.filterwarnings("error")



###metrics###

def nss(s_map, gt):
    x,y = np.where(gt > 0)
    s_map_norm = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    temp = []
    for i in zip(x,y):
        temp.append(s_map_norm[i[0], i[1]])
    return np.mean(temp)


def similarity(s_map, gt):
    s_map = s_map / (np.sum(s_map) + 1e-7)
    gt = gt / (np.sum(gt) + 1e-7)
    return np.sum(np.minimum(s_map, gt))


def cc(s_map, gt):
    a = (s_map - np.mean(s_map))/(np.std(s_map) + 1e-7)
    b = (gt - np.mean(gt))/(np.std(gt) + 1e-7)
    r = (a*b).sum() / np.sqrt((a*a).sum() * (b*b).sum() + 1e-7)
    return r


def auc_judd(S, F):

    Sth = S[F > 0]
    Nfixations = len(Sth)
    Npixels = np.prod(S.shape)

    allthreshes = sorted(Sth, reverse=True)
    tp = np.zeros(Nfixations + 2)
    fp = np.zeros(Nfixations + 2)
    tp[0] = fp[0] = 0
    tp[-1] = fp[-1] = 1

    for i in np.arange(1, Nfixations + 1):
        aboveth = np.sum(S >= allthreshes[i - 1])
        tp[i] = i / Nfixations
        fp[i] = (aboveth - i) / (Npixels - Nfixations)
    
    return np.trapz(tp, fp)


def kldiv(s_map, gt):
    s_map = s_map / (np.sum(s_map) * 1.0)
    gt = gt / (np.sum(gt) * 1.0)
    eps = 2.2204e-16
    res = np.sum(gt * np.log(eps + gt / (s_map + eps)))
    return res


######

def calculate_frame_metrics(frame):
    gt_fix = padding_fixation(read_sm(frame['gt_fixations_path']))
    gt_120_sm = padding(read_sm(frame['gt_saliency_path']))
    pred_sm = padding(read_sm(frame['predictions_path']))

    if robust_metric_res is not None:
        pred_sm = np.clip(apply_transforms(pred_sm, frame['robust_metric_res'][0], frame['robust_metric_res'][1]), 0, 1)

    return {
        'sim_score': similarity(pred_sm, gt_120_sm),
        'nss_score': nss(pred_sm, gt_fix),
        'cc_score': cc(pred_sm, gt_120_sm),
        'auc_judd_score': auc_judd(pred_sm, gt_fix),
        'kldiv_score': kldiv(pred_sm, gt_120_sm)
    }


def calculate_metrics(video_name, temp_predictions_path, temp_gt_saliency_path, temp_gt_fixations_path, robust_metric_res, num_workers):
    predictions_path = glob(temp_predictions_path)[0]
    gt_saliency_path = glob(temp_gt_saliency_path)[0]
    gt_fixations_path = glob(temp_gt_fixations_path)[0]

    scores = []
    assert_func = lambda path: set([int(x.split('.')[0]) for x in listdir(path)])
    assert assert_func(gt_fixations_path) == assert_func(gt_saliency_path) == assert_func(predictions_path)
    
    frames = [
        {
            'gt_fixations_path': gt_fix, 
            'gt_saliency_path': gt_sal, 
            'predictions_path': pred,
            'robust_metric_res': robust_metric_res
        } for gt_fix, gt_sal, pred in zip(
            [path.join(gt_fixations_path, x) for x in sorted(listdir(gt_fixations_path))], 
            [path.join(gt_saliency_path, x) for x in sorted(listdir(gt_saliency_path))], 
            [path.join(predictions_path, x) for x in sorted(listdir(predictions_path))]
    )]
    with Pool(num_workers) as pool:
        scores = pool.map(calculate_frame_metrics, frames)
    
    conv_scores = {metric: [x[metric] for x in scores] for metric in scores[0].keys()}

    return {'video_name' : video_name,
            'cc' : np.mean(conv_scores['cc_score']),
            'sim' : np.mean(conv_scores['sim_score']),
            'nss' : np.mean(conv_scores['nss_score']),
            'auc_judd' : np.mean(conv_scores['auc_judd_score']),
            'kldiv' : np.mean(conv_scores['kldiv_score'])}


def calculate_all_videos(video_names, robust_metric_res, num_workers):
    detail_result = []
    for video_name in tqdm(video_names):
        full_video_name = f'{video_name}*'
        model_output = path.join(FROM_MODEL, full_video_name)
        gt_gaussians = path.join(GT, full_video_name, 'gaussians')
        gt_fixations = path.join(GT, full_video_name, 'fixations')

        detail_result += [calculate_metrics(video_name, model_output, gt_gaussians, gt_fixations, robust_metric_res, num_workers)]
    
    return detail_result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Compares multiple models by a variety of metrics using multithreaded data processing')
    parser.add_argument('--models-root', help='path to directory with models predictions', default='savam_from_')
    parser.add_argument('--gt-root', dest='GT', help='path to directory with Ground Truth saliency maps and fixations', default='savam_GT')
    parser.add_argument('--dont-use-domain-adaptation', action='store_true', help='specifies not to use domain adaptation')
    parser.add_argument('--num-workers', type=int, help='number of used threads', default=1)
    parser.add_argument('--skip-incomplete-results-test', action='store_true', help='allowed the results of the tested model do not contain all the videos of the dataset')
    args = parser.parse_args()
    models_root = args.models_root
    GT = args.GT
    use_robust_metric = not args.dont_use_domain_adaptation
    num_workers = args.num_workers
    skip_inc_res = args.skip_incomplete_results_test
    
    print(num_workers, 'worker(s)')

    models_num = len(listdir(models_root))
    for model_num, model_root in enumerate(sorted(listdir(models_root))):
        print(f'testing {model_root} ({model_num + 1}/{models_num})')
        FROM_MODEL = path.join(models_root, model_root)

        robust_metric_res = None
        if use_robust_metric:
            def read_folder(folder_dir):
                return sorted([path.join(folder_dir, p) for p in listdir(folder_dir)])

            sm_listdir = listdir(FROM_MODEL)
            GT_listdir = listdir(GT)
            if len(sm_listdir) < len(GT_listdir):
                msg = f'There are results for only a few videos ({len(sm_listdir)}/{len(GT_listdir)})!'
                if skip_inc_res:
                    print(f'Warning: {msg}')
                else:
                    raise ValueError(msg)
            video_names = sorted(sm_listdir)

            GT_listdir_filtered = [x for x in GT_listdir if x in sm_listdir]
            sm_maps = list(map(lambda x: read_folder(path.join(FROM_MODEL, x)), sorted(sm_listdir)))
            gt_maps = list(map(lambda x: read_folder(path.join(GT, x, 'gaussians')), sorted(GT_listdir_filtered)))
            cp_img = padding(read_sm('gt_cp.png'))
            robust_metric_res = cp_img, ss_robust_metric2(sm_maps, gt_maps, cp_img, nbins=256, num_workers=num_workers)

        detail_result = calculate_all_videos(video_names, robust_metric_res, num_workers)

        result_name = 'Robust_Result' if use_robust_metric else 'Result'
        if not path.exists(result_name):
            mkdir(result_name)
        json_root = path.join(result_name, model_root)
        detail_result = sorted(detail_result, key=lambda res: res['video_name'])
        with open(f'{json_root}.json', mode='w') as output:
            output.write(json.dumps(detail_result))

        result = {'cc' : [], 'sim' : [], 'nss' : [], 'auc_judd' : [], 'kldiv' : []}
        for i in result:
            for j in detail_result:
                result[i].append(j[i])

        model_res = {'model': model_root}
        [model_res.update({key: [np.mean(result[key])]}) for key in result.keys()]
        header = not path.exists(f'{result_name}.csv')
        pd.DataFrame.from_dict(model_res, orient='columns').to_csv(f'{result_name}.csv', mode='a', header=header, index=False)
        print(model_res)
