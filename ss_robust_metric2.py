import numpy as np
import cvxopt
from pprint import pprint
from tqdm import tqdm
from multiprocessing import Pool
from os import listdir
from os.path import join
from utils import read_sm, im2float, padding


def get_sum_under_slices(img, bin_map, nbins):
    res = np.bincount(bin_map.flatten(), weights=img.flatten(), minlength=nbins)
    assert res.size == nbins
    return res

def discretize(data, nbins):
    out_bins = np.around(im2float(data) * (nbins - 1)).astype(np.int32)
    assert(np.min(out_bins) >= 0 and np.max(out_bins) < nbins)
    return out_bins

# Applies computed transform to sm. Returns result in [0; 1] range
def apply_transforms(sm, cp_img, transform_params):
    sm_bins = discretize(sm, transform_params['nbins'])
    map_func = np.array(transform_params['map'])
    res = map_func[sm_bins.flatten()].reshape(sm.shape) + transform_params['beta'] * cp_img
    return res

def check_sm_from_gen(sm):
    assert sm.dtype in (np.float32, np.float64)
    assert sm.max() <= 1
    return sm

def unroll_videos(sm_videos):
    res = []
    for video_desc in sm_videos:
        if isinstance(video_desc, list): # list of paths
            for path in video_desc:
                res += [lambda path=path: padding(read_sm(path))]
        else:
            raise TypeError('Unknown instance {}'.format(video_desc))
    return res


def job_func(batch):
    global g_nbins
    global sm_getters
    global gt_getters
    global g_cp_img

    dg = np.zeros(g_nbins)
    bc = np.zeros(g_nbins)
    sum_gt_per_bin = np.zeros(g_nbins)
    sum_gt_weighted = 0
    sum_gt_sqr = 0
    idxs = tqdm(batch[0]) if batch[1] else batch[0]
    for idx in idxs:
        frame_sm = sm_getters[idx]()
        frame_gt = gt_getters[idx]()

        sum_gt_weighted += np.sum(frame_gt * g_cp_img)
        sum_gt_sqr += np.sum(frame_gt ** 2)

        frame_sm_bins = discretize(frame_sm, g_nbins)
        dg += np.bincount(frame_sm_bins.flatten(), minlength=g_nbins)
        bc += get_sum_under_slices(g_cp_img, frame_sm_bins, g_nbins)
        sum_gt_per_bin += get_sum_under_slices(frame_gt, frame_sm_bins, g_nbins)

    return dg, bc, sum_gt_per_bin, sum_gt_weighted, sum_gt_sqr

def ss_robust_metric2(sm_maps, gt_maps, cp_img, nbins=256, num_workers=20):
    assert(nbins >= 2)

    global g_nbins
    g_nbins = nbins
    global sm_getters
    global gt_getters
    global g_cp_img
    g_cp_img = cp_img
    
    sm_getters = unroll_videos(sm_maps)
    gt_getters = unroll_videos(gt_maps)
    assert len(sm_getters) == len(gt_getters)
    frames_to_process = len(sm_getters)
            
    batch_size = frames_to_process // num_workers
    borders = [[batch_size * i, batch_size * (i + 1)] for i in range(num_workers)]
    borders[-1][1] = frames_to_process
    splitted_idxs = [[np.arange(l, r), False] for [l, r] in borders]
    splitted_idxs[0][1] = True

    with Pool(num_workers) as pool:
        job_result = pool.map(job_func, splitted_idxs)

    dg = np.array([x[0] for x in job_result]).sum(axis=0)
    bc = np.array([x[1] for x in job_result]).sum(axis=0)
    sum_gt_per_bin = np.array([x[2] for x in job_result]).sum(axis=0)
    sum_gt_weighted = sum([x[3] for x in job_result])
    sum_gt_sqr = sum([x[4] for x in job_result])

    sum_cp_sqr = frames_to_process * np.sum(cp_img.flatten() ** 2)
    
    V = np.concatenate([[sum_cp_sqr], dg, bc, bc])
    I = np.concatenate([np.arange(nbins+1), np.arange(1, nbins+1), np.zeros(nbins, dtype=np.int64)])
    J = np.concatenate([np.arange(nbins+1), np.zeros(nbins, dtype=np.int64), np.arange(1, nbins+1)])
    H_sparse = cvxopt.spmatrix(V, I, J, size=(nbins+1, nbins+1))
    
    f = -np.concatenate([[sum_gt_weighted], sum_gt_per_bin])

    V = np.concatenate([np.ones(nbins-1), -np.ones(nbins-1), [1, -1]])
    I = np.concatenate([np.arange(0, nbins-1), np.arange(0, nbins-1), [nbins-1, nbins]])
    J = np.concatenate([np.arange(1, nbins), np.arange(2, nbins+1), [nbins, 1]])
    A_sparse = cvxopt.spmatrix(V, I, J, size=(nbins+1, nbins+1))
    b = np.concatenate([np.zeros(nbins-1), [1, 0.5]])
    
    solution_res = cvxopt.solvers.qp(H_sparse, 
                                     cvxopt.matrix(f), 
                                     G=A_sparse, 
                                     h=cvxopt.matrix(b))
    x = np.array(solution_res['x']).flatten()

    return dict(beta=x[0],
                map=x[1:],
                c0=sum_gt_sqr,
                nbins=nbins)


def main():
    np.set_printoptions(threshold=200, linewidth=160)
    
    folder_sm = 'savam_from_vinet_dave'
    folder_gt = 'savam_GT'
    path_cp_img = 'gt_cp.png'
    
    def read_folder(folder_dir):
        return sorted([join(folder_dir, p) for p in listdir(folder_dir)])

    sm_maps = list(map(lambda x: read_folder(join(folder_sm, x)), sorted(listdir(folder_sm))))
    gt_maps = list(map(lambda x: read_folder(join(folder_gt, x, 'gaussians')), sorted(listdir(folder_gt))))
    cp_img = padding(read_sm(path_cp_img))
    
    res = ss_robust_metric2(sm_maps, gt_maps, cp_img, nbins=256)
    pprint(res)


if __name__ == '__main__':
    main()
