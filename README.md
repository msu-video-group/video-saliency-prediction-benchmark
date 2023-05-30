# Video Saliency Prediction Benchmark
### Explore the best methods of video saliency prediction (VSP) algorithms
This repository provides the code for work:
[Video Saliency Prediction Benchmark](https://videoprocessing.ai/benchmarks/video-saliency-prediction.html)<br>
(Will be moved to [MSU Video Group](https://github.com/msu-video-group))

To install the dependencies into your conda environment with `python 3.8`, run:
```bash
pip install -r requirements.txt
```

To execute benchmark calculations run:
```bash
python bench.py --models_root <path to directory with models predictions>
                --gt_root <path to directory with Ground Truth saliency maps and fixations>
                --dont_use_domain_adaptation <specifies not to use domain adaptation>
                --num_workers <number of used threads>
