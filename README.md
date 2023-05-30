# Video Saliency Prediction Benchmark
Explore the best methods of video saliency prediction (VSP) algorithms

To install the dependencies into your conda environment with `python 3.8`, run:
```
pip install -r requirements.txt
```

To execute benchmark calculations run:
```
python bench.py --models_root <path to directory with models predictions>
                --gt_root <path to directory with Ground Truth saliency maps and fixations>
                --dont_use_domain_adaptation <specifies don't use domain adaptation>
                --num_workers <number of used threads>