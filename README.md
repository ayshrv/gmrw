# Self-Supervised Any-Point Tracking by Contrastive Random Walks

We present a simple, self-supervised approach to the Tracking Any Point (TAP) problem. We train a global matching transformer to find cycle consistent tracks through video via contrastive random walks, using the transformer’s attention-based global matching to define the transition matrices for a random walk on a space-time graph.

![](assets/teaser.gif)

[Self-Supervised Any-Point Tracking by Contrastive Random Walks](https://arxiv.org/pdf/2409.16288), ECCV 2024

Ayush Shrivastava and Andrew Owens



## Setup

```
git clone https://github.com/ayshrv/gmrw/
```

### Environment
```
conda create -y -n gmrw python=3.9
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c conda-forge tensorflow-datasets==4.8.3 opencv
python -m pip install kornia==0.6.3 scikit-image==0.18.3
```

### Datasets

#### Tap-Vid-Kubric
Download Tap-Vid-Kubric from [here](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid) and place it in `datasets/movi/movi_e`.

#### Kinetics-700-2020
Download and extract the Kinetics dataset using `scripts/kinetics_download.sh` and `scripts/kinetics_extract.sh`. You can also refer to the [kinetics-dataset repo](https://github.com/cvdfoundation/kinetics-dataset) for detailed instructions. Place the dataset in `datasets/kinetics700-2020`.

## Training

We train models on Kubric and Kinetics datasets.

To train on Kubric dataset, run
```
bash experiments/train_kubric.sh
```

To train on Kinetics dataset, run
```
bash experiments/train_kinetics.sh
```
Then, for finetuning with smoothness loss, run
```
bash experiments/train_kinetics_finetune.sh
```

## Evaluations
Pretrained models on Kubric and Kinetics can be downloaded from [this link](https://www.dropbox.com/scl/fo/2uutl18md0c83et2aw0oz/ADkl-RAW0fneflvSKjd3MB0?rlkey=fk35z55x95t9mu6i7afbqkzp6&st=ara9ej5f&dl=0). Put them in `results/`

To evaluate the model trained with Kubric dataset on Tap-Vid-DAVIS.
```
bash experiments/eval_kubric_trained_davis.sh
```
To evaluate using upsampled images on Tap-Vid-DAVIS. Use this to replicate the best numbers in the paper.
```
bash experiments/eval_kubric_trained_davis_upsampled.sh
```

- To evalute on Tap-Vid-Kubric, use [eval_kubric_trained_kubric.sh](experiments/eval_kubric_trained_kubric.sh) and [eval_kubric_trained_kubric_upsampled.sh](experiments/eval_kubric_trained_kubric_upsampled.sh)
- To evaluate the models trained on Kinetics, use [eval_kinetics_trained_davis.sh](experiments/eval_kinetics_trained_davis.sh) and [eval_kinetics_trained_kubric.sh](experiments/eval_kinetics_trained_kubric.sh)
- In the eval script [test_on_tapvid.py](test_on_tapvid.py), change `--eval_dataset` to `rgb_stacking` and `kinetics` to evalute on Tap-Vid-Kinetics and Tap-Vid-RGBStacking.
- In [test_on_tapvid.py](test_on_tapvid.py), change `setting` to `forward_backward_time_independent_query_point` to run for direct setting in the paper.

# Citation

If you find our work useful, please cite:
```
@InProceedings{shrivastava2024gmrw,
      title     = {Self-Supervised Any-Point Tracking by Contrastive Random Walks},
      author    = {Shrivastava, Ayush and Owens, Andrew},
      journal   = {European Conference on Computer Vision (ECCV)},
      year      = {2024},
      url       = {https://arxiv.org/abs/2409.16288},
}
```
