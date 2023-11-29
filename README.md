# ADBCR: Adversarial Distribution Balancing for Counterfactual Reasoning

This repository implements ADBCR (Schrod et al. 2023 [[1]](https://arxiv.org/pdf/2311.16616.pdf)). ADBCR is a counterfactual regression learning algorithm optimized for observational treatment data associated with continuous outcomes.
ADBCR extends the Counterfactual Regression Framework by Shalit et al. 2016 [[2]](https://arxiv.org/pdf/1605.03661.pdf) with an adaptation of the adversarial learning proceedure by Saito et al. 2018 [[3]](https://openaccess.thecvf.com/content_cvpr_2018/papers/Saito_Maximum_Classifier_Discrepancy_CVPR_2018_paper.pdf), i.e. treatment bias is regularized using an adversary consisting of two supporting outcome heads for each treatment.

## Get started
You can install ADBCR using pip
```sh
  git clone https://github.com/sschrod/ADBCR.git
  cd ADBCR
  pip install -r requirements.txt
  pip install .
```

Alternatively, we provide a `DOCKERFILE` to build a Docker container (uses PyTorch 1.13.0 with CUDA 11.7)
```shell
docker build -t adbcr -f Dockerfile_ADBCR .
```

## CODE

ADBCR uses [ray[tune]](https://docs.ray.io/en/latest/tune/index.html) to parallelize the hyper-parameter search, all parameters are predefined in the `config` dictionary where you can also set the number of CPUs and GPUs (set 0 for cpu training only) used for each trial.
For each trial the model with minimal validation loss is selected and saved in `./saved_models/`.
To reproduce the results from Schrod et al. 2023[[1]](ADDLINK) for ADBCR, UADBCR and DANNCR run [IHDP_main.py](IHDP_main.py) and [NEWS_main.py](NEWS_main.py) using the respective model flag `-- model ADBCR`, `-- model UADBCR`, `-- model DANNCR`
The optimized models for each data representation are evaluated using [IHDP_analyze.py](IHDP_analyze.py) or [NEWS_analyze.py](NEWS_analyze.py) for each model.

```shell
#Example for ADBCR on IHDP
docker run --gpus all -it --rm -v ADBCR:/mnt ssc_adbcr python3 /mnt/IHDP_main.py --method ADBCR --num_rep 100
docker run --gpus all -it --rm -v ADBCR:/mnt ssc_adbcr python3 /mnt/IHDP_analyze.py --method ADBCR --num_rep 100
```



## References
[1] Schrod, Stefan, et al. "Adversarial Distribution Balancing for Counterfactual Reasoning" arXiv:2311.16616. 2023

[2] Johansson, Fredrik, et al. "Learning representations for counterfactual inference." International conference on machine learning. PMLR, 2016.

[3] Saito, Kuniaki, et al. "Maximum classifier discrepancy for unsupervised domain adaptation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018



