# HDMI: High-order Deep Multiplex Infomax
This is the PyTorch implementation of the paper:

Baoyu Jing, Chanyoung Park and Hanghang Tong, 
[HDMI: High-order Deep Multiplex Infomax](https://arxiv.org/abs/2102.07810), WWW'2021 

## Requirements
- Python 3.6
- numpy>=1.19.5
- scipy>=1.5.4
- scikit-learn>=0.24.1
- tqdm>=4.59.0
- torch>=1.6.0 
- torchvision>=0.7.0

Packages can be installed via: `pip install -r requirements.txt`.
For PyTorch, please install the version compatible with your machine.


## Data
The pre-processed data can be downloaded from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0). 
Please put the pre-processed data under the folder `data`.
Each pre-processed dataset is a dictionary containing the following keys:
- `train_idx`, `val_idx` and `test_idx` are indices for training, validation and testing; 
`label` corresponds to the labels of the nodes;
- the layer names of the dataset: e.g., `MAM` and `MDM` for the `imdb` dataset.

## Run
1. Download the pre-processed data from [here](https://www.dropbox.com/s/48oe7shjq0ih151/data.tar.gz?dl=0)
   and put it to the folder `data`.
2. Specify the arguments in the `main.py`.
3. Run the code by `python main.py`.


## Citation
Please cite the following paper, if you find the repository or the paper useful.

Baoyu Jing, Chanyoung Park and Hanghang Tong, [HDMI: High-order Deep Multiplex Infomax](https://arxiv.org/abs/2102.07810), WWW'2021 

```
@article{jing2021hdmi,
  title={HDMI: High-order Deep Multiplex Infomax},
  author={Jing, Baoyu and Park, Chanyoung and Tong, Hanghang},
  journal={arXiv preprint arXiv:2102.07810},
  year={2021}
}
```
