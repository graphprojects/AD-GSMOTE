# AD-GSMOTE

# Adaptive Graph Enhancement for Imbalanced Multi-relation Graph Learning

Official source code for the paper "Adaptive Graph Enhancement for Imbalanced Multi-relation Graph Learning" (WSDM'25) ![Static Badge](https://img.shields.io/badge/WSDM-2025-blue)

<div align="center">
<img src="https://github.com/MTYBilly3/AD-GSMOTE/blob/main/figs/framework.png" width="1200", height="400" alt="Framework of AD-GSMOTE">
<p>Fig. 1: The overall framework of AD-GSMOTE: (a) It first leverages the degree filter to select tail (blue) nodes for minority classes (black nodes) and then generates the adaptive center node (brown node 10) based on the correctly classified nodes from the previous training process. It then designs the adaptive node generator to generate synthetic nodes (brown) for tail nodes (e.g., node 1' is generated by node 1, the center node 10, and the similar node 5.) and the triadic edge generator to create connections (i.e., $E_{\text {cen}}$, $E_{\text {tail}}$, and $E_{\text {sim}}$) for tail and synthetic nodes. Afterward, the enhanced graph under each relation $r$ will be fed to a GNN-based classifier to obtain the classification results for generating the next-round adaptive center node dynamically; (b) It obtains the fused node embeddings with different relations via a semantic attention module and further designs a class-aware adjustment module to adjust the pre-softmax logits $\mathbf{L}$ via a temperature-scaled class-aware offset during model training. All parameters would be updated via optimizing $\mathcal{L}_{\text {enh}}^{\text{logit}}$ and $\mathcal{L}_{\text {fuse}}^{\text {logit}}$.
</p>
</div>

## Getting Started
### Setup Environment 

We use conda for environment setup. Please run the following command to install the required packages.
```bash
conda create -n AD-GSMOTE python=3.11
conda activate AD-GSMOTE

pip install -r requirements.txt
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.1.2+cu121.html
```
## New Dataset

### Twitter-Drug

We introduce a new **multi-relation graph dataset**, called **Twitter-Drug**, which is adapted from our previous studies [HyGCL-DC](https://github.com/GraphResearcher/HyGCL-DC) and [HyGCL-AdT](https://github.com/graphprojects/HyGCL-AdT). 
While these prior works focused on hypergraphs, we adapt it into a multi-relation graph dataset that models real-world social media interactions through pairwise relationships. Our focus is on identifying drug-related user roles in a highly imbalanced setting where drug users represent a very small minority of the total user population.

The dataset captures both class imbalance (few positive drug users) and topology imbalance challenges. It consists of three types of pairwise relations between users:

1. User-Tweet-User (U-T-U): Connections formed through tweet interactions (replies, quotes, retweets, likes)
2. User-Follow-User (U-F-U): Connections based on following/follower relationships
3. User-Keyword-User (U-K-U): Connections between users whose tweets share common keywords

More detail about the dataset collection and annotation can be found in [README](https://github.com/graphprojects/AD-GSMOTE/blob/main/data/README.md). 


### Twitter-Series Dataset.

Here we summarize the datasets and the code for the Twitter Series Dataset.


|Dataset | Graph Type | Label Type|  Link|
|:---:|:---:|:---:|:---:|
|Twitter-Drug| Multi-relation Graph | Drug User Roles | [Link](https://github.com/MTYBilly3/AD-GSMOTE) |
|Twitter-HyDrug-Comm| Hypergraph | Drug User Roles| [Link](https://github.com/GraphResearcher/HyGCL-DC) |
|Twitter-HyDrug-Role| Hypergraph | Drug Communities| [Link](https://github.com/graphprojects/HyGCL-AdT) |



## Usage 
### Basic Usage 

```Python
cd AD-GSMOTE
python src/main.py --dataset twitter_drug --train_ratio 0.4 --load_best_params
```

We provide the best parameters for each dataset {twitter_drug, yelpchi, amazon} with train ratio {0.05, 0.4} in `config/dataset_params.yaml`. You can directly load the best parameters by setting the `--load_best_params` flag.


The overall structure of the project is as follows:
```AD-GSMOTE/
├── config/                  # Configuration files
├── data/                   # Dataset files and README
|   |-- twitter_drug/
|   |-- yelpchi/
|   |-- amazon/
|   |-- README.md
├── src/                    # Source code
|   |-- models/             # Model implementations
|   |-- utils/              # Utility functions
|   |-- main.py             # Main training script
├── requirements.txt        # Dependencies
└── README.md               # Project documentation
```


### Dataset
To download the dataset, please download the files [here] and unzip them into the `data` folder. 
The statistics of dataset employed in our paper are shown in Table 1. 


Table 1: Statistics of employed datasets.
| Dataset        | # of nodes | CIR             | TIR           | Relation | # of relation |
|----------------|------------|-----------------|---------------|----------|---------------|
|                |            |                 |               | U-T-U    | 392,190       |
| Twitter-Drug   | 27,945     | 56.3 : 4.4 : 1.2 : 1.0 | 1.0 : 37.4 : 29.1 : 21.6 | U-F-U    | 69,675        |
|                |            |                 |               | U-K-U    | 253,602       |
 |                |            |                 |               |R-U-R    | 49,315        |
| YelpChi        | 45,954     | 5.9 : 1.0       | 1.1 : 1.0     | R-S-R    | 3,402,743     |
|                |            |                 |               | R-T-R    | 573,616       |
 |                |            |                 |               |U-P-U    | 175,608       |
| Amazon         | 11,944     | 13.5 : 1.0      | 2.9 : 1.0     | U-S-U    | 3,566,479     |
|                |            |                 |               | U-V-U    | 1,036,737     |



## Logger

This is a sample running logger which records the output and the model performance for Twitter-Drug data with training split ratio 0.4:
```
Epoch: 0021, Train Loss: 2.30, Val Loss: 2.25, Train F1: 52.54%, Val F1: 58.93%, Train GMean: 34.36%, Val GMean: 40.34%, Test F1: 55.40%, Test GMean: 35.74%
Epoch: 0041, Train Loss: 1.80, Val Loss: 1.73, Train F1: 64.70%, Val F1: 66.34%, Train GMean: 52.36%, Val GMean: 52.95%, Test F1: 62.33%, Test GMean: 47.77%
Epoch: 0061, Train Loss: 1.41, Val Loss: 1.35, Train F1: 69.95%, Val F1: 69.18%, Train GMean: 59.12%, Val GMean: 57.87%, Test F1: 67.18%, Test GMean: 54.81%
Epoch: 0081, Train Loss: 1.20, Val Loss: 1.11, Train F1: 73.37%, Val F1: 71.94%, Train GMean: 63.29%, Val GMean: 63.06%, Test F1: 69.47%, Test GMean: 58.22%
Epoch: 0101, Train Loss: 1.01, Val Loss: 0.96, Train F1: 77.07%, Val F1: 73.38%, Train GMean: 68.43%, Val GMean: 64.59%, Test F1: 70.77%, Test GMean: 60.12%
Epoch: 0121, Train Loss: 0.92, Val Loss: 0.86, Train F1: 78.99%, Val F1: 74.75%, Train GMean: 70.48%, Val GMean: 66.14%, Test F1: 72.79%, Test GMean: 63.85%
Epoch: 0141, Train Loss: 0.84, Val Loss: 0.78, Train F1: 81.69%, Val F1: 73.41%, Train GMean: 75.07%, Val GMean: 63.03%, Test F1: 71.54%, Test GMean: 59.95%
Epoch: 0161, Train Loss: 0.79, Val Loss: 0.73, Train F1: 82.40%, Val F1: 71.62%, Train GMean: 75.49%, Val GMean: 60.10%, Test F1: 71.99%, Test GMean: 60.29%
Epoch: 0181, Train Loss: 0.75, Val Loss: 0.68, Train F1: 83.86%, Val F1: 73.00%, Train GMean: 77.41%, Val GMean: 63.67%, Test F1: 73.36%, Test GMean: 63.94%
Epoch: 0201, Train Loss: 0.71, Val Loss: 0.64, Train F1: 84.54%, Val F1: 72.49%, Train GMean: 78.29%, Val GMean: 62.59%, Test F1: 73.25%, Test GMean: 63.46%
Epoch: 0221, Train Loss: 0.69, Val Loss: 0.61, Train F1: 84.11%, Val F1: 72.58%, Train GMean: 78.34%, Val GMean: 61.81%, Test F1: 73.33%, Test GMean: 63.58%
Epoch: 0241, Train Loss: 0.65, Val Loss: 0.59, Train F1: 86.45%, Val F1: 72.98%, Train GMean: 80.41%, Val GMean: 62.62%, Test F1: 73.80%, Test GMean: 64.05%
Epoch: 0261, Train Loss: 0.63, Val Loss: 0.56, Train F1: 86.00%, Val F1: 73.47%, Train GMean: 80.36%, Val GMean: 64.71%, Test F1: 74.25%, Test GMean: 65.27%
Epoch: 0281, Train Loss: 0.60, Val Loss: 0.54, Train F1: 87.31%, Val F1: 71.49%, Train GMean: 81.93%, Val GMean: 60.14%, Test F1: 73.35%, Test GMean: 63.28%
Epoch: 0301, Train Loss: 0.59, Val Loss: 0.52, Train F1: 87.27%, Val F1: 72.47%, Train GMean: 81.67%, Val GMean: 63.50%, Test F1: 74.87%, Test GMean: 66.84%
Epoch: 0321, Train Loss: 0.57, Val Loss: 0.50, Train F1: 88.95%, Val F1: 73.20%, Train GMean: 84.63%, Val GMean: 63.93%, Test F1: 74.74%, Test GMean: 66.44%
Epoch: 0341, Train Loss: 0.55, Val Loss: 0.48, Train F1: 89.91%, Val F1: 73.40%, Train GMean: 85.23%, Val GMean: 63.66%, Test F1: 75.47%, Test GMean: 66.65%
Epoch: 0361, Train Loss: 0.53, Val Loss: 0.47, Train F1: 90.39%, Val F1: 72.43%, Train GMean: 86.73%, Val GMean: 61.99%, Test F1: 73.88%, Test GMean: 64.50%
Epoch: 0381, Train Loss: 0.53, Val Loss: 0.45, Train F1: 89.71%, Val F1: 72.58%, Train GMean: 85.00%, Val GMean: 63.52%, Test F1: 74.46%, Test GMean: 65.32%
Epoch: 0401, Train Loss: 0.50, Val Loss: 0.44, Train F1: 91.15%, Val F1: 71.97%, Train GMean: 86.75%, Val GMean: 62.12%, Test F1: 74.28%, Test GMean: 65.32%
Epoch: 0421, Train Loss: 0.49, Val Loss: 0.43, Train F1: 91.45%, Val F1: 71.38%, Train GMean: 88.07%, Val GMean: 61.44%, Test F1: 73.86%, Test GMean: 64.78%
Epoch: 0441, Train Loss: 0.47, Val Loss: 0.41, Train F1: 92.27%, Val F1: 72.65%, Train GMean: 88.91%, Val GMean: 63.39%, Test F1: 74.35%, Test GMean: 65.51%
Epoch: 0461, Train Loss: 0.47, Val Loss: 0.41, Train F1: 92.43%, Val F1: 72.85%, Train GMean: 89.86%, Val GMean: 60.91%, Test F1: 73.10%, Test GMean: 61.71%
Epoch: 0481, Train Loss: 0.46, Val Loss: 0.41, Train F1: 91.73%, Val F1: 74.18%, Train GMean: 87.81%, Val GMean: 66.91%, Test F1: 75.06%, Test GMean: 67.75%
Best Test F1: 75.54%, Best Test GMean: 68.92%
Total time: 37.63 seconds
```
## Contact 

Yiyue Qian - yyqian5@gmail.com

Tianyi (Billy) Ma - tma2@nd.edu

Discussions, suggestions and questions are always welcome!


## Citation 
```
@inproceedings{qian2025adgsmote,
  title={Adaptive Graph Enhancement for Imbalanced Multi-relation Graph Learning},
  author={Qian, Yiyue and Ma, Tianyi and Zhang, Chuxu and Ye, Yanfang},
  booktitle={WSDM},
  year={2025}
}
```
