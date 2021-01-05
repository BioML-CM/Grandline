# Grandline
**Gra**ph convolutional **n**eural network classification mo**d**e**l** for **i**ntegrating biological **n**etwork and gene **e**xpression (Grandline). It is a framework for predicting phenotype based on an integration of gene expression data and protein-protein interaction network. It can also identify important subnetworks that are critical for phenotype prediction in each sample.

<img width="959" alt="Grandline framework" src="https://user-images.githubusercontent.com/76929527/103628608-d9dd7480-4f71-11eb-9978-2606747865c8.png">

## How to use Grandline
```bash
git lfs clone https://github.com/BioML-CM/Grandline
conda create -n grandline python=3.7
conda activate grandline
cd grandline
pip install -r requirements.txt
nohup jupyter notebook --port 8888 &
```
Next, open `tutorial.ipynb` on your browser.

### To install git lfs on Ubuntu
Alternatively, you may download the example dataset directly from github and add to data directory.
```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```
For other operating systems: https://github.com/git-lfs/git-lfs/wiki/Installation
