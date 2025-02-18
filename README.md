## Description 

This repository contains the code accompanying our paper in XXX. We propose a method that integrates pre-trained discrete diffusion models for protein sequences with reward models (i.e., seq → target property) at test time for computational protein design. Our algorithm effectively optimizes both the reward function and sequence naturalness characterzied by pre-trained diffusion models. Unlike existing single-shot guided approaches in diffusion models, our method uses an iterative refinement process inspired by evolutionary algorithms, alternating between (derivative-free) reward-guided generation and noising. 

<p align="center">
  <img src="medias/sum_algorithm.png" width="33%">
</p>

Below are examples of trajectories obtained when optimizing structural properties as rewards

<p align="center">
  <img src="medias/output_ss.gif" width="44%">
  <img src="medias/output_cRMSD.gif" width="44%">
</p>


#### Generated Proteins 

We present results on optimizing several fundamental structural rewards, including ``symmetry``, ``globularity``, ``match_ss``, ``crmsd``.  We can further optimize ``ptm``,``plddt``,``tm``,``lddt``,``hydrophobic``,``surface_expose``. All rewards are defined based on the outputs of a sequence-to-structure model using ESMFold. Below, we visualize examples of the generated sequences, where ESMFold is used to predict their structures.


<div style="display: flex; gap: 20px;">

  <figure style="margin: 0;">
    <img src="medias/symmetric.png" alt="Image 1" width="180"/>
    <figcaption style="text-align: center;">Symmetric proteins</figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="medias/globularity.png" alt="Image 2" width="150"/>
    <figcaption style="text-align: center;">Globularity</figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="medias/crsmd_5kph_0.8.png" alt="Image 3" width="190"/>
    <figcaption style="text-align: center;">cRMSD</figcaption>
  </figure>

  <figure style="margin: 0;">
    <img src="medias/ss_match_r15.png" alt="Image 4" width="160"/>
    <figcaption style="text-align: center;">ss-match</figcaption>
  </figure>


</div>


----

## How to run the code. 

```
CUDA_VISIBLE_DEVICES=0 python refinement.py
```
with several flags. The following is an expanation of options. 

* ``--decoding``: decoding method e.g., SVDD_edit (our proposal), SVDD (single-shot generation)
* ``--repeatnum``: batch size. This is the beam size of the tree in SVDD. 
* ``--duplicate``: important hyperparameter in decoding methods. It reflects how many states we replicate. This is the width of the tree.  
* ``--metrics_name``: reward functions e.g., ``match_ss``, ``crmsd``, ``tm``, ``globularity``, ``plddt``, ``ptm``, ``hydrophobic``, ``symmetry``
* ``--metrics_list``: how to set weights for the above rewards. e.g., ``--metric_name match_ss,crmsd,plddt --metrics_list 1,1,1`` means we optimize ``match_ss + crmsd + plddt``. 
* ``--proteinname``: In tasks such as ``match_ss``, ``crmsd``, ``tm``, we can set some target structure names in a pdb format in the folder ``/datasets`` (e.g., 5KPH, XX:run1_0367_0004, etc.) 
* ``--iteraiton``: number of iterations in our proposal 
* ``--seq_length``: length of proteins we want to design 
*  ``--edit_seqlength``: How much portion of the sequence we edit   



#### 1. Symmetricity 

Design symmetric proteins with 7 folds. 

```
CUDA_VISIBLE_DEVICES=3 python refinement.py --decoding SVDD_edit  --repeatnum 10 --duplicate 20 --seq_length 30 --metrics_name symmetry,hydrophobic,plddt --metrics_list 1,1,1 --iteration 20 --num_symmetry 7
```

#### 2. Globularity 

Design globular proteins. 

```
CUDA_VISIBLE_DEVICES=2 python refinement.py --decoding SVDD_edit  --repeatnum 10 --duplicate 20  --metrics_name globularity,plddt  --metrics_list 1,1 --iteration 20 --seq_length 150
```

#### 3. cRMSD 

Design a sequence that fold into a target structure. 

```
CUDA_VISIBLE_DEVICES=3 python refinement.py --decoding SVDD_edit  --repeatnum 20 --duplicate 20  --metrics_name crmsd  --metrics_list 1 --proteinname 5KPH --iteration 40
```

#### 4. SS (secondary structure) match

Design a sequence that fold into a target secondary structure. 

```
CUDA_VISIBLE_DEVICES=4 python refinement.py --decoding SVDD_edit  --repeatnum 10 --duplicate 20  --metrics_name match_ss  --metrics_list 1 --proteinname r15_96_TrROS_Hall --iteration 30
```

#### 5. TM score 

Design a sequence that fold into a target structure. 

```
CUDA_VISIBLE_DEVICES=5 python refinement.py --decoding SVDD_edit  --repeatnum 10 --duplicate 20  --metrics_name tm  --metrics_list 1 --proteinname 5KPH --iteration 30
```

----

### Editting Sequences 

When we want to edit existing sequences, set ``--seed_design`` as ``True``, and ``--initial_seq`` as a protein sequence. The following is an example when optimizing symmetry. 

```
CUDA_VISIBLE_DEVICES=6 python refinement.py --decoding SVDD_edit  --repeatnum 10 --duplicate 20 --metrics_name crmsd,plddt,hydrophobic --proteinname r15_96_TrROS_Hall --metrics_list 1,1,1 --iteration 20 --seed_design True --initial_seq MMELEIEIKVEGMTEEELRELAERLAAELTPEGWKVVAVRVERVDEEEGVVRVTVVVEPV  
```

### Outputs 

Refer to the notebook ```medias/evaluate.ipynb```. We save the pdb files of batches in each iteration. 

----

### Installation 

Install pytroch, pyrosseta. Then, run the following

```
conda create -n RERD python=3.9 
conda activate RERD
pip3 install torch torchvision torchaudio
pip3 install -r requirements.txt
```
-----------

### Acknolwdgements 

Our codebase is hevily based on evodiff, openfold, ESM. 

--------------------


