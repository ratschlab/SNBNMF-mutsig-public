# SNBNMF - Supervised Negative Binomial NMF for Mutational Signature Discovery

## Reference
> Xinrui Lyu, Jean Garret, Gunnar Rätsch, Kjong-Van Lehmann, Mutational signature learning with supervised negative binomial non-negative matrix factorization, Bioinformatics, Volume 36, Issue Supplement_1, July 2020, Pages i154–i160, https://doi.org/10.1093/bioinformatics/btaa473

## Training

The SNBNMF model is defined `model/snbnmf.py`.
To train snbnmf model on the ICGC PCAWG dataset using default parameters:

````python snbnmf.py````


Other configurations:
- `numAtoms`: Number of atoms in the dictionary (default: 35).
- `alpha`: Dispersion parameter of negative binomial distriution (default: 1e+08).
- `epochs`: Number of epochs to train the model (default: 1e+06).
- `seed`: Random seed for initialization (default: 0).
- `convergence_epochs`: Minimum number of epochs for which the model has converged for early stopping (default: 10).
- `lambda_c`: Regularization parameter for classification loss in the objective function (default: 10).
- `lambda_w`: Regularization parameter for classifier weights in the objective function (default: 1e-4).
- `lambda_g`: Regularization parameter for axiliary classification loss in the objective function (default: 10).
- `learning_rate_lbl`: Learning rate for updating classification parameters (default: 1e-3).
- `learning_rate_constr`: Learning rate for updating auxiliary classification/regression parameters (default: 1e-3).
- `decay_steps`: Decaying step for learning rate (default: 1000).
- `decay_rate`: Decaying rate for learning rate (default: 0.96). 
- `norm_exp`: If True, normalize exposure vectors after each epoch (default: False).
- `weight_sample`: If True, adjust weights inversely proportional to class frequencies in unbalanced class (default: False).
- `constr_sig1`: If True, force signature 1 to be corresponding to auxiliary task (default: False).
- `reg4constr`: If True, use regression instead of classification for auxiliary task (default: False).
- `trimming`: If True, trim out the exposure values smaller than 0.01 after each epoch (default: False).
- `update_constr_bias`: If True, learn the bias for the auxiliary task (default: False).
