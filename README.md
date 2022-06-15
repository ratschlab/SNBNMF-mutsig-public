# SNBNMF-mutsig-public
Supervised Negative Binomial NMF for Mutational Signature Discovery


# SNBNMF - Supervised Negative Binomial NMF for Mutational Signature Discovery

## Reference
> Xinrui Lyu, Jean Garret, Gunnar Rätsch, Kjong-Van Lehmann, Mutational signature learning with supervised negative binomial non-negative matrix factorization, Bioinformatics, Volume 36, Issue Supplement_1, July 2020, Pages i154–i160, https://doi.org/10.1093/bioinformatics/btaa473

## Training and Evaluation

### Deep Probabilistic SOM

The SNBNMF model is defined `model/snbnmf.py`.
To train snbnmf model on the ICGC PCAWG dataset using default parameters:

````python snbnmf.py````


Other configurations:
- `numAtoms`: Number of atoms in the dictionary
- `alpha`: Dispersion parameter of negative binomial distriution.
- `epochs`: Number of epochs to train the model.
- `seed`: Random seed for initialization.
- `convergence_epochs`: Minimum number of epochs for which the model has converged for early stopping.
- `lambda_c`: Regularization parameter for classification loss in the objective function.
- `lambda_w`: Regularization parameter for classifier weights in the objective function.
- `lambda_g`: Regularization parameter for axiliary classification loss in the objective function.
- `learning_rate_lbl`: Learning rate for updating classification parameters.
- `learning_rate_constr`: Learning rate for updating negative-binomial negative matrix factorization parameters.
