"""
train.py: this script performs training with the pytorch lightning package.
For details see main README.md.
"""

if __name__ == "__main__":
    from experiments.src.gin import parse_gin_config_files_and_bindings
    from experiments.src.pretraining.pretraining_pretrain_rmat import pretrain_rmat

    # Bind values to functions/methods parameters by parsing appropriate gin-config files and bindings.
    parse_gin_config_files_and_bindings(base='pretrain_rmat', dataset='contextual_pretraining')

    pretrain_rmat()
