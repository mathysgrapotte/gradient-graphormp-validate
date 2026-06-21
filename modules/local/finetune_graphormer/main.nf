process FINETUNE_GRAPHORMER {
    tag "${meta.id}"
    label 'process_medium'

    container "${params.model_container}"

    input:
    tuple val(meta), path(public_data)
    val task_type
    val finetune_mode
    val head
    val lr
    val epochs
    val ensemble
    val batch_size
    val weight_decay
    val head_dropout
    val freeze_layers
    val fuse_descriptors
    val morgan_radius
    val morgan_bits
    val seed

    output:
    tuple val(meta), path("submission.csv") , emit: submission
    tuple val(meta), path("model_meta.json"), emit: model_meta
    path "versions.yml"                      , emit: versions

    script:
    """
    finetune_graphormer.py \\
        --train ${public_data}/train.csv \\
        --test-features ${public_data}/test_features.csv \\
        --sample-submission ${public_data}/sample_submission.csv \\
        --out submission.csv \\
        --model-meta model_meta.json \\
        --task-type ${task_type} \\
        --finetune-mode ${finetune_mode} \\
        --head ${head} \\
        --lr ${lr} \\
        --epochs ${epochs} \\
        --ensemble ${ensemble} \\
        --batch-size ${batch_size} \\
        --weight-decay ${weight_decay} \\
        --head-dropout ${head_dropout} \\
        --freeze-layers ${freeze_layers} \\
        --fuse-descriptors ${fuse_descriptors} \\
        --morgan-radius ${morgan_radius} \\
        --morgan-bits ${morgan_bits} \\
        --seed ${seed}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        torch: \$(pip show torch 2>/dev/null | sed -n 's/^Version: //p')
        transformers: \$(pip show transformers 2>/dev/null | sed -n 's/^Version: //p')
        rdkit: \$(pip show rdkit 2>/dev/null | sed -n 's/^Version: //p')
    END_VERSIONS
    """
}
