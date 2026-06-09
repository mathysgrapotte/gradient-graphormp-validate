process TRAIN_BASELINE_MODEL {
    tag "${meta.id}"
    label 'process_low'

    container "${params.model_container}"

    input:
    tuple val(meta), path(public_data)
    val task_type
    val model_family
    val morgan_radius
    val morgan_bits
    val ridge_alpha
    val logistic_c
    val rf_n_estimators
    val hgb_learning_rate
    val seed

    output:
    tuple val(meta), path("submission.csv") , emit: submission
    tuple val(meta), path("model_meta.json"), emit: model_meta
    path "versions.yml"                      , emit: versions

    script:
    """
    train_predict.py \\
        --train ${public_data}/train.csv \\
        --test-features ${public_data}/test_features.csv \\
        --sample-submission ${public_data}/sample_submission.csv \\
        --out submission.csv \\
        --model-meta model_meta.json \\
        --task-type ${task_type} \\
        --model-family ${model_family} \\
        --morgan-radius ${morgan_radius} \\
        --morgan-bits ${morgan_bits} \\
        --ridge-alpha ${ridge_alpha} \\
        --logistic-c ${logistic_c} \\
        --rf-n-estimators ${rf_n_estimators} \\
        --hgb-learning-rate ${hgb_learning_rate} \\
        --seed ${seed}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        scikit-learn: \$(pip show scikit-learn 2>/dev/null | sed -n 's/^Version: //p')
        rdkit: \$(pip show rdkit 2>/dev/null | sed -n 's/^Version: //p')
    END_VERSIONS
    """
}
