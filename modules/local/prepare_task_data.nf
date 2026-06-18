process PREPARE_TASK_DATA {
    tag "${meta.id}"
    label 'process_low'

    container "${params.prepare_container}"

    input:
    tuple val(meta), path(data_dir)

    output:
    tuple val(meta), path("public")     , emit: public_data
    tuple val(meta), path("answers.csv"), emit: answers
    path "versions.yml"                 , emit: versions

    script:
    """
    prepare_task.py \\
        --data-dir ${data_dir} \\
        --public-dir public \\
        --answers answers.csv

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        pandas: \$(pip show pandas 2>/dev/null | sed -n 's/^Version: //p')
    END_VERSIONS
    """
}
