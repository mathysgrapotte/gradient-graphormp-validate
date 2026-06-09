process PREPARE_TASK_DATA {
    tag "${meta.id}"
    label 'process_low'

    container "${params.prepare_container}"

    input:
    tuple val(meta), val(benchmark_id)

    output:
    tuple val(meta), path("public"), emit: public_data
    path "versions.yml"           , emit: versions

    script:
    """
    prepare_task.py \\
        --benchmark-id ${benchmark_id} \\
        --public-dir public

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        polaris: \$(pip show polaris-lib 2>/dev/null | sed -n 's/^Version: //p')
    END_VERSIONS
    """
}
