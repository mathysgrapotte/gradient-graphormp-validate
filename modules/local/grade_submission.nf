process GRADE_SUBMISSION {
    tag "${meta.id}"
    label 'process_single'

    container "${params.grade_container}"

    input:
    tuple val(meta), path(submission)
    val benchmark_id
    val main_metric
    val objective_mode
    val task_type

    output:
    tuple val(meta), path("result.json"), emit: result
    path "versions.yml"                 , emit: versions

    script:
    """
    grade_submission.py \\
        --submission ${submission} \\
        --benchmark-id ${benchmark_id} \\
        --main-metric ${main_metric} \\
        --objective-mode ${objective_mode} \\
        --task-type ${task_type} \\
        --out result.json

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        polaris: \$(pip show polaris-lib 2>/dev/null | sed -n 's/^Version: //p')
    END_VERSIONS
    """
}
