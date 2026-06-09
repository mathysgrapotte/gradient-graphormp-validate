//
// SCORE_SUBMISSION: grade the submission with the official Polaris grader and
// write the Stimulus-facing result.json (result.objective).
//
include { GRADE_SUBMISSION } from '../../modules/local/grade_submission'

workflow SCORE_SUBMISSION {
    take:
    ch_submission   // tuple(meta, submission.csv)
    benchmark_id
    main_metric
    objective_mode
    task_type

    main:
    ch_versions = channel.empty()

    GRADE_SUBMISSION(
        ch_submission,
        benchmark_id,
        main_metric,
        objective_mode,
        task_type,
    )
    ch_versions = ch_versions.mix(GRADE_SUBMISSION.out.versions)

    emit:
    result   = GRADE_SUBMISSION.out.result
    versions = ch_versions
}
