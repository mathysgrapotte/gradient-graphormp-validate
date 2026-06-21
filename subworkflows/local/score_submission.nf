//
// SCORE_SUBMISSION: grade the submission against the pinned private answers and
// write the Stimulus-facing result.json (result.objective).
//
include { GRADE_SUBMISSION } from '../../modules/local/grade_submission/main'

workflow SCORE_SUBMISSION {
    take:
    ch_submission   // tuple(meta, submission.csv)
    ch_answers      // tuple(meta, answers.csv)
    benchmark_id
    main_metric
    objective_mode
    task_type

    main:
    ch_versions = channel.empty()

    ch_graded = ch_submission.join(ch_answers)

    GRADE_SUBMISSION(
        ch_graded,
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
