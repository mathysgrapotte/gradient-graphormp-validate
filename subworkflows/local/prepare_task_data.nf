//
// PREPARE_TASK_DATA: stage and validate the pinned, immutable dataset prefix.
// Emits the public training contract (train.csv, test_features.csv,
// sample_submission.csv) for the model and a separate private answers.csv that
// is wired ONLY into the grade stage.
//
include { PREPARE_TASK_DATA as PREPARE_TASK_DATA_MODULE } from '../../modules/local/prepare_task_data/main'

workflow PREPARE_TASK_DATA {
    take:
    ch_task   // tuple(meta, data_dir)

    main:
    ch_versions = channel.empty()

    PREPARE_TASK_DATA_MODULE(ch_task)
    ch_versions = ch_versions.mix(PREPARE_TASK_DATA_MODULE.out.versions)

    emit:
    public_data = PREPARE_TASK_DATA_MODULE.out.public_data
    answers     = PREPARE_TASK_DATA_MODULE.out.answers
    versions    = ch_versions
}
