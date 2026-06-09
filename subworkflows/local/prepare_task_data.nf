//
// PREPARE_TASK_DATA: download the Polaris benchmark and write the public
// training contract (train.csv, test_features.csv, sample_submission.csv).
//
include { PREPARE_TASK_DATA as PREPARE_TASK_DATA_MODULE } from '../../modules/local/prepare_task_data'

workflow PREPARE_TASK_DATA {
    take:
    ch_task   // tuple(meta, benchmark_id)

    main:
    ch_versions = channel.empty()

    PREPARE_TASK_DATA_MODULE(ch_task)
    ch_versions = ch_versions.mix(PREPARE_TASK_DATA_MODULE.out.versions)

    emit:
    public_data = PREPARE_TASK_DATA_MODULE.out.public_data
    versions    = ch_versions
}
