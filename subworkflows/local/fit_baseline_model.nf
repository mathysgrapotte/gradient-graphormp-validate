//
// FIT_BASELINE_MODEL: featurize molecules, fit the baseline estimator, and
// emit predictions for the test split in the sample-submission schema.
//
include { TRAIN_BASELINE_MODEL } from '../../modules/local/train_baseline_model'

workflow FIT_BASELINE_MODEL {
    take:
    ch_public          // tuple(meta, public_dir)
    task_type
    model_family
    morgan_radius
    morgan_bits
    ridge_alpha
    logistic_c
    rf_n_estimators
    hgb_learning_rate
    seed

    main:
    ch_versions = channel.empty()

    TRAIN_BASELINE_MODEL(
        ch_public,
        task_type,
        model_family,
        morgan_radius,
        morgan_bits,
        ridge_alpha,
        logistic_c,
        rf_n_estimators,
        hgb_learning_rate,
        seed,
    )
    ch_versions = ch_versions.mix(TRAIN_BASELINE_MODEL.out.versions)

    emit:
    submission = TRAIN_BASELINE_MODEL.out.submission
    model_meta = TRAIN_BASELINE_MODEL.out.model_meta
    versions   = ch_versions
}
