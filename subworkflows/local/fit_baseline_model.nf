//
// FIT_BASELINE_MODEL: featurize molecules, fit the chosen model, and emit
// predictions for the test split in the sample-submission schema.
//
// This step's subworkflow picks ONE interchangeable model module per run:
//   - model_family == 'graphormer' -> FINETUNE_GRAPHORMER (pretrained graph
//     transformer fine-tune; the orthogonal pretrained-foundation-model arm)
//   - otherwise                    -> TRAIN_BASELINE_MODEL (Morgan/sklearn baseline)
//
include { TRAIN_BASELINE_MODEL } from '../../modules/local/train_baseline_model/main'
include { FINETUNE_GRAPHORMER }  from '../../modules/local/finetune_graphormer/main'

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
    gr_finetune_mode
    gr_head
    gr_lr
    gr_epochs
    gr_ensemble
    gr_batch_size
    gr_weight_decay
    gr_head_dropout
    gr_freeze_layers
    gr_fuse_descriptors
    seed

    main:
    ch_versions = channel.empty()

    if (model_family == 'graphormer') {
        FINETUNE_GRAPHORMER(
            ch_public,
            task_type,
            gr_finetune_mode,
            gr_head,
            gr_lr,
            gr_epochs,
            gr_ensemble,
            gr_batch_size,
            gr_weight_decay,
            gr_head_dropout,
            gr_freeze_layers,
            gr_fuse_descriptors,
            morgan_radius,
            morgan_bits,
            seed,
        )
        ch_submission = FINETUNE_GRAPHORMER.out.submission
        ch_model_meta = FINETUNE_GRAPHORMER.out.model_meta
        ch_versions   = ch_versions.mix(FINETUNE_GRAPHORMER.out.versions)
    }
    else {
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
        ch_submission = TRAIN_BASELINE_MODEL.out.submission
        ch_model_meta = TRAIN_BASELINE_MODEL.out.model_meta
        ch_versions   = ch_versions.mix(TRAIN_BASELINE_MODEL.out.versions)
    }

    emit:
    submission = ch_submission
    model_meta = ch_model_meta
    versions   = ch_versions
}
