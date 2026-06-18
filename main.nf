#!/usr/bin/env nextflow

/*
 * BioML-bench drug-discovery evaluation (Polaris / TDC). The exact benchmark,
 * metric and direction are set by params (see nextflow.config / nextflow_schema.json).
 *
 * One run = one parameter assignment, the contract Stimulus optimises over.
 * The pipeline stages a pinned, immutable dataset prefix from S3 (train +
 * test_features, no test labels), fits a baseline molecular-property model,
 * predicts the test split, and grades the submission against the pinned private
 * answers. The single objective is written to result.json under result.objective
 * (objective_mode says whether to minimise or maximise it).
 *
 * This is a deliberately-minimal *base* pipeline: the Gradient autonomous loop
 * and Stimulus search the model knobs exposed in nextflow_schema.json to push
 * the objective past this baseline.
 */

nextflow.enable.dsl = 2

include { PREPARE_TASK_DATA }  from './subworkflows/local/prepare_task_data'
include { FIT_BASELINE_MODEL } from './subworkflows/local/fit_baseline_model'
include { SCORE_SUBMISSION }   from './subworkflows/local/score_submission'

workflow {

    // ----------------------------
    // Parameter setup
    // ----------------------------
    run_id            = params.run_id
    benchmark_id      = params.benchmark_id
    main_metric       = params.main_metric
    objective_mode    = params.objective_mode
    task_type         = params.task_type
    dataset_s3_prefix = params.dataset_s3_prefix
    model_family      = params.model_family
    morgan_radius     = params.morgan_radius
    morgan_bits       = params.morgan_bits
    ridge_alpha       = params.ridge_alpha
    logistic_c        = params.logistic_c
    rf_n_estimators   = params.rf_n_estimators
    hgb_learning_rate = params.hgb_learning_rate
    seed              = params.seed

    ch_task = channel.of(tuple([id: run_id], file(dataset_s3_prefix)))

    // ----------------------------
    // Pipeline run
    // ----------------------------

    /*
    Stage and validate the pinned, immutable dataset prefix from S3 (Fusion):
    copy the public training contract (train.csv, test_features.csv,
    sample_submission.csv) for the model and the private answers.csv for the
    grader only.
    */
    PREPARE_TASK_DATA(ch_task)

    /*
    Featurize molecules, fit the baseline estimator on train, and predict the
    held-out test split in the sample-submission schema.
    */
    FIT_BASELINE_MODEL(
        PREPARE_TASK_DATA.out.public_data,
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

    /*
    Grade the submission against the pinned private answers and write the
    Stimulus-facing result.json (result.objective = main_metric).
    */
    SCORE_SUBMISSION(
        FIT_BASELINE_MODEL.out.submission,
        PREPARE_TASK_DATA.out.answers,
        benchmark_id,
        main_metric,
        objective_mode,
        task_type,
    )
}
