import logging

logger = logging.getLogger('earlystopper')


def train_until_earlystop(train, validate, metric_key='metrics/auPRC', patience=4, tolerance=1e-4,
                          max_epochs=100):
    history = 0.0
    timer = 0
    checkpoint = None

    for epoch in range(max_epochs):
        logger.info('Starting epoch {}'.format(epoch))

        checkpoint = train(checkpoint)
        validation_metrics = validate(checkpoint)

        metric_value = validation_metrics[metric_key]

        if abs(history - metric_value) < tolerance:
            timer += 1
        else:
            history = metric_value
            timer = 0

        if timer > patience:
            logger.info('Early stopping at epoch {}'.format(epoch))
            break

    return checkpoint
