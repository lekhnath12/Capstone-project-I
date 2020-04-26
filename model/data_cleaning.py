import re
import pandas as pd
import numpy as np
import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_csv("Traffic_Violations.csv")
df = df[['Description', 'Violation Type']]

df = df[df['Violation Type'].isin(['Warning', 'Citation'])]
df = df.drop_duplicates()

for col in df.columns:
    df[col] = df[col].apply(lambda x: re.sub("\s+", " ", str(x).strip().upper()))

df = df.drop_duplicates()
train_boolean = np.random.choice([True, False], size=len(df), replace=True, p=(0.5, 0.5))
df_train = df[train_boolean]
df_test = df[~train_boolean]

features_train = df_train['Description'].tolist()
features_test = df_test['Description'].tolist()
labels_train = df_train['Violation Type'].tolist()
labels_test = df_test['Violation Type'].tolist()

train_InputSamples = list(map(lambda x,y: bert.run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y),
                              features_train, labels_train))
test_InputSamples = list(map(lambda x,y: bert.run_classifier.InputExample(guid=None, text_a=x, text_b=None, label=y),
                              features_test, labels_test))

BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
tf.logging.set_verbosity(tf.logging.INFO)

def create_tokenizer_from_hub_module():
    """
    Load the pre-trained BERT model and extract the vocab file and tokenizer from TF HUB

    Returns
    -------
    BERT tokenizer object: bert.tokenization.FullTokenizer
        See: https://github.com/google-research/bert/blob/master/tokenization.py
        
    """

    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()

MAX_SEQ_LENGTH = 20
label_list = list(set(labels_train))

train_features = bert.run_classifier.convert_examples_to_features(train_InputSamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputSamples, label_list, MAX_SEQ_LENGTH, tokenizer)

def bert_model(is_predicting, input_ids, input_mask, segment_ids, labels, num_labels):
    """
    Our Custom fine-tuning Q&A classifier definition using BERT output layers.

    Parameters
    ----------
    is_predicting: boolean
        Boolean variable to indicate Training or Prediction mode.

    input_ids: Numpy Array
        BERT vocab token index for the input sample.

    input_mask: Numpy Array
        Flag to indicate if the input token is masked (1: Yes, 0:No).

    segment_ids: Numpy Array
        Flag to indicate which sentence the token belongs to. (0: 1st sentence, 1:2nd sentence).
        
    labels: Numpy Array
        Classification label for the input.
        
    num_labels: integer
        Total number of labels

    Returns
    -------
    In Training Mode return (Training Loss, Evaluation Labels, Evaluation probs per sample) tuple
    In Prediction Mode return (Evaluation Labels, Evaluation probs per sample) tuple

    """
    bert_module = hub.Module(BERT_MODEL_HUB,trainable=True)
    bert_inputs = dict( input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids)
    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens", as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence.
    # Use "sequence_outputs" for token-level output.
    output_layer = bert_outputs["pooled_output"]

    hidden_size = output_layer.shape[-1].value

    # Tunable layer.
    output_weights = tf.get_variable("output_weights", [num_labels, hidden_size],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    output_bias = tf.get_variable("output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):

        # Dropout helps prevent overfitting
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        # Convert labels into one-hot encoding
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
        # If we're predicting, we want predicted labels and the probabiltiies.
        if is_predicting:
            return (predicted_labels, log_probs)

        # If we're train/eval, compute loss between predicted and actual label
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        
        return (loss, predicted_labels, log_probs)
    
def model_fn_builder(num_labels, learning_rate, num_train_steps, num_warmup_steps):
    """
    Estimator driver logic for Training, Evaluation and Predict modes
    
    Parameters
    ----------
    num_labels : integer
        Total number of labels
        
    learning_rate : float
        Learning rate for underlying neural network
        
    num_train_steps: integer
        Number of steps to train (Sample Size/(Batch Size*Number of Epochs))
        
    num_warmup_steps: float
        Dynamic learning rate adjustment proportion
        
    Returns
    -------
    model_fn closure: Python Object
        Returns a closure of the driver logic
    
    """

    def model_fn(features, labels, mode, params):
        """
        Definition for Training, Evaluation and Predict modes
        
        Parameters
        ----------
        features: Dictionary
            Training/Test features
            
        labels: Numpy Array
            Train/Test labels
            
        mode: Numpy Array
            Train/Eval/Predict
            
        params: Dictionary
            Dict with training hyperparams
            
        Returns
        -------
        EstimatorSpec: tf.estimator.EstimatorSpec
            https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec
        
        """

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_predicting = (mode == tf.estimator.ModeKeys.PREDICT)

        # TRAIN and EVAL
        if not is_predicting:

            # Get BERT model definition
            (loss, predicted_labels, log_probs) = bert_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu=False)

            # Calculate evaluation metrics.
            def metric_fn(label_ids, predicted_labels):
                """
                Function to calculate training/evaluation metrics
                """
                
                recall = tf.metrics.recall(label_ids, predicted_labels)
                precision = tf.metrics.precision(label_ids, predicted_labels)
                true_pos = tf.metrics.true_positives(label_ids, predicted_labels)
                true_neg = tf.metrics.true_negatives(label_ids, predicted_labels)
                false_pos = tf.metrics.false_positives(label_ids, predicted_labels)
                false_neg = tf.metrics.false_negatives(label_ids, predicted_labels)
                
                return {
                    "precision": precision,
                    "recall": recall,
                    "true_positives": true_pos,
                    "true_negatives": true_neg,
                    "false_positives": false_pos,
                    "false_negatives": false_neg
                }

            eval_metrics = metric_fn(label_ids, predicted_labels)

            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        else:
            (predicted_labels, log_probs) = bert_model(is_predicting, input_ids, input_mask, segment_ids, label_ids, num_labels)

            predictions = {
                'probabilities': log_probs,
                'labels': predicted_labels
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Return the actual model function in the closure
    return model_fn


# Exercise: Modify the below values and observe the change in the training process
# Compute train and warmup steps from batch size
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_TRAIN_EPOCHS = 5.0
WARMUP_PROPORTION = 0.1
# Model configs
SAVE_CHECKPOINTS_STEPS = 10
SAVE_SUMMARY_STEPS = 10

num_train_steps = int(len(train_features) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

run_config = tf.estimator.RunConfig(model_dir='models',
                                    save_summary_steps=SAVE_SUMMARY_STEPS,
                                    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
model_fn = model_fn_builder(num_labels=len(label_list), learning_rate=LEARNING_RATE,
                            num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)

estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config, params={"batch_size": BATCH_SIZE})

train_input_fn = bert.run_classifier.input_fn_builder( features=train_features, seq_length=MAX_SEQ_LENGTH,
                                                      is_training=True, drop_remainder=False)

print('Start Training')
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print("End Training")

print("evaluate")
test_input_fn = run_classifier.input_fn_builder(features=test_features, seq_length=MAX_SEQ_LENGTH,
                                                is_training=False, drop_remainder=False)

metrics = estimator.evaluate(input_fn=test_input_fn, steps=None)
metrics["accuracy"] = (metrics["true_positives"] + metrics["true_negatives"])/(metrics["true_positives"] + metrics["true_negatives"]+metrics["false_positives"] + metrics["false_negatives"])
metrics["f1_score"] = (2*metrics["precision"]*metrics["recall"])/(metrics["precision"]+metrics["recall"])

print(metrics)

