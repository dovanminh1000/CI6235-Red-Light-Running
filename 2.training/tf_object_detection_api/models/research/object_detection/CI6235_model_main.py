# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Binary to run train and evaluation on object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import tensorflow as tf

from object_detection import model_hparams
from object_detection import model_lib

flags.DEFINE_string(
    "model_dir",
    None,
    "Path to output model directory "
    "where event and checkpoint files will be written.",
)
flags.DEFINE_string("pipeline_config_path", None, "Path to pipeline config " "file.")
flags.DEFINE_integer("num_train_steps", None, "Number of train steps.")
flags.DEFINE_boolean(
    "eval_training_data",
    False,
    "If training data should be evaluated for this job. Note "
    "that one call only use this in eval-only mode, and "
    "`checkpoint_dir` must be supplied.",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_examples",
    1,
    "Will sample one of " "every n eval input examples, where n is provided.",
)
flags.DEFINE_integer(
    "sample_1_of_n_eval_on_train_examples",
    5,
    "Will sample "
    "one of every n train input examples for evaluation, "
    "where n is provided. This is only used if "
    "`eval_training_data` is True.",
)
flags.DEFINE_string(
    "hparams_overrides",
    None,
    "Hyperparameter overrides, "
    "represented as a string containing comma-separated "
    "hparam_name=value pairs.",
)
flags.DEFINE_string(
    "checkpoint_dir",
    None,
    "Path to directory holding a checkpoint.  If "
    "`checkpoint_dir` is provided, this binary operates in eval-only mode, "
    "writing resulting metrics to `model_dir`.",
)
flags.DEFINE_boolean(
    "run_once",
    False,
    "If running in eval-only mode, whether to run just "
    "one round of eval vs running continuously (default).",
)
FLAGS = flags.FLAGS

from tensorflow.python.training import session_run_hook
import numpy as np

#CI6235: Use early stopping for training model, reference from https://github.com/hongym7/early_stopping
class EarlyStoppingHook(session_run_hook.SessionRunHook):
    """Hook that requests stop at a specified step."""

    def __init__(self, monitor="val_loss", min_delta=0, patience=0, mode="auto"):
        """
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        if mode not in ["auto", "min", "max"]:
            logging.warning(
                "EarlyStopping mode %s is unknown, " "fallback to auto mode.",
                mode,
                RuntimeWarning,
            )
            mode = "auto"

        if mode == "min":
            self.monitor_op = np.less
        elif mode == "max":
            self.monitor_op = np.greater
        else:
            if "acc" in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def begin(self):
        # Convert names to tensors if given
        graph = tf.get_default_graph()
        self.monitor = graph.as_graph_element(self.monitor)
        print("  self.monitor: ", self.monitor)

        if isinstance(self.monitor, tf.Operation):
            self.monitor = self.monitor.outputs[0]

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs(self.monitor)

    def after_run(self, run_context, run_values):
        current = run_values.results

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                run_context.request_stop()


def main(unused_argv):
    print(": Entered model_main.py main function... ")
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    flags.mark_flag_as_required("model_dir")
    flags.mark_flag_as_required("pipeline_config_path")
    config = tf.estimator.RunConfig(
        model_dir=FLAGS.model_dir, session_config=session_config
    )

    train_and_eval_dict = model_lib.create_estimator_and_inputs(
        run_config=config,
        hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
        pipeline_config_path=FLAGS.pipeline_config_path,
        train_steps=FLAGS.num_train_steps,
        sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
        sample_1_of_n_eval_on_train_examples=(
            FLAGS.sample_1_of_n_eval_on_train_examples
        ),
    )
    estimator = train_and_eval_dict["estimator"]
    train_input_fn = train_and_eval_dict["train_input_fn"]
    eval_input_fns = train_and_eval_dict["eval_input_fns"]
    eval_on_train_input_fn = train_and_eval_dict["eval_on_train_input_fn"]
    predict_input_fn = train_and_eval_dict["predict_input_fn"]
    train_steps = train_and_eval_dict["train_steps"]

    if FLAGS.checkpoint_dir:
        if FLAGS.eval_training_data:
            name = "training_data"
            input_fn = eval_on_train_input_fn
        else:
            name = "validation_data"
            # The first eval input will be evaluated.
            input_fn = eval_input_fns[0]

        if FLAGS.run_once:
            print("CI6235: Ready to test ONCE ... ")
            estimator.evaluate(
                input_fn,
                steps=None,
                checkpoint_path=tf.train.latest_checkpoint(FLAGS.checkpoint_dir),
            )
        else:
            model_lib.continuous_eval(
                estimator, FLAGS.checkpoint_dir, input_fn, train_steps, name
            )
    else:
        print("CI6235: Ready to train and evaluate ... ")
	#CI6235: Apply early stopping with patience is 6000
        early_stopping_hook = EarlyStoppingHook(
            monitor="total_loss", patience=6000, mode="min"
        )
	#CI6235: Configuration for the "train" part for the train_and_evaluate call.
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=train_steps, hooks=[early_stopping_hook]
        )

	#CI6235: Configure to save the best model and fintal model in validation steps 

        final_exporter_name = "final_exporter"
        eval_specs = []
        eval_spec_names = [str(i) for i in range(len(eval_input_fns))]
        for index, (eval_spec_name, eval_input_fn) in enumerate(
            zip(eval_spec_names, eval_input_fns)
        ):
            if index == 0:
                exporter_name = final_exporter_name
            else:
                exporter_name = "{}_{}".format(final_exporter_name, eval_spec_name)

            final_exporter = tf.estimator.FinalExporter(
                name=final_exporter_name, serving_input_receiver_fn=predict_input_fn
            )

            best_exporter = tf.estimator.BestExporter(
                name="best_exporter",
                serving_input_receiver_fn=predict_input_fn,
                event_file_pattern="eval_0/*.tfevents.*",
                exports_to_keep=5,
            )
            exporter = [final_exporter, best_exporter]

            eval_specs.append(
                tf.estimator.EvalSpec(
                    name=eval_spec_name,
                    input_fn=eval_input_fn,
                    steps=None,
                    exporters=exporter,
                )
            )

        #CI6235: Currently only a single Eval Spec is allowed.
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])


if __name__ == "__main__":
    tf.app.run()
