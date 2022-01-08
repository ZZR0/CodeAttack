"""
Managing Attack Logs.
========================
"""

from codeattack.goal_function_results.search_goal_function_result import (
    SearchGoalFunctionResult, 
    CloneGoalFunctionResult
)
from codeattack.goal_function_results.classification_goal_function_result import ClassificationGoalFunctionResult
from codeattack.goal_function_results.text_to_text_goal_function_result import TextToTextGoalFunctionResult
from codeattack.metrics.attack_metrics import (
    AttackQueries,
    AttackSuccessRate,
    WordsPerturbed,
    MRR, MAP, BLEU,
    F1, Precision, Recall,
)
from codeattack.metrics.quality_metrics import Perplexity, USEMetric

from . import CSVLogger, FileLogger, VisdomLogger, WeightsAndBiasesLogger

class AttackLogManager:
    """Logs the results of an attack to all attached loggers."""

    def __init__(self):
        self.loggers = []
        self.results = []
        self.enable_advance_metrics = False

    def enable_stdout(self):
        self.loggers.append(FileLogger(stdout=True))

    def enable_visdom(self):
        self.loggers.append(VisdomLogger())

    def enable_wandb(self, **kwargs):
        self.loggers.append(WeightsAndBiasesLogger(**kwargs))

    def disable_color(self):
        self.loggers.append(FileLogger(stdout=True, color_method="file"))

    def add_output_file(self, filename, color_method):
        self.loggers.append(FileLogger(filename=filename, color_method=color_method))

    def add_output_csv(self, filename, color_method):
        self.loggers.append(CSVLogger(filename=filename, color_method=color_method))

    def log_result(self, result):
        """Logs an ``AttackResult`` on each of `self.loggers`."""
        self.results.append(result)
        for logger in self.loggers:
            logger.log_attack_result(result)

    def log_results(self, results):
        """Logs an iterable of ``AttackResult`` objects on each of
        `self.loggers`."""
        for result in results:
            self.log_result(result)
        self.log_summary()

    def log_summary_rows(self, rows, title, window_id):
        for logger in self.loggers:
            logger.log_summary_rows(rows, title, window_id)

    def log_sep(self):
        for logger in self.loggers:
            logger.log_sep()

    def flush(self):
        for logger in self.loggers:
            logger.flush()

    def log_attack_details(self, attack_name, model_name):
        # @TODO log a more complete set of attack details
        attack_detail_rows = [
            ["Attack algorithm:", attack_name],
            ["Model:", model_name],
        ]
        self.log_summary_rows(attack_detail_rows, "Attack Details", "attack_details")

    def log_summary(self):
        total_attacks = len(self.results)
        if total_attacks == 0:
            return

        # Default metrics - calculated on every attack
        attack_success_stats = AttackSuccessRate().calculate(self.results)
        words_perturbed_stats = WordsPerturbed().calculate(self.results)
        attack_query_stats = AttackQueries().calculate(self.results)

        # @TODO generate this table based on user input - each column in specific class
        # Example to demonstrate:
        # summary_table_rows = attack_success_stats.display_row() + words_perturbed_stats.display_row() + ...
        summary_table_rows = [
            [
                "Number of successful attacks:",
                attack_success_stats["successful_attacks"],
            ],
            ["Number of failed attacks:", attack_success_stats["failed_attacks"]],
            ["Number of skipped attacks:", attack_success_stats["skipped_attacks"]],
            [
                "Original accuracy:",
                str(attack_success_stats["original_accuracy"]) + "%",
            ],
            [
                "Accuracy under attack:",
                str(attack_success_stats["attack_accuracy_perc"]) + "%",
            ],
            [
                "Attack success rate:",
                str(attack_success_stats["attack_success_rate"]) + "%",
            ],
            [
                "Average perturbed word %:",
                str(words_perturbed_stats["avg_word_perturbed_perc"]) + "%",
            ],
            [
                "Average num. words per input:",
                words_perturbed_stats["avg_word_perturbed"],
            ],
        ]

        summary_table_rows.append(
            ["Avg num queries:", attack_query_stats["avg_num_queries"]]
        )
        
        if isinstance(self.results[0].original_result, ClassificationGoalFunctionResult):
            stats = F1().calculate(self.results)
            summary_table_rows.append(
            ["Original F1:", stats["original_f1"]])
            summary_table_rows.append(
            ["Perturbed F1:", stats["perturbed_f1"]])

            stats = Precision().calculate(self.results)
            summary_table_rows.append(
            ["Original Precision:", stats["original_precision"]])
            summary_table_rows.append(
            ["Perturbed Precision:", stats["perturbed_precision"]])

            stats = Recall().calculate(self.results)
            summary_table_rows.append(
            ["Original Recall:", stats["original_recall"]])
            summary_table_rows.append(
            ["Perturbed Recall:", stats["perturbed_recall"]])

        if isinstance(self.results[0].original_result, SearchGoalFunctionResult):
            mrr_stats = MRR().calculate(self.results)
            summary_table_rows.append(
            ["Original MRR:", mrr_stats["original_mrr"]])
            summary_table_rows.append(
            ["Perturbed MRR:", mrr_stats["perturbed_mrr"]])
        
        if isinstance(self.results[0].original_result, CloneGoalFunctionResult):
            map_stats = MAP().calculate(self.results)
            summary_table_rows.append(
            ["Original MAP:", map_stats["original_map"]])
            summary_table_rows.append(
            ["Perturbed MAP:", map_stats["perturbed_map"]])
        
        if isinstance(self.results[0].original_result, TextToTextGoalFunctionResult):
            map_stats = BLEU().calculate(self.results)
            summary_table_rows.append(
            ["Original BLEU-4:", map_stats["original_bleu4"]])
            summary_table_rows.append(
            ["Perturbed BLEU-4:", map_stats["perturbed_bleu4"]])

        if self.enable_advance_metrics:
            perplexity_stats = Perplexity().calculate(self.results)
            use_stats = USEMetric().calculate(self.results)

            summary_table_rows.append(
                [
                    "Average Original Perplexity:",
                    perplexity_stats["avg_original_perplexity"],
                ]
            )

            summary_table_rows.append(
                [
                    "Average Attack Perplexity:",
                    perplexity_stats["avg_attack_perplexity"],
                ]
            )
            summary_table_rows.append(
                ["Average Attack USE Score:", use_stats["avg_attack_use_score"]]
            )

        self.log_summary_rows(
            summary_table_rows, "Attack Results", "attack_results_summary"
        )
        # Show histogram of words changed.
        numbins = max(words_perturbed_stats["max_words_changed"], 10)
        for logger in self.loggers:
            logger.log_hist(
                words_perturbed_stats["num_words_changed_until_success"][:numbins],
                numbins=numbins,
                title="Num Words Perturbed",
                window_id="num_words_perturbed",
            )
