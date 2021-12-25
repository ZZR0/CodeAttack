def test_perplexity():
    from codeattack.attack_results import SuccessfulAttackResult
    from codeattack.goal_function_results.classification_goal_function_result import (
        ClassificationGoalFunctionResult,
    )
    from codeattack.metrics.quality_metrics import Perplexity
    from codeattack.shared.attacked_text import AttackedCode

    sample_text = "hide new secretions from the parental units "
    sample_atck_text = "Ehide enw secretions from the parental units "

    results = [
        SuccessfulAttackResult(
            ClassificationGoalFunctionResult(
                AttackedCode(sample_text), None, None, None, None, None, None
            ),
            ClassificationGoalFunctionResult(
                AttackedCode(sample_atck_text), None, None, None, None, None, None
            ),
        )
    ]
    ppl = Perplexity(model_name="distilbert-base-uncased").calculate(results)

    assert int(ppl["avg_original_perplexity"]) == int(81.95)


def test_use():
    import transformers

    from codeattack import AttackArgs, Attacker
    from codeattack.attack_recipes import DeepWordBugGao2018
    from codeattack.datasets import HuggingFaceDataset
    from codeattack.metrics.quality_metrics import USEMetric
    from codeattack.models.wrappers import HuggingFaceModelWrapper

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english"
    )
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    attack = DeepWordBugGao2018.build(model_wrapper)
    dataset = HuggingFaceDataset("glue", "sst2", split="train")
    attack_args = AttackArgs(
        num_examples=1,
        log_to_csv="log.csv",
        checkpoint_interval=5,
        checkpoint_dir="checkpoints",
        disable_stdout=True,
    )
    attacker = Attacker(attack, dataset, attack_args)

    results = attacker.attack_dataset()

    usem = USEMetric().calculate(results)

    assert usem["avg_attack_use_score"] == 0.76
