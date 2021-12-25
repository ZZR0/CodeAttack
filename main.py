from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from codeattack import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from codeattack.commands import TextAttackCommand
from codeattack.commands.attack_command import AttackCommand
from codeattack.commands.attack_resume_command import AttackResumeCommand
from codeattack.commands.augment_command import AugmentCommand
from codeattack.commands.benchmark_recipe_command import BenchmarkRecipeCommand
from codeattack.commands.eval_model_command import EvalModelCommand
from codeattack.commands.list_things_command import ListThingsCommand
from codeattack.commands.peek_dataset_command import PeekDatasetCommand
from codeattack.commands.train_model_command import TrainModelCommand

def run(args):
    a = vars(args)
    attack_args = CommandLineAttackArgs(**vars(args))
    dataset = DatasetArgs._create_dataset_from_args(attack_args)

    model_wrapper = ModelArgs._create_model_from_args(attack_args)
    attack = CommandLineAttackArgs._create_attack_from_args(
        attack_args, model_wrapper
    )
    attacker = Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()

if __name__ == "__main__":
    
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(help="textattack command helpers")

    AttackCommand.register_subcommand(subparsers)

    args = parser.parse_args()

    run(args)