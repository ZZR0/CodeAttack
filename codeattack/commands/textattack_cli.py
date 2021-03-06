"""

TextAttack CLI main class
==============================

"""


# !/usr/bin/env python
import argparse

from codeattack.commands.attack_command import AttackCommand
from codeattack.commands.attack_resume_command import AttackResumeCommand
from codeattack.commands.augment_command import AugmentCommand
from codeattack.commands.benchmark_recipe_command import BenchmarkRecipeCommand
from codeattack.commands.eval_model_command import EvalModelCommand
from codeattack.commands.list_things_command import ListThingsCommand
from codeattack.commands.peek_dataset_command import PeekDatasetCommand
from codeattack.commands.train_model_command import TrainModelCommand


def main():
    parser = argparse.ArgumentParser(
        "TextAttack CLI",
        usage="[python -m] texattack <command> [<args>]",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(help="textattack command helpers")

    # Register commands
    AttackCommand.register_subcommand(subparsers)
    AttackResumeCommand.register_subcommand(subparsers)
    AugmentCommand.register_subcommand(subparsers)
    BenchmarkRecipeCommand.register_subcommand(subparsers)
    EvalModelCommand.register_subcommand(subparsers)
    ListThingsCommand.register_subcommand(subparsers)
    TrainModelCommand.register_subcommand(subparsers)
    PeekDatasetCommand.register_subcommand(subparsers)

    # Let's go
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    # Run
    func = args.func
    del args.func
    func.run(args)


if __name__ == "__main__":
    main()
