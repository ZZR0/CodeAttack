"""

AttackCommand class
===========================

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from codeattack import Attacker, CommandLineAttackArgs, DatasetArgs, ModelArgs
from codeattack.commands import TextAttackCommand


class AttackCommand(TextAttackCommand):
    """The TextAttack attack module:

    A command line parser to run an attack from user specifications.
    """


    @staticmethod
    def register_subcommand(main_parser: ArgumentParser):
        parser = main_parser.add_parser(
            "attack",
            help="run an attack on an NLP model",
            formatter_class=ArgumentDefaultsHelpFormatter,
        )
        parser = CommandLineAttackArgs._add_parser_args(parser)
