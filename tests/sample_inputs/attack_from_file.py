import codeattack


def Attack(model):
    goal_function = codeattack.goal_functions.UntargetedClassification(model)
    search_method = codeattack.search_methods.GreedyWordSwapWIR()
    transformation = codeattack.transformations.WordSwapRandomCharacterSubstitution()
    constraints = []
    return codeattack.Attack(goal_function, constraints, transformation, search_method)
