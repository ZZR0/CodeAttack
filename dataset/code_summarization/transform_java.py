import os
import re
import json
import tqdm
import random
import itertools
import multiprocessing
import argparse
from parser import JAVA_AST as AST

random.seed(0)

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--max_holes', dest='max_holes', default=50, help='max number of holes to be inserted')
args = args_parser.parse_args()


def t_rename_func(the_code, uid=1, all_sites=False):
    """
    all_sites=True: a single, randomly selected, referenced field 
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    func = the_ast.get_func_name()

    func_dict = AST.conv2dict(func)
    func_dict, code_list = AST.split_code(func_dict, the_code)
    
    site_map = {}

    for name in func_dict:
        for p in func_dict[name]:
            p, deep = p[0], p[1]

            site_map["@R_{}@".format(uid+count)] = (code_list[p], "transforms.RenameFields")
            code_list[p] = "REPLACEME{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_rename_fields(the_code, uid=1, all_sites=False):
    """
    all_sites=True: a single, randomly selected, referenced field 
    (self.field in Python) has its name replaced by a hole
    all_sites=False: all possible fields are selected
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    fields = the_ast.get_fields()

    fields_dict = AST.conv2dict(fields)
    fields_dict, code_list = AST.split_code(fields_dict, the_code)
    
    site_map = {}

    for name in fields_dict:
        for p in fields_dict[name]:
            p, deep = p[0], p[1]

            site_map["@R_{}@".format(uid+count)] = (code_list[p], "transforms.RenameFields")
            code_list[p] = "REPLACEME{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_rename_parameters(the_code, uid=1, all_sites=False):
    """
    Parameters get replaced by holes.
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    parameters = the_ast.get_parameters()

    parameters = [par.source for par in parameters]
    parameters = the_ast.get_selected_parameters(parameters)
    parameters_dict = AST.conv2dict(parameters)
    parameters_dict, code_list = AST.split_code(parameters_dict, the_code)
    
    site_map = {}

    for name in parameters_dict:
        for p in parameters_dict[name]:
            p, deep = p[0], p[1]

            site_map["@R_{}@".format(uid+count)] = (code_list[p], "transforms.RenameParameters")
            code_list[p] = "REPLACEME{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_rename_local_variables(the_code, uid=1, all_sites=False):
    """
    Local variables get replaced by holes.
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    variables = the_ast.get_local_variables()

    variables = [par.source for par in variables]
    variables = the_ast.get_selected_variables(variables)
    variables_dict = AST.conv2dict(variables)
    variables_dict, code_list = AST.split_code(variables_dict, the_code)
    
    site_map = {}

    for name in variables_dict:
        for p in variables_dict[name]:
            p, deep = p[0], p[1]

            site_map["@R_{}@".format(uid+count)] = (code_list[p], "transforms.RenameLocalVariables")
            code_list[p] = "REPLACEME{}".format(uid+count)
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_add_dead_code(the_code, uid=1, all_sites=False):
    """
    Statement of the form if False:\n <HOLE> = 1 is added to the target program. 
    all_sites=False: The insertion location (either beginning, or end) is chosen at random.
    all_sites=True: The statement is inserted at all possible locations.
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    statements = the_ast.get_statements()

    statements_dict = AST.conv2dict(statements)
    statements_dict, code_list = AST.split_code(statements_dict, the_code)
    site_map = {}
    first = True
    for name in statements_dict:
        for p in statements_dict[name]:
            p, deep = p[0], p[1]
            if first:
                first = False
                site_map["if (false) {{ int @R_{}@ = 1; }};".format(uid+count)] = ("", "transforms.AddDeadCode")
                code_list[p] = "if (false) {{ int REPLACEME{} = 1; }};\n".format(uid+count) + "    "*deep + code_list[p]
                count += 1
            site_map["if (false) {{ int @R_{}@ = 1; }};".format(uid+count)] = ("", "transforms.AddDeadCode")
            code_list[p] = code_list[p] + "\n"+"    "*deep+"if (false) {{ int REPLACEME{} = 1; }};\n".format(uid+count)+"    "*deep
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_insert_print_statements(the_code, uid=1, all_sites=False):
    """
    Print statements of the form 'print( <HOLE>)' are inserted in the 
    target program. 
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    statements = the_ast.get_statements()

    statements_dict = AST.conv2dict(statements)
    statements_dict, code_list = AST.split_code(statements_dict, the_code)
    site_map = {}
    first = True
    for name in statements_dict:
        for p in statements_dict[name]:
            p, deep = p[0], p[1]
            if first:
                first = False
                site_map["System.out.println(\"@R_{}@\");".format(uid+count)] = ("", "transforms.InsertPrintStatements")
                code_list[p] = "System.out.println(\"REPLACEME{}\");\n".format(uid+count) + "    "*deep + code_list[p]
                count += 1
            site_map["System.out.println(\"@R_{}@\");".format(uid+count)] = ("", "transforms.InsertPrintStatements")
            code_list[p] = code_list[p] + "\n"+"    "*deep+"System.out.println(\"REPLACEME{}\");\n".format(uid+count)+"    "*deep
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


def t_replace_true_false(the_code, uid=1, all_sites=False):
    """
    Boolean literals are replaced by an equivalent
    expression containing a single hole 
    (e.g., ("<HOLE>" == "<HOLE>") to replace true).
    """
    count = 0
    changed = False
    the_ast = AST.build_ast(the_code)
    booleans = the_ast.get_booleans()

    booleans_dict = AST.conv2dict(booleans)
    booleans_dict, code_list = AST.split_code(booleans_dict, the_code)
    
    site_map = {}

    for name in booleans_dict:
        for p in booleans_dict[name]:
            p, deep = p[0], p[1]

            if name.lower() == "true":
                site_map['"@R_{}@" == "@R_{}@"'.format(uid+count, uid+count)] = ("True", "transforms.ReplaceTrueFalse")
                code_list[p] = "\"REPLACEME{}\" == \"REPLACEME{}\"".format(uid+count, uid+count)
            if name.lower() == "false":
                site_map['"@R_{}@" != "@R_{}@"'.format(uid+count, uid+count)] = ("False", "transforms.ReplaceTrueFalse")
                code_list[p] = "\"REPLACEME{}\" != \"REPLACEME{}\"".format(uid+count, uid+count)
        count += 1
    
    the_code = "".join(code_list)
    changed = count > 0
    return changed, the_code, uid+count-1, site_map


class t_seq(object):
    def __init__(self, transforms, all_sites):
        self.transforms = transforms
        self.all_sites = all_sites
    def __call__(self, the_ast, all_sites=False):
        did_change = False
        cur_ast = the_ast
        cur_idx = -1
        new_site_map = {}
        for t in self.transforms:
            changed, cur_ast, cur_idx, site_map = t(cur_ast, cur_idx+1, self.all_sites)
            if changed:
                did_change = True
                new_site_map.update(site_map)
        return did_change, cur_ast, cur_idx, new_site_map

def t_identity(the_code, all_sites=None):
    return True, the_code, 0, {}

def handle_replacement_tokens(line):
  new_line = line
  uniques = set()
  for match in re.compile('REPLACEME\d+').findall(line):
    uniques.add(match.strip())
  uniques = list(uniques)
  uniques.sort()
  uniques.reverse()
  for match in uniques:
    replaced = match.replace("REPLACEME", "@R_") + '@'
    new_line = new_line.replace(match, replaced)
  return new_line


def process(item):
    (split, the_hash, og_code, as_json) = item

    transforms = [
        (
        'transforms.Identity',
        t_identity
        )
    ]

    transforms.append(('transforms.Combined', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false, t_insert_print_statements, t_add_dead_code], all_sites=True)))
    transforms.append(('transforms.Insert', t_seq([t_insert_print_statements, t_add_dead_code], all_sites=True)))
    transforms.append(('transforms.Replace', t_seq([t_rename_local_variables, t_rename_parameters, t_rename_fields, t_replace_true_false], all_sites=True)))
        
    results = []
    for t_name, t_func in transforms:
        try:
            # print(t_func)
            changed, result, last_idx, site_map = t_func(
                og_code,
                all_sites=True
            )
            result = handle_replacement_tokens(result)
            results.append((changed, split, t_name, the_hash, result, site_map, as_json)) 
        except Exception as ex:
            import traceback
            traceback.print_exc()
            results.append((False, split, t_name, the_hash, og_code, {}, as_json))
    return results


if __name__ == "__main__":
    print("Starting transform:")
    pool = multiprocessing.Pool(1)

    tasks = []

    print("  + Loading tasks...")
    splits = ['test']

    for split in splits:
        with open('./{}.jsonl'.format(split)) as f:
            for idx, line in enumerate(f.readlines()):
                as_json = json.loads(line)
                as_json["code_tokens"] = " ".join(as_json['code_tokens'])
                as_json["docstring_tokens"] = " ".join(as_json['docstring_tokens'])
                the_code = as_json["code_tokens"]
                the_code = AST.remove_comments_and_docstrings(the_code, "java")
                tasks.append((split, idx, the_code, as_json))

    print("    + Loaded {} transform tasks".format(len(tasks)))
    # task[114:115] has multiple variables.
    results = pool.imap_unordered(process, tasks, 3000)

    print("  + Transforming in parallel...")
    names_covered = []
    all_sites = {}
    all_results = {}
    idx = 0
    for changed, split, t_name, the_hash, code, site_map, as_json in itertools.chain.from_iterable(tqdm.tqdm(results, desc="    + Progress", total=len(tasks))):
        idx += 1
        if not changed: continue
        # all_sites[(t_name, split, the_hash)] = site_map
        adv_json = {}
        adv_json["code"] = as_json["code_tokens"]
        adv_json["nl"] = as_json["docstring_tokens"]

        adv_json["adv"] = code
        adv_json["idx"] = the_hash
        adv_json["site_map"] = site_map

        if t_name not in all_sites:
            all_sites[t_name] = {split:{the_hash:site_map}}
            all_results[t_name] = {split:[adv_json]}
        else:
            if split not in all_sites[t_name]:
                all_sites[t_name][split] = {the_hash:site_map}
                all_results[t_name][split] = [adv_json]
            else:
                all_sites[t_name][split][the_hash] = site_map
                all_results[t_name][split] += [adv_json]

    for t_name in all_sites:
        for split in all_sites[t_name]:
            if not os.path.exists('./'+t_name):
                os.makedirs('./'+t_name)
            with open('./{}/adv_{}_site_map.json'.format(t_name, split), 'w') as f:
                json.dump(all_sites[t_name][split], f)
    
    for t_name in all_results:
        for split in all_results[t_name]:
            if not os.path.exists('./'+t_name):
                os.makedirs('./'+t_name)
            with open('./{}/adv_{}.jsonl'.format(t_name, split), 'w') as f:
                for line in all_results[t_name][split]:
                    f.write(json.dumps(line)+"\n")

    print("  + Transforms complete!")
