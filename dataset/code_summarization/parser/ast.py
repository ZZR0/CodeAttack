from tree_sitter import Language, Parser
import re
from functools import cmp_to_key

parsers={}      
for lang in ["python", "java", "cpp"]:  
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parsers[lang] = parser

class P_AST(object):
    def __init__(self, root, code, idx=0, parent=None, deep=0) -> None:
        self.type = root.type
        self.parent = parent
        if self.parent is not None:
            self.path = self.parent.path + "|" + self.type
        else:
            self.path = self.type

        self.deep = deep
        if self.type == "block":
            self.deep += 1

        self.start_byte = root.start_byte
        self.end_byte = root.end_byte

        self.source = code[self.start_byte:self.end_byte]
        
        self.idx = idx

        self.children = [P_AST(child, code, idx=i, parent=self, deep=self.deep) for i, child in enumerate(root.children)]

    @classmethod
    def build_ast(cls, code, lang="python"):
        node = parsers[lang].parse(bytes(code,'utf8'))
        the_ast = P_AST(node.root_node, code)
        the_ast.link_ast()
        return the_ast

    @classmethod
    def conv2dict(cls, nodes):
        nodes_dict = {}
        for node in nodes:
            if node.source in nodes_dict:
                nodes_dict[node.source].append((node.start_byte,node.end_byte,node.deep))
            else:
                nodes_dict[node.source] = [(node.start_byte,node.end_byte,node.deep)]
        return nodes_dict
    
    @classmethod
    def split_code(cls, identity_dict, code):
        # def cmp(x, y):
        #     if x[0] > y[0]: return 1
        #     if x[0] < y[0]: return -1
        #     if x[1] > y[1]: return -1
        #     if x[1] < y[1]: return 1
        #     return 0

        def overlap_split(p_list, code):
            if len(p_list) < 2: return p_list
            new_p_list, o_list = [], []
            end = 0
            for idx in range(len(p_list)):
                if end >= p_list[idx][1]:
                    if not o_list:
                        o_list.append(p_list[idx-1])
                        new_p_list = new_p_list[:-1]
                    o_list.append(p_list[idx])
                else:
                    if o_list:
                        sub_list = []
                        sub_list.append((o_list[0][0], o_list[1][0], str(code[o_list[0][0]: o_list[1][0]], encoding="utf-8"), o_list[1][3]))
                        sub_list.extend(o_list[1:])
                        if o_list[-1][1] < o_list[0][1]:
                            sub_list.append((o_list[-1][1], o_list[0][1], str(code[o_list[-1][1]: o_list[0][1]], encoding="utf-8"), o_list[-1][3]))
                        sub_list = overlap_split(sub_list, code)
                        o_list = []
                        new_p_list.extend(sub_list)
                    new_p_list.append(p_list[idx])
                    end = p_list[idx][1]
            
            if o_list:
                sub_list = []
                sub_list.append((o_list[0][0], o_list[1][0], str(code[o_list[0][0]: o_list[1][0]], encoding="utf-8"), o_list[1][3]))
                sub_list.extend(o_list[1:])
                if o_list[-1][1] < o_list[0][1]:
                    sub_list.append((o_list[-1][1], o_list[0][1], str(code[o_list[-1][1]: o_list[0][1]], encoding="utf-8"), o_list[-1][3]))
                sub_list = overlap_split(sub_list, code)
                o_list = []
                new_p_list.extend(sub_list)
                
            return new_p_list

        def process_dict(ori_dict, code):
            split_p = []
            for item in ori_dict.items():
                key, value = item[0], item[1]
                for p in value:
                    split_p.append((p[0], p[1], key, p[2]))
            # split_p.sort(key=cmp_to_key(cmp))
            split_p.sort(key=lambda x:x[0])
            split_p = overlap_split(split_p, code)
            return split_p

        code = bytes(code, "utf8")
        split_p = process_dict(identity_dict, code)
        split_str = []

        last_end = 0
        idx = -1
        identity_dict = {}
        for p in split_p:
            if last_end != p[0]:
                split_str.append(code[last_end:p[0]])
                idx += 1
            split_str.append(code[p[0]:p[1]])
            idx += 1
            if p[2] in identity_dict:
                identity_dict[p[2]].append((idx, p[3]))
            else:
                identity_dict[p[2]] = [(idx, p[3])]
            last_end = p[1]
        split_str.append(code[last_end:])
        split_str = [str(c, encoding="utf-8") for c in split_str]

        return identity_dict, split_str

    def link_ast(self):
        self.brother = self.parent.children if self.parent is not None else None

        self.left = None
        self.right = None
        if self.parent is not None:
            if self.idx > 0:
                self.left = self.parent.children[self.idx-1]
            
            if self.idx < len(self.parent.children)-1:
                self.right = self.parent.children[self.idx+1]
        
        for child in self.children:
            child.link_ast()
        
        # self.prev = None
        # if self.left is not None:
        #     self.prev = self.left
        # elif self.parent is not None:
        #     self.prev = self.parent.brother[-1]

        # self.next = None
        # if self.right is not None:
        #     self.next = self.right
        # elif self.brother is not None:
        #     self.next = self.brother[0]

    def print_ast(self, deep=0):
        print(" "*deep*4 + self.type)
        for child in self.children:
            child.print_ast(deep+1)

    @staticmethod
    def remove_comments_and_docstrings(source,lang):
        if lang in ['python']:
            """
            Returns 'source' minus comments and docstrings.
            """
            the_ast = PYTHON_AST.build_ast(source)
            comments = the_ast.get_comments()
            comments_dict = PYTHON_AST.conv2dict(comments)
            comments_dict, code_list = PYTHON_AST.split_code(comments_dict, source)

            for name in comments_dict:
                for p in comments_dict[name]:
                    p = p[0]
                    code_list[p] = ""
            
            the_code = "".join(code_list)
            return the_code

        elif lang in ['ruby']:
            return source
        else:
            def replacer(match):
                s = match.group(0)
                if s.startswith('/'):
                    return " " # note: a space and not an empty string
                else:
                    return s
            pattern = re.compile(
                r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
                re.DOTALL | re.MULTILINE
            )
            temp=[]
            for x in re.sub(pattern, replacer, source).split('\n'):
                if x.strip()!="":
                    temp.append(x)
            return '\n'.join(temp)

class PYTHON_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0) -> None:
        self.type = root.type
        self.parent = parent
        if self.parent is not None:
            self.path = self.parent.path + "|" + self.type
        else:
            self.path = self.type

        self.deep = deep
        if self.type == "block":
            self.deep += 1

        self.start_byte = root.start_byte
        self.end_byte = root.end_byte

        self.source = code[self.start_byte:self.end_byte]
        
        self.idx = idx

        self.children = [PYTHON_AST(child, code, idx=i, parent=self, deep=self.deep) for i, child in enumerate(root.children)]

    @classmethod
    def build_ast(cls, code, lang="java"):
        node = parsers[lang].parse(bytes(code,'utf8'))
        the_ast = PYTHON_AST(node.root_node, code)
        the_ast.link_ast()
        return the_ast

    def check_left(self):
        right = self.right
        while right:
            if right.type == "=" or right.type == "in" or right.type == "as":
                return True
            right = right.right
        return False

    def check_right(self):
        left = self.left
        while left:
            if left.type == "=" or left.type == "in" or left.type == "as":
                return True
            left = left.left
        return False


    def get_fields(self):
        fields = []
        if self.path.endswith("attribute|identifier") and self.source == "self" \
            and self.parent.parent is not None and self.parent.parent.type != "call":
            fields += [self.brother[-1]]
        
        for child in self.children:
            fields += child.get_fields()
        
        return fields

    
    def get_parameters(self):
        parameters = []
        if (self.path.endswith("|parameters|identifier")
            or (self.path.endswith("|parameters|default_parameter|identifier") and self.idx==0)
            or self.path.endswith("|parameters|dictionary_splat_pattern|identifier")
            or self.path.endswith("|parameters|list_splat_pattern|identifier")
            ) and self.source != "self":
            parameters += [self]
        
        for child in self.children:
            parameters += child.get_parameters()
        
        return parameters


    def get_local_variables(self):
        variables = []
        # if self.source == "u_MAC":
        #     print()
        if ((self.path.endswith("assignment|identifier") and self.check_left())
            or (self.path.endswith("assignment|pattern_list|identifier") and self.parent.check_left())
            or (self.path.endswith("for_statement|identifier") and self.check_left())
            or (self.path.endswith("for_in_clause|identifier") and self.check_left())
            or (self.path.endswith("with_item|identifier") and self.check_right())
            ):
            variables += [self]
        
        for child in self.children:
            variables += child.get_local_variables()
        
        return variables


    def get_statements(self):
        statements = []
        pattern = re.compile(r'block\|\w+_statement$')
        result = pattern.search(self.path)
        # if self.path.endswith("try_statement|block|expression_statement"):
        #     print()
        if result is not None:
            statements += [self]
        
        for child in self.children:
            statements += child.get_statements()
        
        return statements

    
    def get_selected_identifier(self, vars):
        identifier = []
        if self.path.endswith("identifier") and self.source in vars:
            identifier += [self]
        
        for child in self.children:
            identifier += child.get_selected_identifier(vars)
        
        return identifier


    def get_selected_parameters(self, vars):
        variables = []
        if self.path.endswith("identifier") and self.source in vars:
            if self.path.endswith("attribute|identifier"):
                if self.idx == 0:
                    variables += [self]
            elif self.path.endswith("keyword_argument|identifier"):
                if self.idx != 0:
                    variables += [self]
            else:
                variables += [self]
        
        for child in self.children:
            variables += child.get_selected_parameters(vars)
        
        return variables

    
    def get_selected_variables(self, vars):
        variables = []
        if self.path.endswith("identifier") and self.source in vars:
            if self.path.endswith("attribute|identifier"):
                if self.idx == 0:
                    variables += [self]
            elif self.path.endswith("keyword_argument|identifier"):
                if self.idx != 0:
                    variables += [self]
            elif self.path.endswith("parameters|identifier") or self.path.endswith("default_parameter|identifier"):
                pass
            else:
                variables += [self]
        
        for child in self.children:
            variables += child.get_selected_variables(vars)
        
        return variables

    
    def get_booleans(self):
        results = []
        if self.type == "true" or self.type == "false":
            results += [self]
        
        for child in self.children:
            results += child.get_booleans()
        
        return results

    def get_func_name(self):
        results = []
        if self.path == "module|function_definition|identifier":
            results += [self]
        
        for child in self.children:
            results += child.get_func_name()
        
        return results
                
    def get_comments(self):
        comments = []
        if self.path.endswith("comment") or self.path.endswith("expression_statement|string"):
            comments += [self]
        
        for child in self.children:
            comments += child.get_comments()
        
        return comments

class JAVA_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0) -> None:
        self.type = root.type
        self.parent = parent
        if self.parent is not None:
            self.path = self.parent.path + "|" + self.type
        else:
            self.path = self.type

        self.deep = deep
        if self.type == "block":
            self.deep += 1

        self.start_byte = root.start_byte
        self.end_byte = root.end_byte

        self.source = str(code[self.start_byte:self.end_byte], encoding = "utf-8")
        
        self.idx = idx

        self.children = [JAVA_AST(child, code, idx=i, parent=self, deep=self.deep) for i, child in enumerate(root.children)]

    @classmethod
    def build_ast(cls, code, lang="java"):
        the_code = bytes(code,'utf8')
        node = parsers[lang].parse(the_code)
        the_ast = JAVA_AST(node.root_node, the_code)
        the_ast.link_ast()
        return the_ast

    def get_fields(self):
        fields = []
        if self.path.endswith("field_access|this"):
            fields += [self.brother[-1]]
        
        for child in self.children:
            fields += child.get_fields()
        
        return fields
    
    def get_parameters(self):
        parameters = []
        if (self.path.endswith("|formal_parameter|identifier")
            or self.path.endswith("|spread_parameter|variable_declarator|identifier")):
            parameters += [self]
        
        for child in self.children:
            parameters += child.get_parameters()
        
        return parameters
    
    def get_selected_parameters(self, vars):
        variables = []
        if self.path.endswith("identifier") and self.source in vars:
            if  self.path.endswith("field_access|identifier"):
                if self.idx==0:
                    variables += [self]
            else:
                variables += [self]
        
        for child in self.children:
            variables += child.get_selected_parameters(vars)
        
        return variables

    def check_left(self):
        right = self.right
        while right:
            if right.type == "=" or right.type == "in" or right.type == ":":
                return True
            right = right.right
        return False

    def get_local_variables(self):
        variables = []
        # if self.source == "u_MAC":
        #     print()
        if self.path.endswith("|local_variable_declaration|variable_declarator|identifier") \
            and self.idx == 0:
            variables += [self]
        elif self.path.endswith("|enhanced_for_statement|identifier") and self.check_left():
            variables += [self]
        for child in self.children:
            variables += child.get_local_variables()
        
        return variables

    def get_selected_variables(self, vars):
        variables = []
        # if self.path.endswith("argument_list|identifier"):
        #     print()
        if self.path.endswith("identifier") and self.source in vars:
            if self.path.endswith("field_access|identifier"):
                if self.idx == 0:
                    variables += [self]
            elif self.path.endswith("formal_parameter|identifier"):
                pass
            elif self.path.endswith("method_invocation|identifier"):
                if self.idx == 0:
                    variables += [self]
            else:
                variables += [self]
        
        for child in self.children:
            variables += child.get_selected_variables(vars)
        
        return variables
    
    def get_statements(self):
        statements = []
        pattern = re.compile(r'block\|\w+_statement$')
        result = pattern.search(self.path)
        # if self.path.endswith("try_statement|block|expression_statement"):
        #     print()
        if result is not None:
            statements += [self]
        
        for child in self.children:
            statements += child.get_statements()
        
        return statements

    def get_func_name(self):
        results = []
        if self.path == "class_body|method_declaration|identifier":
            results += [self]
        
        for child in self.children:
            results += child.get_func_name()
        
        return results
    
    def get_booleans(self):
        results = []
        if self.type == "true" or self.type == "false":
            results += [self]
        
        for child in self.children:
            results += child.get_booleans()
        
        return results
    
class CPP_AST(P_AST):
    def __init__(self, root, code, idx=0, parent=None, deep=0) -> None:
        self.type = root.type
        self.parent = parent
        if self.parent is not None:
            self.path = self.parent.path + "|" + self.type
        else:
            self.path = self.type

        self.deep = deep
        if self.type == "compound_statement":
            self.deep += 1

        self.start_byte = root.start_byte
        self.end_byte = root.end_byte

        self.source = str(code[self.start_byte:self.end_byte], encoding = "utf-8")
        
        self.idx = idx

        self.children = [CPP_AST(child, code, idx=i, parent=self, deep=self.deep) for i, child in enumerate(root.children)]

    @classmethod
    def build_ast(cls, code, lang="cpp"):
        the_code = bytes(code,'utf8')
        node = parsers[lang].parse(the_code)
        the_ast = CPP_AST(node.root_node, the_code)
        the_ast.link_ast()
        return the_ast

    def get_fields(self):
        fields = []
        
        for child in self.children:
            fields += child.get_fields()
        
        return fields
    
    def get_parameters(self):
        parameters = []
        pattern = re.compile(r'\|parameter_declaration.*\|identifier$')
        result = pattern.search(self.path)
        # if self.source == "ret":
        #     print()
        if result is not None:
            parameters += [self]
        
        for child in self.children:
            parameters += child.get_parameters()
        
        return parameters
    
    def get_selected_parameters(self, vars):
        variables = []
        if self.path.endswith("|identifier") and self.source in vars:
            variables += [self]
        
        for child in self.children:
            variables += child.get_selected_parameters(vars)
        
        return variables

    def get_first_identifier(self):
        if self.type == "identifier":
            return self

        for child in self.children:
            child_i = child.get_first_identifier()
            if child_i is not None:
                return child_i
        
        return None
    
    def get_local_variables(self):
        variables = []
        # if self.source == "u_MAC":
        #     print()
        if self.path.endswith("|declaration"):
            var = self.get_first_identifier()
            if var is not None:
                variables += [var]
        
        for child in self.children:
            variables += child.get_local_variables()
        
        return variables

    def get_selected_variables(self, vars):
        variables = []
        if self.path.endswith("|identifier") and self.source in vars:
            pattern = re.compile(r'\|parameter_declaration.*\|identifier$')
            result = pattern.search(self.path)
            if result is not None:
                pass
            else:
                variables += [self]
        
        for child in self.children:
            variables += child.get_selected_variables(vars)
        
        return variables

    def get_selected_identitier(self, vars):
        variables = []
        if self.path.endswith("|identifier") and self.source in vars:
            variables += [self]
        
        for child in self.children:
            variables += child.get_selected_identitier(vars)
        
        return variables
    
    def get_statements(self):
        statements = []
        pattern = re.compile(r'\|compound_statement\|\w+_statement$')
        result = pattern.search(self.path)
        # if self.path.endswith("try_statement|block|expression_statement"):
        #     print()
        if result is not None:
            statements += [self]
        
        for child in self.children:
            statements += child.get_statements()
        
        return statements

    def get_func_name(self):
        results = []
        if self.path.endswith("function_definition|function_declarator|identifier"):
            results += [self]
        
        for child in self.children:
            results += child.get_func_name()
        
        return results

    def get_booleans(self):
        results = []
        if self.type == "true" or self.type == "false":
            results += [self]
        
        for child in self.children:
            results += child.get_booleans()
        
        return results