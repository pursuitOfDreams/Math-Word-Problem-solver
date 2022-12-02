import re


OPERATORS = {'+', '-', '*', '/', '(', ')', '^'}
PRIORITY = {'+': 2, '-': 2, '*': 3, '/': 3, '^': 4}


class Node():
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

class Tree():
    def __init__(self):
        self.root = None
        # Used for printing in various orders
        # This data is flushed after every traversal
        self.__ordered_data = []


    def insert(self, value, node="EMPTY_TREE"):
        # Inserts a node to the BinaryTree
        if node == "EMPTY_TREE":
            node = self.root

        if node == None:
            self.root = Node(value)

        else:
            if value < node.data:
                if node.left is None:
                    node.left = Node(value)
                else:
                    self.insert(value, node.left)
            else:
                if node.right is None:
                    node.right = Node(value)
                else:
                    self.insert(value, node.right)
        return 0

    def tree_from_postfix(self, postfix_expression):
        postfix_expression_array = re.findall(r"(\d*\.?\d+|[^0-9])",
                                              postfix_expression)

        elements = []
        for element in postfix_expression_array:
            if not re.match('\s+', element):
                elements.append(element)

        stack = []

        for element in elements:
            if self.__is_operator(element) == True:
                subtree = Node(element)
                subtree.right = stack.pop()
                subtree.left = stack.pop()

                stack.append(subtree)

            else:
                stack.append(Node(element))

        self.root = stack.pop()

    def has_tree(self):
        return not self.root is None

    def inorder(self, node="DEFAULT"):
        # Returns the tree values in inorder
        node = self.__is_root_or_self(node)

        self.__traverse_inorder(node)

        return self.__get_tree()

    def __traverse_inorder(self, node):
        # Traverses the tree from the node provided in inorder
        if not node is None:
            self.__traverse_inorder(node.left)

            self.__ordered_data.append(node.data)

            self.__traverse_inorder(node.right)

    def preorder(self, node="DEFAULT"):
        # Returns the tree values in preorder
        node = self.__is_root_or_self(node)

        self.__traverse_preorder(node)

        return self.__get_tree()

    def __traverse_preorder(self, node):
        # Traverses the tree from the node provided in preorder
        if not node is None:
            self.__ordered_data.append(node.data)

            self.__traverse_preorder(node.left)

            self.__traverse_preorder(node.right)

    def postorder(self, node="DEFAULT"):
        # Returns the tree values in postorder
        node = self.__is_root_or_self(node)

        self.__traverse_postorder(node)

        return self.__get_tree()

    def __traverse_postorder(self, node):
        # Traverses the tree from the node provided in postorder
        if not node is None:
            self.__traverse_postorder(node.left)

            self.__traverse_postorder(node.right)

            self.__ordered_data.append(node.data)

    def levelorder(self, node="DEFAULT"):
        # Traverses the tree from the node provided in levelorder
        node = self.__is_root_or_self(node)

        if not node is None:
            result = []
            current = [node]

            while current:
                next_level = []
                values = []

                for tn in current:
                    values.append(tn.data)

                    if not tn.left is None:
                        next_level.append(tn.left)

                    if not tn.right is None:
                        next_level.append(tn.right)

                result.append(values)

                current = next_level

            return result

        else:
            return []

    def __is_root_or_self(self, node):
        # An easy way to simplify the use of the BinaryTree
        if node == "DEFAULT":
            return self.root
        else:
            return node

    def __flush_data(self):
        # Clear stored data
        self.__ordered_data = []

    def __is_operator(self, value):
        return value == '+' or value == '-' or value == '/' or value == '*' or value == '^'

    def __get_tree(self):
        # Return the stored order of data ie. postorder
        # Erase the previous traversal method
        result = self.__ordered_data

        self.__flush_data()

        return result



class EquationConverter():
    def __init__(self, equation="DEFAULT"):
        self.original_equation = equation
        self.tree = Tree()
        self.equals_what = None

    def show_expression_tree(self):
        print(self.tree.levelorder())

    def expr_as_prefix(self):
        preorder_list = self.tree.preorder()
        prefix_expression = " ".join(preorder_list)

        return f"{self.equals_what} = {prefix_expression}"

    def expr_as_postfix(self):
        postorder_list = self.tree.postorder()
        postfix_expression = " ".join(postorder_list)

        return f"{self.equals_what} = {postfix_expression}"

    def expr_as_infix(self):
        inorder_list = self.tree.inorder()
        infix_expression = " ".join(inorder_list)

        return f"{self.equals_what} = {infix_expression}"

    def eqset(self, equation="DEFAULT"):
        self.original_equation = equation

        self.postfix_expression = self.__get_postfix_from_infix()

        self.__fill_tree()

    def is_eqset(self):
        return len(self.equation) > 0

    def __infix_to_postfix(self):
        filtered_expression, equation_equals = self.__filter_equation(
            self.original_equation)

        self.equals_what = equation_equals

        stack = []
        output = ""

        split_expression = re.findall(r"(\d*\.?\d+|[^0-9])",
                                      filtered_expression)

        for char in split_expression:
            if char not in OPERATORS:
                output += char
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while len(stack)!=0 and stack[-1] != '(':
                    output += ' '
                    output += stack.pop()
                stack.pop()
            else:
                output += ' '

                while len(stack)!=0 and stack[-1] != '(' and PRIORITY[char] <= PRIORITY[stack[-1]]:
                    output += stack.pop()

                stack.append(char)

        while len(stack)!=0:
            output += ' '
            output += stack.pop()

        return output

    def __get_postfix_from_infix(self):
        return self.__infix_to_postfix()

    def __filter_equation(self, equation):
        equation_equals = ""
        # Clean the equation
        try:
            equation_equals = re.search(r"([a-z]+(\s+)?=|=(\s+)?[a-z]+)",
                                        equation).group(1)
            equation_equals = re.sub("=", "", equation_equals)
        except:
            pass

        equation = re.sub(r"([a-z]+(\s+)?=|=(\s+)?[a-z]+)",
                          "", equation)

        return equation.replace(' ', ""), equation_equals.replace(' ', "")

    def __fill_tree(self):
        # Start with the reversed postfix expression
        try:
            self.tree.tree_from_postfix(self.postfix_expression)
        except:
            pass
