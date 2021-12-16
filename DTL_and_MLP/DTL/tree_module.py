# https://stackoverflow.com/questions/2358045/how-can-i-implement-a-tree-in-python with some modifications
import operator
class Tree:
    def __init__(self, data, names=None, children=None):
        self.data = data
        self.children = [] #array of tree
        self.names = [] #array of tree

        self.line_indexes_dict = {}
        self.data_from_file = []

        if children:
            for child in children:
                self.add_child(child)
        if names:
            for name in names:
                self.add_name(name)
        
    def add_child(self, tree):
        self.children.append(tree)

    def add_name(self, name):
        self.names.append(name)

    def export_tree(self, paramSpace=0):
        if (self.data):
            if (len(self.children) > 0):
                print(paramSpace * ' ' + self.data)
            else:
                print(paramSpace * ' ' + 'class: ' + self.data)

        space = paramSpace + 1

        for index in range(0, len(self.children)):
            print(space * ' ' + str(self.names[index]))
            self.children[index].export_tree(space + 2)

    def save_to_file(self, filename):
        f = open(filename, "w+")
        self.save_to_file_helper(f)
        f.close()
        
    def save_to_file_helper(self, f, paramSpace=0):
        if (self.data):
            if (len(self.children) > 0):
                f.write(paramSpace * ' ' + self.data + "\n")
            else:
                f.write(paramSpace * ' ' + 'class: ' + self.data + "\n")
        space = paramSpace + 1
        for index in range(0, len(self.children)):
            f.write(space * ' ' + str(self.names[index]) + "\n")
            self.children[index].save_to_file_helper(f, space + 2)

    def clear_all_attributes(self):
        self.line_indexes_dict.clear()
        self.data_from_file.clear()
        self.children.clear()
        self.names.clear()

    def get_space_count_and_string_without_space(self, str):
        space_count = 0
        while (str[space_count] == ' '):
            space_count += 1
        # Strip "\n" from the end of the string
        str_without_first_spaces = str[space_count:len(str)-1]
        return space_count, str_without_first_spaces

    def load_from_file(self, filename):
        self.clear_all_attributes()
        f = open(filename, "r")
        # If file is in open mode, then proceed
        if (f.mode == "r"):
            # Read per line
            lines = f.readlines()
            for idx_line in range(len(lines)):
                space_count, str_without_first_spaces = self.get_space_count_and_string_without_space(lines[idx_line])
                if (space_count not in self.line_indexes_dict.keys()):
                    indexes_array = []
                    indexes_array.append(idx_line)
                    self.line_indexes_dict[space_count] = indexes_array
                else:
                    self.line_indexes_dict[space_count].append(idx_line)
                self.data_from_file.append(str_without_first_spaces)
        f.close()
        final_tree = self.construct_tree_from_file_data()
        self.data = final_tree.data
        self.children = final_tree.children
        self.names = final_tree.names
        # self.export_tree()
        
    def construct_tree_from_file_data(self, idx_line_indexes_dict=0, idx_indexes_array=0):
        idx_attr_name = self.line_indexes_dict[idx_line_indexes_dict][idx_indexes_array]
        if (idx_indexes_array >= len(self.line_indexes_dict[idx_line_indexes_dict]) - 1):
            next_idx_attr_name = len(self.data_from_file) + 1
        else:
            next_idx_attr_name = self.line_indexes_dict[idx_line_indexes_dict][idx_indexes_array + 1]
        # Attribute name (e.g "class: Iris-setosa", "sepal_length")
        attr_name_from_file = self.data_from_file[idx_attr_name]
        if (attr_name_from_file.startswith("class: ")):
            # Split "class: " and the rest of the string
            attr_name = attr_name_from_file[7:]
            return Tree(attr_name)
        else:
            tree_children = []
            tree_names = []
            for j in range(len(self.line_indexes_dict[idx_line_indexes_dict + 1])):
                idx_attr_value = self.line_indexes_dict[idx_line_indexes_dict + 1][j]
                if (idx_attr_name < idx_attr_value and idx_attr_value < next_idx_attr_name):
                    # attr name
                    attr_value = self.data_from_file[idx_attr_value]
                    tree_names.append(attr_value)
                    tree = self.construct_tree_from_file_data(idx_line_indexes_dict + 3, j)
                    tree_children.append(tree)
            return Tree(attr_name_from_file, tree_names, tree_children)

    

    def getRules(self):
        if (len(self.children)<=0):
            rule = dict()
            rule['class'] = self.data
            return [rule]
        count = 0
        rules = []
        for index in range(len(self.children)):
            temp = self.children[index].getRules()
            for i in range(len(temp)):
                rules.append(temp[i])
                rules[count][self.data] = self.names[index]
                count+=1
        return rules

    def isLeafsParent(self):
        for child in self.children:
            if (len(child.children) > 0):
                return False
        return True

    def prune(self):
        counter = dict()
        for child in self.children:
            if (child.data in counter):
                counter[child.data] += 1
            else:
                counter[child.data] = 1
        if (counter != dict()):
            self.data = max(counter.items(), key=operator.itemgetter(1))[0]
            self.children = []

# tree = Tree("aing cupu")
# tree.load_from_file("Output-ID3")
# print(tree.line_indexes_dict[1][0])
# print(tree.data_from_file[2])
# for i in range (0, 5):
#     print(data_from_file[i])