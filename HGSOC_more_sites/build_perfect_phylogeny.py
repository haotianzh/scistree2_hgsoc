"""
    Construct a perfect phylogeny (no conflicts) (Dan Gusfield/ https://doi.org/10.1002/net.3230210104)
    Author: haotian Z
    Date: 12/27/22
"""
import numpy as np
import pandas as pd 
import sys 
# sys.path.insert(0, '/home/haz19024/projects/')
import popgen


class BNode(popgen.Node):
    # extended Node class for recording mutations 
    def __init__(self, identifier=None, name=None, branch=0):
        super().__init__(identifier, name, branch)
        self.mutations = []

    def add_mutations(self, mutations):
        self.mutations.extend(mutations)


#  vanilla implementation takes O(nm^2) that simply check every pair of columns to see if they are disjoint or one includes the other one.
def check_no_conflict_vanilla(mat):
    nrow, ncol = mat.shape
    gametes = [[0,1], [1,0], [1,1]]
    for i in range(ncol):
        for j in range(ncol):
            flag = False
            for gamete in gametes:
                # print(gamete, mat[:, [i,j]])
                if gamete not in mat[:, [i,j]].tolist():
                    flag = True
                    break
            if not flag:
                return False
    return True


def remove_homozygous_columns(mat):
    mask = mat.sum(axis=0) > 0
    return mat[:, mask]


# useless one
def binary_number(arr):
    s = 0
    for i, ele in enumerate(arr[::-1]):
        if ele:
            s += 2**i
    return s 


# sort by binary numbers and remove duplicates
def rearrangement(mat):
    nrow, ncol = mat.shape
    binary_strs = []
    for i in range(ncol):
        binary_strs.append(''.join([str(val) for val in mat[:, i]]))
    sorted_index = sorted(range(ncol), key=lambda x:binary_strs[x], reverse=True)
    # build index and remove duplicates
    groups = {}
    group = []
    final_index = []
    for i in range(ncol):
        group.append(sorted_index[i])
        if i==ncol-1 or binary_strs[sorted_index[i]] != binary_strs[sorted_index[i+1]]:
            final_index.append(sorted_index[i])
            groups[sorted_index[i]] = group
            group = []
    return final_index, groups
    

# get a matrix for determining L(j)
def preprocess(mat):
    nrow, ncol = mat.shape
    indices, groups = rearrangement(mat)
    mat_ = mat[:, indices].copy()
    # for loop to check every entity with value 1
    pre_mat = np.zeros_like(mat_)
    for i in range(mat_.shape[0]):
        pre = 0
        for j in range(mat_.shape[1]):
            if mat_[i,j] == 1:
                pre_mat[i,j] = pre
                pre = j+1
    return mat_, pre_mat, indices, groups


#  Gusfield here did some sortings based on binary numbers then make an improvement to O(nm) [n: # of cells, m: # of columns]
def check_no_conflict(mat):
    nrow, ncol = mat.shape
    mat_, L, indices, groups = preprocess(mat)
    for j in range(L.shape[1]):
        maxx = L[:, j].max()
        for i in range(L.shape[0]):
            if mat_[i,j] == 1 and L[i,j] != maxx:
                return False
    return True
                    

# phylogeny construction from Gusfield's paper from a conflict-free matrix
def build_perfect_phylogeny(mat):
    mat = remove_homozygous_columns(mat)
    if not check_no_conflict(mat):
        raise Exception('This are some conflicts in the matrix provided.')
    mat_, L, indices, groups = preprocess(mat)
    L = np.max(L, axis=0)
    # step 1: build mutation tree
    root = BNode(identifier='root')
    mutation_nodes = {}
    for mutation in groups:
        node = BNode()
        # node = BNode(name=str(groups[mutation]))
        node.add_mutations(groups[mutation])
        mutation_nodes[mutation] = node
    for j in range(len(L)):
        if L[j] == 0:
            mutation = indices[j]
            root.add_child(mutation_nodes[mutation])
            mutation_nodes[mutation].set_parent(root)
        else:
            mutation_from = indices[L[j]-1]
            mutation_to = indices[j]
            node_from = mutation_nodes[mutation_from]
            node_to = mutation_nodes[mutation_to]
            node_from.add_child(node_to)
            node_to.set_parent(node_from)

    # step 2: add cells into the mutation tree
    nrow, ncol = mat_.shape
    for i in range(nrow):
        max_index = ncol-np.argmax(mat_[i][::-1])-1
        last_mutation_node = mutation_nodes[indices[max_index]]
        leaf_node = BNode(identifier=i)
        last_mutation_node.add_child(leaf_node)
        leaf_node.set_parent(last_mutation_node)
    
    # phylogeny = popgen.utils.from_node(root)
    # phylogeny.print()
    # step 3: tree prune
    for label in mutation_nodes:
        node = mutation_nodes[label]
        if len(node.get_children()) == 1:
            child = node.get_children()[0]
            node.parent.children.pop(node.identifier)
            node.parent._children.remove(node)
            node.parent.add_child(child)
            child.set_parent(node.parent)
            child.add_mutations(node.mutations)
    phylogeny = popgen.utils.from_node(root)
    return phylogeny



def build_single_file(file):
    df = pd.read_table(file, index_col=0)
    mat = df.values
    tree = build_perfect_phylogeny(mat)
    prefix = '/'.join(file.split('/')[:-1])
    name = file.split('/')[-1].split('.')[0] + '.HUNTRESS.bestTree'
    print(f'{prefix}/{name}')
    with open(f'{prefix}/{name}', 'w') as out:
        out.write(tree.output())
    # print(tree.output())


def test(n, m):
    gt = np.zeros((m, n), dtype=int)
    tree = popgen.utils.get_random_binary_tree(n)
    tree.draw()
    # now randomly assign mutations to the branches
    traversor = popgen.utils.TraversalGenerator()
    total_tree_length = 0
    for node in tree.get_all_nodes():
        node = tree[node]
        if node.is_root():
            continue
        total_tree_length += node.branch

    for i in range(m):
        rand_length = np.random.rand() * total_tree_length
        current_length = 0
        for node in traversor(tree):
            if node.is_root():
                continue
            current_length += node.branch
            if current_length > rand_length:
                print(f'mut{i}: {node.identifier}')
                if node.is_leaf():
                    gt[i, node.identifier] = 1
                else:
                    leaves = [leaf.identifier for leaf in node.get_leaves()]
                    gt[i, leaves] = 1
                break
    return gt, tree

if __name__ == "__main__":

    gt, tree = test(5 ,5)
    print(gt)
    tree2 = build_perfect_phylogeny(gt.T)
    tree2.draw()
    print(tree.get_splits(True, tree[tree.get_leaves()[0]].name))
    print(tree2.get_splits(True, tree[tree.get_leaves()[0]].name))
    print(tree == tree2)
    # import glob
    # for file in glob.glob('/data/sdd/haotian/scistree_test/test-data-3-21/*/*.SC.CFMatrix'):
    #     try:
            
    #         build_single_file(file)
    #         print(f'Success: {file}')
    #     except:
    #         print(f'Failed: {file}')



