import copy
import math
import random

def node_score_error(prob):
    '''
        TODO:
        Calculate the node score using the train error of the subdataset and return it.
        For a dataset with two classes, C(p) = min{p, 1-p}
    '''
    
    # min
    return min(prob, 1 - prob)


def node_score_entropy(prob):
    '''
        TODO:
        Calculate the node score using the entropy of the subdataset and return it.
        For a dataset with 2 classes, C(p) = -p * log(p) - (1-p) * log(1-p)
        For the purposes of this calculation, assume 0*log0 = 0.
        HINT: remember to consider the range of values that p can take!
    '''
    
    # HINT: If p < 0 or p > 1 then entropy = 0
    if prob <= 0 or prob >= 1:
        return 0
    else:
        return -prob * math.log(prob) - (1-prob) * math.log(1-prob)


def node_score_gini(prob):
    '''
        TODO:
        Calculate the node score using the gini index of the subdataset and return it.
        For dataset with 2 classes, C(p) = 2 * p * (1-p)
    '''
    
    gini = 2 * prob * (1 - prob)
    return gini
    
    
class Node:
    '''
    Helper to construct the tree structure.
    '''
    def __init__(self, left=None, right=None, depth=0, index_split_on=0, isleaf=False, label=1):
        self.left = left
        self.right = right
        self.depth = depth
        self.index_split_on = index_split_on
        self.isleaf = isleaf
        self.label = label
        self.info = {} # used for visualization


    def _set_info(self, gain, num_samples):
        '''
        Helper function to add to info attribute.
        You do not need to modify this. 
        '''

        self.info['gain'] = gain
        self.info['num_samples'] = num_samples


class DecisionTree:

    def __init__(self, data, validation_data=None, gain_function=node_score_entropy, max_depth=40):
        self.max_depth = max_depth
        self.root = Node()
        self.gain_function = gain_function

        indices = list(range(1, len(data[0])))

        self._split_recurs(self.root, data, indices)

        # Pruning
        if validation_data is not None:
            self._prune_recurs(self.root, validation_data)


    def predict(self, features):
        '''
        Helper function to predict the label given a row of features.
        You do not need to modify this.
        '''
        return self._predict_recurs(self.root, features)


    def accuracy(self, data):
        '''
        Helper function to calculate the accuracy on the given data.
        You do not need to modify this.
        '''
        return 1 - self.loss(data)


    def loss(self, data):
        '''
        Helper function to calculate the loss on the given data.
        You do not need to modify this.
        '''
        cnt = 0.0
        test_Y = [row[0] for row in data]
        for i in range(len(data)):
            prediction = self.predict(data[i])
            if (prediction != test_Y[i]):
                cnt += 1.0
        return cnt/len(data)


    def _predict_recurs(self, node, row):
        '''
        Helper function to predict the label given a row of features.
        Traverse the tree until leaves to get the label.
        You do not need to modify this.
        '''
        if node.isleaf or node.index_split_on == 0:
            return node.label
        split_index = node.index_split_on
        if not row[split_index]:
            return self._predict_recurs(node.left, row)
        else:
            return self._predict_recurs(node.right, row)


    def _prune_recurs(self, node, validation_data):
        '''
        TODO:
        Prune the tree bottom up recursively. Nothing needs to be returned.
        Do not prune if the node is a leaf.
        Do not prune if the node is non-leaf and has at least one non-leaf child.
        Prune if deleting the node could reduce loss on the validation data.
        
        NOTE:
        This might be slightly different from the pruning described in lecture.
        Here we won't consider pruning a node's parent if we don't prune the node 
        itself (i.e. we will only prune nodes that have two leaves as children.)
        
        HINT: Think about what variables need to be set when pruning a node!
        '''
        
        # Checks if node is not a Leaf
        if not node.isleaf:
            if node.left is not None:
                # TODO: Prune node.left
                self._prune_recurs(node.left, validation_data)
                
            if node.right is not None:
                # TODO: Prune node.right
                self._prune_recurs(node.right, validation_data)
                
            if (node.left.isleaf) and (node.right.isleaf):
                # TODO: Prune node only if loss is reduced
                
                # calculate the loss before pruning
                loss_before = self.loss(validation_data)
                # check the node, no children
                node.isleaf = True
                # calculate the loss after pruning
                loss_after = self.loss(validation_data)
                
                # prune the node if it reduces the loss
                if loss_after < loss_before:
                    node.left = None
                    node.right = None
                    
                else:
                    # keep the node if loss is not reduced
                    node.isleaf = False


    def _is_terminal(self, node, data, indices): 
        '''
        TODO:
        Helper function to determine whether the node should stop splitting.
        Stop the recursion if:
            1. The dataset is empty.
            2. There are no more indices to split on.
            3. All the instances in this dataset belong to the same class
            4. The depth of the node reaches the maximum depth.
        Return:
            - A boolean, True indicating the current node should be a leaf and 
              False if the node is not a leaf.
            - A label, indicating the label of the leaf (or the label the node would 
              be if we were to terminate at that node). If there is no data left, you
              can return either label at random.
        '''
        
        y = [row[0] for row in data]
        isleaf = False
        label = 0

        if len(y) == 0: 
            # TODO: If there is no more data return random label
            isleaf = True
            label = node.label
            
        else:
            # TODO: Check Cases if the node should stop splitting and update label
            # Conditions for stop the recursion
            if len(indices) == 0 or len(set(y)) == 1 or node.depth == self.max_depth :
                # if it satisfy any conditions above, isleaf true, and then back to the majority class
                label = max(set(y), key = y.count)
                isleaf = True
            
            else:
                # if it not satisfy conditions above, labeling to the majority class
                # majority: returns the most common value in the list y
                label = max(set(y), key = y.count)

        return isleaf, label
    
        '''
        Q1: Should the label be a boolean?
        '''


    def _split_recurs(self, node, data, indices):
        '''
        TODO:
        Recursively split the node based on the rows and indices given.
        Nothing needs to be returned.

        1. First use _is_terminal() to check if the node needs to be split.
        2. If so, select the column that has the maximum infomation gain to split on.
        3. Store the label predicted for this node, the split column, and use _set_info()
        to keep track of the gain and the number of datapoints at the split.
        4. Then, split the data based on its value in the selected column.
        The data should be recursively passed to the children.
        '''
        
        y = [row[0] for row in data]
        
        # TODO: Check if node needs to be split
        node.isleaf, node.label = self._is_terminal(node, data, indices)

        if not node.isleaf:
            # TODO: Initialize and Calculate Gain
            max_gain = -1
            best_col = None
            num_samples = len(data)
            
            for col in indices:
                gain = self._calc_gain(data, col, self.gain_function)
            
                if gain > max_gain:
                    max_gain = gain
                    best_col = col

            # TODO: Split the column and use _set_info() 
            node.index_split_on = best_col
            node._set_info(max_gain, num_samples)
            indices.remove(node.index_split_on)

            # TODO: Split the Data and Pass it recursively to the  children
            left_data = []
            right_data = []
            
            for row in data:
                if not row[node.index_split_on]:
                    # 0
                    left_data.append(row)
                else:
                    # 1
                    right_data.append(row)
            
            node.left = Node(depth=node.depth+1)
            node.right = Node(depth=node.depth+1)

            self._split_recurs(node.left, left_data, copy.copy(indices))
            self._split_recurs(node.right, right_data, copy.copy(indices))

       
    def _calc_gain(self, data, split_index, gain_function):
        '''
        TODO:
        Calculate the gain of the proposed splitting and return it.
        Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
        Here the C(p) is the gain_function. For example, if C(p) = min(p, 1-p), this would be
        considering training error gain. Other alternatives are entropy and gini functions.
        '''
        
        y = [row[0] for row in data]
        xi = [row[split_index] for row in data]

        if len(y) != 0 and len(xi) != 0:
            # TODO: Calculate Gain
            p_y1 = sum(y) / len(y)
            p_xi_true = sum(xi) / len(xi)
            p_xi_false = 1 - p_xi_true
            

            
            # true: 1,1
            y1_xi_true = sum([1 for i in range(len(y)) if y[i] == 1 and xi[i] == 1])
            # false: 1,0
            y1_xi_false = sum([1 for i in range(len(y)) if y[i] == 1 and xi[i] == 0])
            
            # true: 1
            p_y1_xi_true = y1_xi_true / xi.count(1) if xi.count(1) != 0 else 0
            # false: 0
            p_y1_xi_false = y1_xi_false / xi.count(0) if xi.count(0) != 0 else 0
            
            # apply gain_function
            # Gain = C(P[y=1]) - P[x_i=True] * C(P[y=1|x_i=True]) - P[x_i=False] * C(P[y=0|x_i=False])
            gain = gain_function(p_y1) - p_xi_true * gain_function(p_y1_xi_true) - p_xi_false * gain_function(p_y1_xi_false)
  
        else:
            gain = 0
        return gain
    

    def print_tree(self):
        '''
        Helper function for tree_visualization.
        Only effective with very shallow trees.
        You do not need to modify this.
        '''
        print('---START PRINT TREE---')
        def print_subtree(node, indent=''):
            if node is None:
                return str("None")
            if node.isleaf:
                return str(node.label)
            else:
                decision = 'split attribute = {:d}; gain = {:f}; number of samples = {:d}'.format(node.index_split_on, node.info['gain'], node.info['num_samples'])
            left = indent + '0 -> '+ print_subtree(node.left, indent + '\t\t')
            right = indent + '1 -> '+ print_subtree(node.right, indent + '\t\t')
            return (decision + '\n' + left + '\n' + right)

        print(print_subtree(self.root))
        print('----END PRINT TREE---')


    def loss_plot_vec(self, data):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        self._loss_plot_recurs(self.root, data, 0)
        loss_vec = []
        q = [self.root]
        num_correct = 0
        while len(q) > 0:
            node = q.pop(0)
            num_correct = num_correct + node.info['curr_num_correct']
            loss_vec.append(num_correct)
            if node.left != None:
                q.append(node.left)
            if node.right != None:
                q.append(node.right)

        return 1 - np.array(loss_vec)/len(data)


    def _loss_plot_recurs(self, node, rows, prev_num_correct):
        '''
        Helper function to visualize the loss when the tree expands.
        You do not need to modify this.
        '''
        labels = [row[0] for row in rows]
        curr_num_correct = labels.count(node.label) - prev_num_correct
        node.info['curr_num_correct'] = curr_num_correct

        if not node.isleaf:
            left_data, right_data = [], []
            left_num_correct, right_num_correct = 0, 0
            for row in rows:
                if not row[node.index_split_on]:
                    left_data.append(row)
                else:
                    right_data.append(row)

            left_labels = [row[0] for row in left_data]
            left_num_correct = left_labels.count(node.label)
            right_labels = [row[0] for row in right_data]
            right_num_correct = right_labels.count(node.label)

            if node.left != None:
                self._loss_plot_recurs(node.left, left_data, left_num_correct)
            if node.right != None:
                self._loss_plot_recurs(node.right, right_data, right_num_correct)
