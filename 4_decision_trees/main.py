

import matplotlib.pyplot as plt

def loss_plot(ax, title, tree, pruned_tree, train_data, test_data):
    '''
        Example plotting code. This plots four curves: the training and testing
        average loss using tree and pruned tree.
        You do not need to change this code!
        Arguments:
            - ax: A matplotlib Axes instance.
            - title: A title for the graph (string)
            - tree: An unpruned DecisionTree instance
            - pruned_tree: A pruned DecisionTree instance
            - train_data: Training dataset returned from get_data
            - test_data: Test dataset returned from get_data
    '''
    fontsize=8
    ax.plot(tree.loss_plot_vec(train_data), label='train non-pruned')
    ax.plot(tree.loss_plot_vec(test_data), label='test non-pruned')
    ax.plot(pruned_tree.loss_plot_vec(train_data), label='train pruned')
    ax.plot(pruned_tree.loss_plot_vec(test_data), label='test pruned')


    ax.locator_params(nbins=3)
    ax.set_xlabel('number of nodes', fontsize=fontsize)
    ax.set_ylabel('loss', fontsize=fontsize)
    ax.set_title(title, fontsize=fontsize)
    legend = ax.legend(loc='upper center', shadow=True, fontsize=fontsize-2)

def explore_dataset(filename, class_name):
    train_data, validation_data, test_data = get_data(filename, class_name)


    # For each measure of gain (training error, entropy, gini):
    #      (a) Print average training loss (not-pruned)
    #      (b) Print average test loss (not-pruned)
    #      (c) Print average training loss (pruned)
    #      (d) Print average test loss (pruned)

    # TODO: Feel free to print or plot anything you like here. Just comment
    # make sure to comment it out, or put it in a function that isn't called
    # by default when you hand in your code!
    
    gain_fcns = [node_score_error, node_score_entropy, node_score_gini]
    gain_names = ["Training Error", "Entropy", "Gini"]

    for gain_name, gain_fcns in zip(gain_names, gain_fcns):
        print(f"Results for {gain_name}:")

        # not-pruned
        dt = DecisionTree(train_data, gain_function=gain_fcns)
        train_loss = dt.loss(train_data)
        test_loss = dt.loss(test_data)

        print(f"(a) Average Training Loss (not-pruned): {train_loss:.4f}")
        print(f"(b) Average Test Loss (not-pruned): {test_loss:.4f}")

        # pruned
        dt_pruned = DecisionTree(train_data, validation_data=validation_data, gain_function=gain_fcns)
        train_loss_pruned = dt_pruned.loss(train_data)
        test_loss_pruned = dt_pruned.loss(test_data)

        print(f"(c) Average Training Loss (pruned): {train_loss_pruned:.4f}")
        print(f"(d) Average Test Loss (pruned): {test_loss_pruned:.4f}")

        print("\n")


def main():
    random.seed(0)
    np.random.seed(0)

    explore_dataset('data/chess.csv', 'won')
    explore_dataset('data/spam.csv', '1')

main()
