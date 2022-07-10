from enum import Enum
from statistics import Histogram, MIN_VAL, MAX_VAL, TableStats
import csv
import pandas as pd
from turtle import left
from sklearn.cluster import KMeans
import numpy as np
from networkx import from_numpy_matrix, connected_components
import range_query as rq


class Operation(Enum):
    CREATE_LEAF = 1
    SPLIT_COLS = 2
    SPLIT_ROWS = 3


class NodeScope:
    """
    NodeScope indicates the scope of the input dataset a node can see in our SPN.
    """

    def __init__(self, row_idxs, col_idxs):
        self.row_idxs = row_idxs
        self.col_idxs = col_idxs

    def n_rows(self):
        return len(self.row_idxs)

    def n_cols(self):
        return len(self.col_idxs)

    def __repr__(self):
        return f'NodeScope{{rows:[{self.row_idxs}), cols:{self.col_idxs}}}'

    @staticmethod
    def full_scope(dataset):
        assert (len(dataset) > 0)  # cannot be empty
        row_idxs = [i for i in range(0, len(dataset))]
        col_idxs = [i for i in range(0, len(dataset[0]))]
        return NodeScope(row_idxs, col_idxs)


class LeafNode():
    """
    LeafNode represents a leaf node in our SPN.
    It uses a histogram to capture data distribution.
    """

    def __init__(self, dataset, col_names, scope):
        assert (scope.n_cols() == 1)
        self.dataset = dataset
        self.col_names = col_names
        self.scope = scope
        # create a histogram over this scope
        col_idx = scope.col_idxs[0]
        vals = []
        for row_idx in scope.row_idxs:
            vals.append(dataset[row_idx][col_idx])
        self.hist = Histogram.construct_from(vals, 5)

    def estimate(self, range_query):
        # YOUR CODE HERE
        # print(range_query.col_left, self.col_names, self.scope.col_idxs)
        col_name = self.col_names[self.scope.col_idxs[0]]
        if col_name in range_query.col_left:
            left_value = range_query.col_left[col_name]
            right_value = range_query.col_right[col_name]
        else:
            left_value = MIN_VAL
            right_value = MAX_VAL
        assert (left_value <= right_value)
        rows = self.hist.between_row_count(left_value, right_value, 0)
        return rows / self.scope.n_rows()

    def debug_print(self, prefix, indent):
        print('%sLeafNode: %s, %s' % (prefix, self.scope, self.hist))


class SumNode():
    """
    SumNode represents a sum node in our SPN.
    """

    def __init__(self, scope, lchild, rchild):
        self.scope = scope
        self.lchild = lchild
        self.rchild = rchild

    def estimate(self, range_query):
        # YOUR CODE HERE
        left_weight = self.lchild.scope.n_rows() / (
            self.lchild.scope.n_rows() + self.rchild.scope.n_rows())
        right_weight = self.rchild.scope.n_rows() / (
            self.lchild.scope.n_rows() + self.rchild.scope.n_rows())
        return self.lchild.estimate(
            range_query) * left_weight + self.rchild.estimate(
                range_query) * right_weight

    def debug_print(self, prefix, indent):
        print('%sSumNode: %s' % (prefix, self.scope))
        self.lchild.debug_print(prefix + indent, indent)
        self.rchild.debug_print(prefix + indent, indent)


class ProductNode():
    """
    ProductNode represents a product node in our SPN.
    """

    def __init__(self, scope, lchild, rchild):
        self.scope = scope
        self.lchild = lchild
        self.rchild = rchild

    def estimate(self, range_query):
        # YOUR CODE HERE
        return self.lchild.estimate(range_query) * self.rchild.estimate(
            range_query)

    def debug_print(self, prefix, indent):
        print('%sProductNode: %s' % (prefix, self.scope))
        self.lchild.debug_print(prefix + indent, indent)
        self.rchild.debug_print(prefix + indent, indent)


class SPN:
    """
    SPN represents a sum-product network.
    """

    def __init__(self, root):
        self.root = root

    def estimate(self, range_query):
        """
        estimate returns the estimated cardinality of the range query on this SPN.
        The input argument range_query is supposed to be a ParsedRangeQuery structure. 
        """
        return self.root.estimate(range_query)

    def debug_print(self, prefix, indent):
        self.root.debug_print(prefix, indent)
        pass

    @staticmethod
    def construct_np_array(dataset, scope):
        """
        construct_np_array constructs a numpy array on the dataset according to the scope.
        """
        row_vals = []
        for row_idx in scope.row_idxs:
            row = []
            for col_idx in scope.col_idxs:
                row.append(dataset[row_idx][col_idx])
            row_vals.append(row)
        return np.array(row_vals)

    @staticmethod
    def split_rows(dataset, scope):
        """
        split_rows splits these rows that specified by dataset and scope into two parts.
        It uses kmeans algorithm to split these rows.
        """
        # YOUR CODE HERE
        #assert (scope.n_rows() > 1)
        np_array = SPN.construct_np_array(dataset, scope)
        quantity = pd.DataFrame(np_array).value_counts()
        left = []
        right = []
        row_idxs = scope.row_idxs
        if len(quantity) > 1:
            kmeans = KMeans(n_clusters=2)
            # new_array = kmeans.fit_transform(np_array)
            result = kmeans.fit_predict(np_array)
            for i in range(len(row_idxs)):
                row_idx = row_idxs[i]
                if result[i] == 0:
                    left.append(row_idx)
                elif result[i] == 1:
                    right.append(row_idx)
        elif len(quantity) == 1:
            left = row_idxs[len(row_idxs) // 2:]
            right = row_idxs[:len(row_idxs) // 2]

        return NodeScope(left,
                         scope.col_idxs), NodeScope(right, scope.col_idxs)

    @staticmethod
    def split_cols(dataset, scope, force):
        """
        split_cols splits these columns that specified by dataset and scope into two parts according to their correlation.
        For simplicity, we use Pearson correlation coefficients to measure their correlation.
        First, we calculate Pearson correlation of each two columns.
        And then we put columns whose correlation are large than a fixed threshold into a group.
        """
        assert (scope.n_cols() > 1)
        # use Pearson correlation coefficients to split cols
        np_array = SPN.construct_np_array(dataset, scope).T
        corr_matrix = np.corrcoef(np_array)
        corr_matrix = np.nan_to_num(corr_matrix)

        threshold = 0.3
        while True:
            edge = corr_matrix.copy()
            edge[edge >= threshold] = True
            edge[edge < threshold] = False
            graph = from_numpy_matrix(edge)
            groups = sorted(connected_components(graph), key=len, reverse=True)
            if len(groups) == 1:
                # all nodes are connected so we cannot split a group of cols in this case
                if force is True:
                    threshold *= 1.5
                    continue
                else:
                    return NodeScope([], []), NodeScope([], [])
            max_group = groups[0]
            l_cols = []
            r_cols = []
            for i in range(scope.n_cols()):
                col = scope.col_idxs[i]
                if i in max_group:
                    l_cols.append(col)
                else:
                    r_cols.append(col)
            return NodeScope(scope.row_idxs,
                             l_cols), NodeScope(scope.row_idxs, r_cols)

    @staticmethod
    def get_next_op(scope, row_batch_threshold, split_col_failed):
        """
        get_next_op returns the next operation to do when constructing a SPN.
        """
        # YOUR CODE HERE: return the next operation
        op = None
        force = False
        if scope.n_cols() == 1:
            if scope.n_rows() < row_batch_threshold:
                op = Operation.CREATE_LEAF
            else:
                op = Operation.SPLIT_ROWS
        elif split_col_failed:
            op = Operation.SPLIT_ROWS
        elif scope.n_rows() < row_batch_threshold:
            op = Operation.SPLIT_COLS
            force = True
        else:
            op = Operation.SPLIT_COLS
        return op, force

    @staticmethod
    def construct_top_down(dataset, col_names, scope, row_batch_threshold):
        """
        construct_top_down constructs a SPN top-down.
        """
        split_col_failed = False
        split_col_force = False
        while True:
            next_op, split_col_force = SPN.get_next_op(scope,
                                                       row_batch_threshold,
                                                       split_col_failed)

            if next_op == Operation.SPLIT_ROWS:
                left_scope, right_scope = SPN.split_rows(dataset, scope)
                lchild = SPN.construct_top_down(dataset, col_names, left_scope,
                                                row_batch_threshold)
                rchild = SPN.construct_top_down(dataset, col_names,
                                                right_scope,
                                                row_batch_threshold)
                return SumNode(scope, lchild, rchild)

            elif next_op == Operation.SPLIT_COLS:
                left_scope, right_scope = SPN.split_cols(
                    dataset, scope, split_col_force)
                if left_scope.n_cols() == 0 or right_scope.n_cols() == 0:
                    split_col_failed = True
                    continue
                lchild = SPN.construct_top_down(dataset, col_names, left_scope,
                                                row_batch_threshold)
                rchild = SPN.construct_top_down(dataset, col_names,
                                                right_scope,
                                                row_batch_threshold)
                return ProductNode(scope, lchild, rchild)

            elif next_op == Operation.CREATE_LEAF:
                return LeafNode(dataset, col_names, scope)

    @staticmethod
    def construct_from_dataset(dataset, col_names, row_batch_threshold):
        """
        construct_from_dataset constructs a SPN from the specified dataset.
        """
        root = SPN.construct_top_down(dataset, col_names,
                                      NodeScope.full_scope(dataset),
                                      row_batch_threshold)
        return SPN(root)

    @staticmethod
    def construct_for_imdb_title(csv_file_path, row_batch_threshold):
        """
        construct_for_imdb_title constructs a SPN from the specified csv file.
        The input csv file is supposed to be sampling data of imdb.title table.
        Not all columns of imdb.title are used to construct the SPN, while only some INT columns 
            are used: kind_id, production_year, imdb_id, episode_of_id, season_nr, episode_nr.
        """
        col_names = [
            'kind_id', 'production_year', 'imdb_id', 'episode_of_id',
            'season_nr', 'episode_nr'
        ]
        col_offsets = [3, 4, 5, 7, 8, 9]
        dataset = []
        with open(csv_file_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                if len(line) != 12:  # imdb.title has 12 columns
                    continue
                row = []
                for offset in col_offsets:
                    if line[offset] == '':
                        row.append(0)
                    else:
                        row.append(int(line[offset]))
                dataset.append(row)
        return SPN.construct_from_dataset(dataset, col_names,
                                          row_batch_threshold)
