import numpy as np
import time

def main():

    #############################################################
    # These first bits are just to help you develop your code
    # and have expected ouputs given. All asserts should pass.
    ############################################################

    # I made up some random 3-dimensional data and some labels for us
    example_train_x = np.array([ [ 1, 0, 2], [3, -2, 4], [5, -2, 4],
                                 [ 4, 2, 1.5], [3.2, np.pi, 2], [-5, 0, 1]])
    example_train_y = np.array([[0], [1], [1], [1], [0], [1]])


    #########
    # Sanity Check 1: If I query with examples from the training set 
    # and k=1, each point should be its own nearest neighbor
    
    for i in range(len(example_train_x)):
        assert([i] == get_nearest_neighbors(example_train_x, example_train_x[i], 1))
        
    #########
    # Sanity Check 2: See if neighbors are right for some examples (ignoring order)
    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, 4, 2] ), 2)
    assert(set(nn_idx).difference(set([4,3]))==set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 1, -4, 2] ), 3)
    assert(set(nn_idx).difference(set([1,0,2]))==set())

    nn_idx = get_nearest_neighbors(example_train_x, np.array( [ 10, 40, 20] ), 5)
    assert(set(nn_idx).difference(set([4, 3, 0, 2, 1]))==set())

    #########
    # Sanity Check 3: Neighbors for increasing k should be subsets
    query = np.array( [ 10, 40, 20] )
    p_nn_idx = get_nearest_neighbors(example_train_x, query, 1)
    for k in range(2,7):
      nn_idx = get_nearest_neighbors(example_train_x, query, k)
      assert(set(p_nn_idx).issubset(nn_idx))
      p_nn_idx = nn_idx
   
    #########
    # Test out our prediction code
    queries = np.array( [[ 10, 40, 20], [-2, 0, 5], [0,0,0]] )
    pred = predict(example_train_x, example_train_y, queries, 3)
    assert( np.all(pred == np.array([[0],[1],[0]])))

    #########
    # Test our our accuracy code
    true_y = np.array([[0],[1],[2],[1],[1],[0]])
    pred_y = np.array([[5],[1],[0],[0],[1],[0]])                    
    assert( compute_accuracy(true_y, pred_y) == 3/6)

    pred_y = np.array([[5],[1],[2],[0],[1],[0]])                    
    assert( compute_accuracy(true_y, pred_y) == 4/6)


    #######################################
    # Now on to the real data!
    #######################################

    # Load training and test data as numpy matrices 
    train_X, train_y, test_X = load_data()

    #######################################
    # Q9 Hyperparmeter Search
    #######################################

    # Search over possible settings of k
    print("Performing 4-fold cross validation")
    for k in [1,3,5,7,9,99,999,8000]:
      t0 = time.time()

      #######################################
      # TODO Compute train accuracy using whole set
      #######################################
      prediction = predict(train_X, train_y, train_X, k)

      train_acc = compute_accuracy(train_y, np.array(prediction))

      #######################################
      # TODO Compute 4-fold cross validation accuracy
      #######################################
      
      val_acc, val_acc_var = cross_validation(train_X, train_y, 4, k)
      
      t1 = time.time()
      print("k = {:5d} -- train acc = {:.2f}%  val acc = {:.2f}% ({:.4f})\t\t[exe_time = {:.2f}]".format(k, train_acc*100, val_acc*100, val_acc_var*100, t1-t0))
    
    #######################################



    #######################################
    # Q10 Kaggle Submission
    #######################################


    # TODO set your best k value and then run on the test set
    best_k = 10

    # Make predictions on test set
    pred_test_y = predict(train_X, train_y, test_X, best_k)    
    
    # add index and header then save to file
    test_out = np.concatenate((np.expand_dims(np.array(range(2000),dtype=np.int32), axis=1), pred_test_y), axis=1)
    header = np.array([["id", "income"]])
    test_out = np.concatenate((header, test_out))
    np.savetxt('test_predicted.csv', test_out, fmt='%s', delimiter=',')


######################################################################
# Q7 get_nearest_neighbors 
######################################################################
# Finds and returns the index of the k examples nearest to
# the query point. Here, nearest is defined as having the 
# lowest Euclidean distance. This function does the bulk of the
# computation in kNN. As described in the homework, you'll want
# to use efficient computation to get this done. Check out 
# the documentaiton for np.linalg.norm (with axis=1) and broadcasting
# in numpy. 
#
# Input: 
#   example_set --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   idx_of_nearest --   a k-by- list of indices for the nearest k
#                       neighbors of the query point
######################################################################

def get_nearest_neighbors(example_set, query, k):

    distance_list = []

    # Get a list of Euclidean distance
    for i in range(len(example_set)):
        temp_arr = example_set[i] - query
        dist = np.linalg.norm(temp_arr)
        distance_list.append(dist)

    indexed_arr = np.argsort(distance_list)
    idx_of_nearest = indexed_arr[:k]

    # indexed_arr = [(element, index) for index, element in enumerate(distance_list)]

    # Sort the list of tuples by the element values
    # sorted_arr = sorted(indexed_arr)

    # Extract the indices of the k smallest elements
    # idx_of_nearest = [index for (element, index) in sorted_arr[:k]]

    # print(idx_of_nearest)

    return idx_of_nearest


######################################################################
# Q7 knn_classify_point 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   query --    a 1-by-d vector representing a single example
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_label --   either 0 or 1 corresponding to the predicted
#                        class of the query based on the neighbors
######################################################################

def knn_classify_point(examples_X, examples_y, query, k):
    
    idx_nearest_neighbor = get_nearest_neighbors(examples_X, query, k)

    # Get the label list
    labels = [examples_y[i][0] for i in idx_nearest_neighbor]

    # count the most popular label
    label_count = {}
    for label in labels:
        if label in label_count:
            label_count[label] += 1
        else:
            label_count[label] = 1
    
    # print(label_count)
    predicted_label = max(label_count, key = label_count.get)
    # print(predicted_label)

    return predicted_label




######################################################################
# Q8 cross_validation 
######################################################################
# Runs K-fold cross validation on our training data.
#
# Input: 
#   train_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   train_Y --  a n-by-1 vector of example class labels
#
# Output:
#   avg_val_acc --      the average validation accuracy across the folds
#   var_val_acc --      the variance of validation accuracy across the folds
######################################################################

def cross_validation(train_X, train_y, num_folds=4, k=1):

    n = len(train_X)
    chunk_size = n // num_folds

    accuracies = []
    for i in range(num_folds):
        start, end = i * chunk_size, (i + 1) * chunk_size
        val_X, val_y = train_X[start:end], train_y[start:end]
        train_X_fold = np.vstack((train_X[:start], train_X[end:]))
        train_y_fold = np.vstack((train_y[:start], train_y[end:]))

        fold_accuracies = []
        for j in range(len(val_X)):
            predicted_label = knn_classify_point(train_X_fold, train_y_fold, val_X[j], k)
            if predicted_label == val_y[j]:
                fold_accuracies.append(1)
            else:
                fold_accuracies.append(0)

        accuracy = np.mean(fold_accuracies)
        accuracies.append(accuracy)

    avg_val_acc = np.mean(accuracies)
    varr_val_acc = np.var(accuracies)

    return avg_val_acc, varr_val_acc



##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################


######################################################################
# compute_accuracy 
######################################################################
# Runs a kNN classifier on the query point
#
# Input: 
#   true_y --  a n-by-1 vector where each value corresponds to 
#              the true label of an example
#
#   predicted_y --  a n-by-1 vector where each value corresponds
#                to the predicted label of an example
#
# Output:
#   predicted_label --   the fraction of predicted labels that match 
#                        the true labels
######################################################################

def compute_accuracy(true_y, predicted_y):
    accuracy = np.mean(true_y == predicted_y)
    return accuracy

######################################################################
# Runs a kNN classifier on every query in a matrix of queries
#
# Input: 
#   examples_X --  a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   examples_Y --  a n-by-1 vector of example class labels
#
#   queries_X --    a m-by-d matrix representing a set of queries 
#
#   k --        the number of neighbors to return
#
# Output:
#   predicted_y --   a m-by-1 vector of predicted class labels
######################################################################

def predict(examples_X, examples_y, queries_X, k): 
    # For each query, run a knn classifier
    predicted_y = [knn_classify_point(examples_X, examples_y, query, k) for query in queries_X]

    return np.array(predicted_y,dtype=np.int32)[:,np.newaxis]

# Load data
def load_data():
    traindata = np.genfromtxt('train.csv', delimiter=',')[1:, 1:]
    train_X = traindata[:, :-1]
    train_y = traindata[:, -1]
    train_y = train_y[:,np.newaxis]
    
    test_X = np.genfromtxt('test_pub.csv', delimiter=',')[1:, 1:]

    return train_X, train_y, test_X

    
if __name__ == "__main__":
    main()



# data_file = "train.csv"
# count = 0

# with open(data_file, 'r') as file:
#     first_row = file.readline()
#     secdond_row = file.readline()
#     print(type(secdond_row))
#     print(secdond_row.count(','))
#     row = file.readlines()

# print(len(first_row))
# for i in range(len(row)):
#     if (int(row[i][-2]) == 1):
#         count += 1

# print(count)