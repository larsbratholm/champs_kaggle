import numpy as np
import pickle
import sklearn.model_selection
import os
try:
    # Newer versions
    import tensorflow.compat.v1 as tf
except ImportError:
    # Fallback for older versions
    import tensorflow as tf

def tf_blend(X, y, type_to_idx, lr, steps, do_individual_scores=True):
    """
    Does linear combination of solutions, where the weights
    are positive and sum to 1.
    """
    x = X.T
    n_samples, n_features = x.shape
    n_classes = len(type_to_idx)
    classes = np.zeros((n_samples, n_classes))
    strat = np.zeros(n_samples, dtype=int)
    # Stratify CV by type
    for i, (type_, idx) in enumerate(type_to_idx.items()):
        classes[idx, i] = 1
        strat[idx] = i
    
    # Initialize weights to zero to avoid bias
    logits_init = np.zeros((n_features,1))

    # Reset graph
    tf.reset_default_graph()
    # Tensorflow placeholders and variables
    classes_tf = tf.placeholder(tf.float32, [None, n_classes])
    x_tf = tf.placeholder(tf.float32, [None, n_features])
    logits_tf = tf.Variable(logits_init, dtype=tf.float32)
    W = tf.nn.softmax(logits_tf, axis=0)
    y_pred = tf.matmul(x_tf,W)
    y_tf = tf.placeholder(tf.float32, [None,1])
    abs_diff = tf.abs(y_tf-y_pred)
    class_diff = abs_diff * classes_tf
    cost = tf.reduce_sum(class_diff, axis=0)
    mean_cost = cost / (tf.reduce_sum(classes_tf, axis=0)+1e-9)
    log_cost = tf.math.log(mean_cost+1e-9) / n_classes
    total_cost = tf.reduce_sum(log_cost)

    test_size = 0.50
    train_step = tf.train.AdamOptimizer(lr).minimize(total_cost)

    # Keep track of progress
    scores = []
    running_weights = []
    running_logits = []

    # Get stratified train, test split
    train, test = sklearn.model_selection.train_test_split(
        np.arange(n_samples), stratify=strat, test_size=test_size,
        shuffle=True, random_state=42)

    batch_size = train.size/10

    average_steps = 50 if steps > 100 else 20


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for i in range(steps):
            for batch in np.array_split(train, train.size // batch_size):
                feed = {x_tf: x[batch], y_tf: y[batch,None], classes_tf: classes[batch]}
                sess.run(train_step, feed_dict=feed)
            np.random.shuffle(train)
            if i > 0:
                weights, ensemble_loss, logits = sess.run([W, log_cost, logits_tf], feed_dict={x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
                running_logits.append(logits)
                running_weights.append(weights)
                #print(i, sum(ensemble_loss), weights.squeeze()[:5])
        # Average logits and get score and weights
        ensemble_logits = np.mean(running_logits[-average_steps:], 0)
        ensemble_loss, ensemble_weights = sess.run([log_cost, W], feed_dict={logits_tf: ensemble_logits, x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
        scores.append(ensemble_loss)
        # Do individual scores
        if do_individual_scores:
            test_weights = np.zeros((n_features,1))
            # Calculate loss for individual solutions
            for i in range(n_features):
                test_weights[:] = 0
                test_weights[i] = 1
                test_loss = sess.run(log_cost, feed_dict={W: test_weights, x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
                scores.append(test_loss)

    return ensemble_weights, np.asarray(scores)

def write_scores(ranks, names, scores, types, filename):
    with open(filename, "w") as f:
        # Write header
        f.write("rank, name")
        for type_ in types:
            f.write(f", score({type_})")
        f.write(", scores(total)\n")
        for i, rank in enumerate(ranks):
            f.write(f"{ranks[i]}, {names[i]}")
            for score in scores[i]:
                f.write(f", {score:.3f}")
            f.write(f", {sum(scores[i]):.3f}\n")

def write_weights(ranks, names, weights, filename, types=None, by_type=False):
    with open(filename, "w") as f:
        # Write header
        f.write("rank, name")
        if by_type:
            for type_ in types:
                f.write(f", weight({type_})")
        else:
            f.write(", weight")
        f.write("\n")
        for i, rank in enumerate(ranks):
            f.write(f"{ranks[i]}, {names[i]}")
            if by_type:
                for weight in weights[i]:
                    f.write(f", {weight:.3f}")
            else:
                f.write(f", {weights[i][0]:.3f}")
            f.write("\n")


if __name__ == "__main__":
    # Get script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(script_dir + '/data/data.pkl', "rb") as f:
            scores, data, id_to_type, id_to_idx, type_to_idx, rank, name, filenames, couplings = pickle.load(f)
    except FileNotFoundError:
        print("No data pickle found")
        raise SystemExit

    try:
        with open(script_dir + '/data/analysis.pkl', "rb") as f:
            weights, blend_scores = pickle.load(f)
    except:
        weights, blend_scores = tf_blend(data, couplings, type_to_idx, lr=0.32, steps=200)
        with open(script_dir + '/data/analysis.pkl', "wb") as f:
            pickle.dump((weights, blend_scores), f, -1)

    try:
        with open(script_dir + '/data/analysis_individual.pkl', "rb") as f:
            individual_weights, individual_scores = pickle.load(f)
    except:
        individual_weights, individual_scores = {}, {}
        for type_, idx in type_to_idx.items():
            #if type_ != '1JHN\n':
            #    continue
            d = {type_:np.arange(len(idx))}
            individual_weights[type_], individual_scores[type_] = tf_blend(data[:,idx], couplings[idx], 
                    d, lr=0.32, steps=200, do_individual_scores=False)
        with open(script_dir + '/data/analysis_individual.pkl', "wb") as f:
            pickle.dump((individual_weights, individual_scores), f, -1)

    ## Write scores of the two ensembling strategies
    write_scores(["E*", "E"] + list(rank), ["Individual Ensemble", "Single Ensemble"] + list(name), 
            [[score[0][0]/8 for score in individual_scores.values()]] + list(blend_scores),
            types=type_to_idx.keys(), filename=script_dir + '/output/scores.csv')

    # Write weights of the two ensembling strategies
    write_weights(rank, name, weights, filename=script_dir + '/output/single_ensemble_weights.csv')
    write_weights(rank, name, [[weight[i][0] for weight in individual_weights.values()] for i in range(len(rank))],
        by_type=True, types=type_to_idx.keys(), filename=script_dir + '/output/individual_ensemble_weights.csv')
