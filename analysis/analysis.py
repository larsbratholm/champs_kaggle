import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pickle
import cvxpy
import tensorflow as tf
import sklearn.model_selection
import gzip

def parse_rank_and_name(filenames):
    rank = []
    name = []
    for filename in filenames:
        tokens = filename.split("/")[0].split()
        rank.append(int(tokens[0]))
        if len(tokens) >= 1:
            name.append((" ".join(tokens[1:])))
        else:
            name.append("")
    return rank, name

def get_test_data(filename):
    id_to_type = {}
    meta_to_id = {}
    id_to_idx = {}
    c = 0
    if filename.endswith(".gz"):
        f = gzip.open(filename, 'rt', encoding='utf-8')
    else:
        f = open(filename)
    next(f)
    for line in f:
        tokens = line.split(",")
        id_ = int(tokens[0])
        id_to_type[id_] = tokens[-1]
        meta_key = ",".join(tokens[1:]).strip()
        meta_to_id[meta_key] = id_
        id_to_idx[id_] = c
        c += 1
    f.close()
    idx_to_id = {v: k for k, v in id_to_idx.items()}
    type_to_idx = {}
    for id_, type_ in id_to_type.items():
        if type_ not in type_to_idx:
            type_to_idx[type_] = []
        idx = id_to_idx[id_]
        type_to_idx[type_].append(idx)
    return id_to_type, meta_to_id, id_to_idx, type_to_idx

def parse_submissions(filenames, id_to_idx):
    data = np.zeros((len(filenames), 2505542))
    for j, filename in enumerate(filenames):
        with open(filename) as f:
            next(f)
            for i, line in enumerate(f):
                tokens = line.split(",")
                id_ = int(tokens[0])
                idx = id_to_idx[id_]
                data[j,idx] = float(tokens[1])
    return data

def geometric_mean(x):
    return np.exp(np.sum(np.log(x), axis=0) / x.shape[0])

def tf_blend(X, y, type_to_idx):
    #X[1:] = 0
    x = X.T
    n_samples, n_features = x.shape
    n_classes = len(type_to_idx)
    classes = np.zeros((n_samples, n_classes))
    strat = np.zeros(n_samples, dtype=int)
    for i, (type_, idx) in enumerate(type_to_idx.items()):
        classes[idx, i] = 1
        strat[idx] = i


    classes_tf = tf.placeholder(tf.float32, [None, n_classes])

    x_tf = tf.placeholder(tf.float32, [None, n_features])
    logits_init = np.zeros((n_features,1))
    logits_init[:5] = 5
    logits_init[10:] = -5
    logits = tf.Variable(logits_init, dtype=tf.float32)
    W = tf.nn.softmax(logits, axis=0)
    y_pred = tf.matmul(x_tf,W)
    y_tf = tf.placeholder(tf.float32, [None,1])
    abs_diff = tf.abs(y_tf-y_pred)
    class_diff = abs_diff * classes_tf
    cost = tf.reduce_sum(class_diff, axis=0)
    mean_cost = cost / (tf.reduce_sum(classes_tf, axis=0)+1e-9)
    log_cost = tf.math.log(mean_cost+1e-9) / n_classes
    total_cost = tf.reduce_sum(log_cost)

    lr = 0.04
    steps = 4001
    batch_size = 10000
    test_size = 0.50
    train_step = tf.train.AdamOptimizer(lr).minimize(total_cost)

    scores = []
    running_weights = []

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        train, test = sklearn.model_selection.train_test_split(np.arange(n_samples), stratify=strat, test_size=test_size, random_state=42)
        sess.run(init)
        for i in range(steps):
            for batch in np.array_split(train, train.size // batch_size):
                feed = {x_tf: x[batch], y_tf: y[batch,None], classes_tf: classes[batch]}
                sess.run(train_step, feed_dict=feed)
            np.random.shuffle(train)
            if i % 5 == 0:
                weights, ensemble_loss = sess.run([W, log_cost], feed_dict={x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
                running_weights.append(weights)
                print(i, sum(ensemble_loss), weights.squeeze()[:5])
        ensemble_weights = geometric_mean(np.asarray(running_weights)[-10:])
        ensemble_loss = sess.run(log_cost, feed_dict={W: ensemble_weights, x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
        scores.append(ensemble_loss)
        test_weights = np.zeros((n_features,1))
        for i in range(n_features):
            test_weights[:] = 0
            test_weights[i] = 1
            test_loss = sess.run(log_cost, feed_dict={W: test_weights, x_tf: x[test], y_tf: y[test,None], classes_tf: classes[test]})
            scores.append(test_loss)

    return ensemble_weights, np.asarray(scores)

def blend(X, y):
    n_methods = X.shape[0]
    n_samples = y.size

    w = cvxpy.Variable(n_methods)
    A = cvxpy.Constant(np.ones((1, n_methods)))
    # Set objective
    obj_fun = cvxpy.sum_squares(X.T * w - y) / n_samples
    objective = cvxpy.Minimize(obj_fun)
    # Set constraints
    constraints = [w >= 0, cvxpy.sum(w) == 1]
    prob = cvxpy.Problem(objective, constraints)
    result = prob.solve(solver="ECOS")
    return w.value

def get_couplings(filename, meta_to_id, id_to_idx):
    couplings = np.zeros(2505542)
    if filename.endswith(".gz"):
        f = gzip.open(filename, 'rt', encoding='utf-8')
    else:
        f = open(filename)
    next(f)
    for i, line in enumerate(f):
        tokens = line.split(",")
        meta_key = ",".join(tokens[:-1])
        if meta_key not in meta_to_id:
            continue
        id_ = meta_to_id[meta_key]
        idx = id_to_idx[id_]
        couplings[idx] = float(tokens[-1])
    f.close()
    return couplings

def get_score_by_type(data, couplings, type_to_idx):
    n = len(type_to_idx)

    scores = {}

    for type_, idx in type_to_idx.items():
        idx = np.asarray(idx)
        type_data = data[:,idx]
        type_couplings = couplings[idx]
        scores[type_] = np.log(np.mean(abs(type_data - type_couplings[None]), axis=1))
    return scores

def plot_correlation(data, couplings, name, type_to_idx, scores, subset=[0,1,2,3,4,5,11]):
    subset = np.asarray(subset, dtype=int)
    name = np.asarray(name)
    weights = np.exp(len(type_to_idx)*scores)
    difference = data[subset,:]-couplings[None]
    for i, (type_, idx) in enumerate(type_to_idx.items()):
        difference[:,idx] /= weights[i]

    cov = np.corrcoef(difference)
    print(cov[:3,:3])
    print(np.min(cov))
    # Plot heatmap
    sns.heatmap((cov), square=True, linewidths=.25, cbar_kws={"shrink": .5},
                    cmap = sns.diverging_palette(220, 10, as_cmap=True),
                            yticklabels=name[subset],#, xticklabels=name[subset],
                                    center=0.5, vmax=1, vmin=0)
    #plt.xticks(rotation=-45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('corr.png')


if __name__ == "__main__":
    try:
        with open('data.pkl', "rb") as f:
            scores, data, id_to_type, meta_to_id, id_to_idx, type_to_idx, rank, name, filenames, couplings = pickle.load(f)
    except:
        filenames = glob.glob("../data/*/*.csv")
        filenames.sort()
        rank, name = parse_rank_and_name(filenames)
        id_to_type, meta_to_id, id_to_idx, type_to_idx = get_test_data("../dataset/data_test.csv.gz")
        couplings = get_couplings("../dataset/data_test.csv.gz", meta_to_id, id_to_idx)
        data = parse_submissions(filenames, id_to_idx)
        scores = get_score_by_type(data, couplings, type_to_idx)
        with open('data.pkl', "wb") as f:
            pickle.dump((scores, data, id_to_type, meta_to_id, id_to_idx, type_to_idx, rank, name, filenames, couplings),f,-1)

    print(scores.shape, data.shape, couplings.shape)
    quit()

    try:
        with open('analysis.pkl', "rb") as f:
            weights, scores = pickle.load(f)
    except:
        weights, scores = tf_blend(data, couplings, type_to_idx)
        with open('analysis.pkl', "wb") as f:
            pickle.dump((weights, scores), f, -1)

    for type_, idx in type_to_idx.items():
        if type_ != '1JHN\n':
            continue
        d = {type_:np.arange(len(idx))}
        print(type_)
        weights, scores = tf_blend(data[:,idx], couplings[idx], d)
        print("Ensemble:", scores[0])
        for i, name_ in enumerate(name):
            print(name_, weights[i][0], sum(scores[i+1]))
    #plot_correlation(data, couplings, name, type_to_idx, scores[0])



    #idx = np.arange(data.shape[1])
    #n = 10
    #m = 15000
    #log_weights = np.zeros(len(filenames))
    #weights = np.zeros(len(filenames))
    #for i in range(n):
    #    print(i)
    #    np.random.shuffle(idx)
    #    w = blend(data[:,idx[:m]], couplings[idx[:m]])
    #    w[w < 1e-9] = 1e-9
    #    w /= sum(w)
    #    log_weights += np.log(w)
    #    weights += w
    #weights1 = np.exp(log_weights / n)
    #weights1 /= sum(weights1)
    #weights2 = weights / n
    #print(sum(weights1), sum(weights2))
    #for i in range(len(filenames)):
    #    print(weights1[i], weights2[i])

    #TODO get scores per type for top 5, tensorflow linear combination
