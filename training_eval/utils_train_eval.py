import scipy.spatial.distance
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns



def get_analogies(set_analogies_1, set_analogies_2, model_without_subwords):

    solution_word_lasolana = []
    results_set_1 = []
    for pair in set_analogies_1:
        example_words = pair[0:3]
        solution_lasolana = pair[-1]
        model_result_1 = model_without_subwords.get_analogies(
            example_words[0], example_words[1], example_words[2]
        )
        solution_word_lasolana.append(solution_lasolana)
        results_set_1.append(model_result_1[0][1])

    solution_word_general = []
    results_set_2 = []
    for pair in set_analogies_2:
        example_words = pair[0:3]
        solution_general = pair[-1]
        model_result_2 = model_without_subwords.get_analogies(
            example_words[0], example_words[1], example_words[2]
        )
        solution_word_general.append(solution_general)
        results_set_2.append(model_result_2[0][1])

    return results_set_1, solution_word_lasolana, results_set_2, solution_word_general


def compute_accuracy(predictions, actual):
    correct_count = 0
    total_count = len(predictions)

    for pred, act in zip(predictions, actual):
        if pred == act:
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    return accuracy


def joint_accuracy(pred1, solution1, pred2, solution2):
    all_results = pred1 + pred2
    all_solutions = solution1 + solution2

    # Compute the joint accuracy
    joint_accuracy = compute_accuracy(all_results, all_solutions)
    return joint_accuracy


def compute_vector_distance(list_of_words, embeddings_dict):
    distancias = []
    for word in list_of_words:
        vec1 = embeddings_dict[word[0]]
        vec2 = embeddings_dict[word[1]]
        vec3 = embeddings_dict[word[2]]
        vec4 = embeddings_dict[word[3]]

        A_vec1 = (
            scipy.spatial.distance.cosine(vec2, vec3)
            + scipy.spatial.distance.cosine(vec2, vec4)
            + scipy.spatial.distance.cosine(vec3, vec4)
        )
        B_vec2 = (
            scipy.spatial.distance.cosine(vec1, vec3)
            + scipy.spatial.distance.cosine(vec1, vec4)
            + scipy.spatial.distance.cosine(vec3, vec4)
        )
        C_vec3 = (
            scipy.spatial.distance.cosine(vec1, vec2)
            + scipy.spatial.distance.cosine(vec1, vec4)
            + scipy.spatial.distance.cosine(vec2, vec4)
        )
        D_vec4 = (
            scipy.spatial.distance.cosine(vec1, vec2)
            + scipy.spatial.distance.cosine(vec1, vec3)
            + scipy.spatial.distance.cosine(vec2, vec3)
        )

        distancias.append([A_vec1, B_vec2, C_vec3, D_vec4])

    return distancias


def get_acc_RandomForestClassifier(train, y_train, test, y_test):
    dt_clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_split=15,
        min_samples_leaf=10,
        max_depth=30,
        random_state=40,
        class_weight="balanced_subsample",
    )
    dt_clf.fit(train, y_train)
    predictions = dt_clf.predict(test)
    accuracy = accuracy_score(y_test, predictions)
    return predictions, accuracy


def plot_conf_matrix(conf_matrix):
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="flare",
        linewidth=0.5,
        xticklabels=[
            "cultura",
            "deporte",
            "economía",
            "educación",
            "política",
            "sociedad",
            "sucesos",
        ],
        yticklabels=[
            "cultura",
            "deporte",
            "economía",
            "educación",
            "política",
            "sociedad",
            "sucesos",
        ],
        cbar=False,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.show()
