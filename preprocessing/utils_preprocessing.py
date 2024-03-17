import scipy.spatial.distance
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt
import re



tokenizer = RegexpTokenizer(r'\S+')
exclude = ['!', '¡', '‘', '“', '”', '˜', '’' , '"', 'ÿ', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '.', '/', ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~''«','»', '/', '—', '...', '¿']
exclude = set(exclude)

def preprocess_sentences(doc):
    data_sentences = sent_tokenize(doc)
    cleaned_sentences = [remove_punctuation(sentence) for sentence in data_sentences]
    # join the characters inspected previously
    cleaned_sentences_text = ["".join(tokens).rstrip() for tokens in cleaned_sentences]
    sentences = []
    for sentence in cleaned_sentences_text:
        no_number = re.sub(r'\d+', '[NUM]', sentence)
        sentences.append(no_number)
    sentences = [sublist for sublist in sentences if sublist]
    return sentences

def interleave_lists(list1, list2, result_list):
    """
    Interleaves the titles with their articles
    
    Args:
    - list1 (list): The titles.
    - list2 (list): The the text.
    - result_list (list): The list to store the interleaved strings.
    """
    i = 0
    j = 0
    while i < len(list1) and j < len(list2):
        result_list.append(list1[i])
        result_list.append(list2[j])
        i += 1
        j += 1
    while i < len(list1):
        result_list.append(list1[i])
        i += 1
    while j < len(list2):
        result_list.append(list2[j])
        j += 1

def remove_punctuation(sentence):
    #at a charachter level
    no_punctuation = [char for char in sentence if char not in exclude]
    return no_punctuation

def get_stats(messages):
    messages = [(len(message)) for message in messages]
    print(f"Total sentences: {len(messages)}")
    print(f"Average sentence length: {round(np.mean(messages))}")
    print(f"Minimum sentence length: {min(messages)}")
    print(f"Maximum sentence length: {max(messages)}")
    print(f"Percentile 25, length: {np.percentile(messages, 25)}")
    print(f"Percentile 50, length: {np.percentile(messages, 75)}")


def sentences_into_flat_tokens(resource):
    tokens_flat = []
    tokens_list = []
    for sent in resource:
        tokens = tokenizer.tokenize(sent)
        tokens_flat.extend(tokens)  # Use extend instead of append
        tokens_list.append(tokens)
    return tokens_list, tokens_flat

def plot_custom_boxplot(data, title):
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a single subplot
    box_colors = ["#87CEEB"]
    box = ax.boxplot(
        data,  # Pass the list of integers as a list of lists
        labels=[title],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=box_colors[0], edgecolor="#2E5984", linewidth=1),
        whiskerprops=dict(color="#2E5984", linewidth=1),
        medianprops=dict(color="#2E5984", linewidth=1),
        capprops=dict(color="#2E5984", linewidth=1),
        meanprops=dict(color="#2E5984", linewidth=1),
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_ylabel("Sentence Length (tokens)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)  # Customize grid lines
    
    # Highlight outliers with red color
    for flier in box["fliers"]:
        flier.set(
            marker="o",
            markerfacecolor="#87CEEB",
            markeredgecolor="#2E5984",
            markersize=8,
        )

    # Increase font size for legend
    legend_labels = ["Median", "Mean"]
    ax.legend([box["medians"][0], box["means"][0]], legend_labels, fontsize=10)

    plt.tight_layout()
    plt.show()


def replace_tokens_in_corpus(corpus, token_list1, token_list2):
    for token1, token2 in zip(token_list1, token_list2):
        corpus = re.sub(r'\b{}\b'.format(token1), token2, corpus)
    return corpus


def get_advanced_tokens(tokens_list, tokens_list_flat):
    total_tokens = len(tokens_list_flat)
    print(f"Total number of tokens: {total_tokens}")
    avg_sentence_length = round(np.mean([len(sentence) for sentence in tokens_list]))
    min_sentence_length = min([len(sentence) for sentence in tokens_list])
    max_sentence_length = max([len(sentence) for sentence in tokens_list])
    print(f"Average sentence length (tokens): {avg_sentence_length}")
    print(f"Minimum sentence length (tokens): {min_sentence_length}")
    print(f"Maximum sentence length (tokens): {max_sentence_length}")
    percentile_25 = np.percentile([len(sentence) for sentence in tokens_list], 25)
    percentile_75 = np.percentile([len(sentence) for sentence in tokens_list], 75)
    print(f"Percentile 25, length: {percentile_25}")
    print(f"Percentile 75, length: {percentile_75}")

def plot_custom_boxplots(ham_data, spam_data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    box_colors = ["#87CEEB", "#FFA07A"]
    ham_box = axs[0].boxplot(
        ham_data,
        labels=["Ham"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=box_colors[0], edgecolor="#2E5984", linewidth=1),
        whiskerprops=dict(color="#2E5984", linewidth=1),
        medianprops=dict(color="#2E5984", linewidth=1),
        capprops=dict(color="#2E5984", linewidth=1),
        meanprops=dict(color="#2E5984", linewidth=1),
    )
    axs[0].set_title("Ham SMSes", fontsize=16, fontweight="bold")
    axs[0].set_ylabel("Sentence Length (tokens)", fontsize=12)
    spam_box = axs[1].boxplot(
        spam_data,
        labels=["Spam"],
        showmeans=True,
        meanline=True,
        patch_artist=True,
        boxprops=dict(facecolor=box_colors[1], edgecolor="#A0522D", linewidth=1),
        whiskerprops=dict(color="#A0522D", linewidth=1),
        medianprops=dict(color="#A0522D", linewidth=1),
        capprops=dict(color="#A0522D", linewidth=1),
        meanprops=dict(color="#A0522D", linewidth=1),
    )
    axs[1].set_title("Spam SMSes", fontsize=16, fontweight="bold")
    axs[1].set_ylabel("Sentence Length (tokens)", fontsize=12)
    # Customize grid lines
    for ax in axs:
        ax.grid(True, linestyle="--", alpha=0.7)
        # Highlight outliers with red color
    for box in [ham_box]:
        for flier in box["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="#87CEEB",
                markeredgecolor="#2E5984",
                markersize=8,
            )
    for box in [spam_box]:
        for flier in box["fliers"]:
            flier.set(
                marker="o",
                markerfacecolor="#FFA07A",
                markeredgecolor="#A0522D",
                markersize=8,
            )
        # Increase font size for legend
    legend_labels = ["Median", "Mean"]
    axs[0].legend(
        [ham_box["medians"][0], ham_box["means"][0]], legend_labels, fontsize=10
    )
    axs[1].legend(
        [spam_box["medians"][0], spam_box["means"][0]], legend_labels, fontsize=10
    )
    # Set y-axis ticks for the second subplot based on the first subplot
    # axs[1].set_yticks(axs[0].get_yticks())

    plt.tight_layout()
    plt.show()


def get_analogies(set_analogies_1, set_analogies_2, model_without_subwords):
    
    solution_word_lasolana = []
    results_set_1 = []
    for pair in set_analogies_1:
        example_words = pair[0:3]
        solution_lasolana = pair[-1]
        model_result_1 = model_without_subwords.get_analogies(example_words[0], example_words[1], example_words[2])
        solution_word_lasolana.append(solution_lasolana)
        results_set_1.append(model_result_1[0][1])
        
    solution_word_general = []    
    results_set_2 = []
    for pair in set_analogies_2:
        example_words = pair[0:3]
        solution_general = pair[-1]
        model_result_2 = model_without_subwords.get_analogies(example_words[0], example_words[1], example_words[2])
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
    all_solutions = solution1+ solution2

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

        A_vec1 = scipy.spatial.distance.cosine(vec2, vec3) + scipy.spatial.distance.cosine(vec2,vec4) + scipy.spatial.distance.cosine(vec3,vec4)
        B_vec2 = scipy.spatial.distance.cosine(vec1, vec3) + scipy.spatial.distance.cosine(vec1,vec4) + scipy.spatial.distance.cosine(vec3,vec4)
        C_vec3 = scipy.spatial.distance.cosine(vec1, vec2) + scipy.spatial.distance.cosine(vec1,vec4) + scipy.spatial.distance.cosine(vec2,vec4)
        D_vec4 = scipy.spatial.distance.cosine(vec1, vec2) + scipy.spatial.distance.cosine(vec1,vec3) + scipy.spatial.distance.cosine(vec2,vec3)

        distancias.append([A_vec1, B_vec2, C_vec3, D_vec4])

    return distancias

