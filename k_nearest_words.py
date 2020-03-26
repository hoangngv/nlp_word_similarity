from util import load_word_dict, most_similar_words

if __name__ == "__main__":
    # Input
    word = input("Enter word: ")
    k = input("Enter k: ")
    word = word.strip()
    k = int(k.strip())

    # Result
    word_dict = load_word_dict()
    most_similar_words(word, k, word_dict)