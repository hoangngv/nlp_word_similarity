from scipy import stats
from util import load_word_dict, load_visim400

if __name__ == "__main__":
    word_dict = load_word_dict()
    visim_results, sim1, sim2 = load_visim400(word_dict)
    
    # Correlation coefficient
    print('Pearson correlation coefficient between our results and Sim1: ', stats.pearsonr(visim_results, sim1))
    print('Pearson correlation coefficient between our results and Sim2: ', stats.pearsonr(visim_results, sim2))
    print('Spearman\'s rank correlation coefficient between our results and Sim1: ', stats.spearmanr(visim_results, sim1))
    print('Spearman\'s rank correlation coefficient between our results and Sim2: ', stats.spearmanr(visim_results, sim2))