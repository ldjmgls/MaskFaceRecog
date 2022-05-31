"""
Compute evaluation metrics
"""
from pyeer.eer_info import get_eer_stats
from pyeer.report import generate_eer_report, export_error_rates
from pyeer.plot import plot_eer_stats
import numpy as np
import os


def evaluate_metrics(gscores: list, iscores: list, clf_name: str = 'A', print_results: bool = True) -> dict:
    """
    Evaluates the Equal Error Rate.
    :params gscores: the scores for genuine predictions
    :params iscores: the scores for imposter predictions
    :params clf_name: name of the classifier, used for generating report files
    """
    if not os.path.exists('eval'):
        os.mkdir('eval')

    # Calculating stats for classifier
    stats = get_eer_stats(gscores, iscores)

    # Generating csv report
    generate_eer_report([stats], [clf_name],
                        os.path.join('eval', 'pyeer_report.csv'))

    # Export DET curve
    export_error_rates(stats.fmr, stats.fnmr, os.path.join(
        'eval', f'{clf_name}_DET.csv'))

    # # Export ROC curve
    export_error_rates(stats.fmr, 1 - stats.fnmr,
                       os.path.join('eval', f'{clf_name}_ROC.csv'))

    # Plotting
    plot_eer_stats([stats], [clf_name], save_path='eval')

    result = {
        "EER": stats.eer,
        "AUC": stats.auc,
        "FMR100": stats.fmr100,
        "FMR10": stats.fmr10
    }

    if print_results:
        print(result)

    return result


def _test_eval():
    """
    Tests evaluate_metrics with randomly generated data
    """
    gscores = np.random.uniform(low=0.46, high=0.66, size=100).tolist()
    iscores = np.random.uniform(low=0.13, high=0.53, size=100).tolist()

    evaluate_metrics(
        gscores=gscores, iscores=iscores, clf_name='test', print_results=True)


if __name__ == '__main__':
    _test_eval()
