import sys
import os
import pandas as pd
import glob
from config_reader import read_config


def parse_lvo(filename):
    with open(filename, "r") as f:
        content = f.read()
        lvo_acc_mean = round(float(content.split()[1]), 4)
        lvo_acc_ci = (
            round(float(content.split()[-2][1:].replace(",", "")), 4),
            round(float(content.split()[-1].replace(")", "")), 4),
        )
    return lvo_acc_mean, str(lvo_acc_ci)


def parse_f1(filename):
    with open(filename, "r") as f:
        content = f.read()
        f1_score_mean = round(float(content.split()[1]), 4)
        f1_score_ci = (
            round(float(content.split()[-2][1:].replace(",", "")), 4),
            round(float(content.split()[-1].replace(")", "")), 4),
        )
    return f1_score_mean, str(f1_score_ci)


def parse_best_params(filename):
    with open(filename, "r") as f:
        content = f.readlines()
        best_params = content[0][len("best parameters: ") :].replace("\n", "")
        best_train = round(float(content[1].split()[-1]), 4)
        best_test = round(float(content[2].split()[-1]), 4)
    return best_params, best_train, best_test


def create_latex_table(data, table_caption, table_label):
    columns = data.columns.tolist()
    num_cols = len(columns)
    latex_str = "\\begin{table}[h!]\n\\centering\n"
    latex_str += "\\begin{tabular}{|l|" + "c|" * (num_cols - 1) + "}\n\\hline\n"
    header_row = (
        " & ".join([f"\\textbf{{{col}}}" for col in columns]) + " \\\\ \\hline\n"
    )
    latex_str += header_row
    for _, row in data.iterrows():
        row_str = " & ".join([str(val) for val in row]) + " \\\\ \\hline\n"
        if row_str.startswith("Best"):
            row_str = (
                row_str.replace("{", "\makecell[l]{\{")
                .replace("}", "\}}")
                .replace(",", ",\\\\")
            )
        latex_str += row_str
    latex_str += "\\end{tabular}\n"
    latex_str += f"\\caption{{{table_caption}}}\n"
    latex_str += f"\\label{{table:{table_label}}}\n"
    latex_str += "\\end{table}\n"
    return latex_str.replace("_", "\_")


def create_table(filepaths):
    table = {}
    table["Metric"] = [
        "LVO Accuracy Mean",
        "LVO Accuracy CI",
        "F1-Score Mean",
        "F1-score CI",
        "Best Parameters",
        "Best Train Score",
        "Best Test Score",
    ]
    for alg in model:
        lvo_acc_mean, lvo_acc_ci = parse_lvo(
            [
                filename
                for filename in filepaths
                if filename.endswith(f"{alg}/lvo_accuracy_score.txt")
            ][0]
        )
        f1_score_mean, f1_score_ci = parse_f1(
            [
                filename
                for filename in filepaths
                if filename.endswith(f"{alg}/F1_score.txt")
            ][0]
        )
        best_params, best_train, best_test = parse_best_params(
            [
                filename
                for filename in filepaths
                if filename.endswith(f"{alg}/best_parameters_rf.txt")
            ][0]
        )
        table[alg] = [
            lvo_acc_mean,
            lvo_acc_ci,
            f1_score_mean,
            f1_score_ci,
            best_params,
            best_train,
            best_test,
        ]
    return pd.DataFrame(table)


def tablesGenerator():
    result_dir = os.path.join(rootDirPath, "results", dataset)
    for dim in dimension:
        dim_dir = os.path.join(result_dir, dim, clip)
        filepaths = []
        for alg in model:
            final_result_dir = os.path.join(dim_dir, alg)
            filepaths += glob.glob(f"{final_result_dir}/*.txt")
        result_df = create_table(filepaths)
        result_df.to_csv(f"{dim_dir}/res_table.csv", index=False)
        latex_table = create_latex_table(
            result_df, f"{dim.title()} Results", f"table:{dim}"
        )
        with open(f"{dim_dir}/res_table.tex", "w") as outf:
            outf.write(latex_table)


if __name__ == "__main__":
    config = read_config()

    # Use the configuration values
    rootDirPath = config["rootDirPath"]
    dataset = config["dataset"]
    dimension = config["dimension"]
    clip = config["clip"]
    model = config["model"]
    task = config["task"]
    modalities = config["modalities"]
    threshold = config["threshold"]
    featureSelection = config["featureSelection"]

    # Example usage in your script
    print(f"Root Directory Path: {rootDirPath}")
    print(f"Dataset: {dataset}")
    print(f"Dimension: {dimension}")
    print(f"Clip: {clip}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    print(f"Modalities: {modalities}")
    print(f"Threshold: {threshold}")
    print(f"Feature Selection: {featureSelection}")

    sys.path.append(rootDirPath)
    tablesGenerator()
